"""Daemon orchestrator (THIN). Loop + logging only. NO trading logic here.

Per-cycle responsibilities (wired Session E):
1. ctx.attach_tracker(tracker) → ctx.refresh() — market + regime + tracker passthrough
2. personas.analyze() per pair → JudgePersona.decide() → JUDGE decision per pair
3. bandit decision-gate — scale JUDGE conf by the (pair×dir×regime) prob-of-profit;
   warm arms can veto, cold arms pass through (gap-8 cold-start safety)
4. per Strategy: exit/hold open lanes FIRST, then entry. PARTIAL_CLOSE → close 50%
   + move SL→entry (daemon-side, latched once per position)
5. risk.accept() — un-disableable fee floor + learnable sizing (position_pct)
6. exec.place() / exec.close() — WEEX V3 algoOrder
7. logger — ONE exp record per evaluated cycle-cell + one per close (new v6.1 schema)

The daemon owns NO trading rules. Every threshold, gate, or per-pair quirk lives
in smt/pairs/<pair>.py or in the personas. This is the "trader-as-config"
boundary V6.0 conflated.

External pollers (kept in v4/):
- drawdown_guardian (max loss in rolling window → pause_trading flag)
- gcp_budget_pause (GCS flag-file on 90% GCP budget breach)
- hot_reload (60s TTL on v4/smt_settings.json — fee mult / pause flags)

It's SIMULATED WEEX funds — the constraint is the deadline, not safety
(CLAUDE.md lesson #10). Experiment to the fullest.
"""

from __future__ import annotations
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from smt.context.global_context import GlobalContext, PAIRS
from smt.core.execution import ExecutionClient, strip_lane_suffix
from smt.core.risk import RiskGate, DEFAULT_POSITION_PCT
from smt.core.tracker import PositionTracker
from smt.core import experience as exp
from smt.pairs.base import Strategy
from smt.pairs.btc import BTCStrategy
from smt.pairs.eth import ETHStrategy
from smt.pairs.bnb import BNBStrategy
from smt.pairs.ltc import LTCStrategy
from smt.pairs.sol import SOLStrategy
from smt.pairs.xrp import XRPStrategy
from smt.pairs.ada import ADAStrategy
from smt.pairs.doge import DOGEStrategy
from smt.personas.base import JUDGE_CONF_FLOOR, PAIR_RAW_JUDGE_FLOOR, JudgeDecision, bare_pair
from smt.personas.judge import JudgePersona
from smt.personas.flow import FlowPersona
from smt.personas.technical import TechnicalPersona
from smt.personas.whale import WhalePersona
from smt.personas.onchain import OnChainPersona
from smt.personas.sentiment import SentimentPersona
from smt.personas.regime import RegimePersona
from smt.learning.bandit import ContextualBandit
from smt.learning.optimizer import load_learned_params
from smt.learning import corpus

log = logging.getLogger("smt.daemon")


STRATEGIES: List[Strategy] = [
    BTCStrategy(),
    ETHStrategy(),
    BNBStrategy(),
    LTCStrategy(),
    SOLStrategy(),
    XRPStrategy(),
    ADAStrategy(),
    DOGEStrategy(),
]

# The 6 voting personas (JUDGE is the aggregator, not a voter). The onchain (Oc)
# slot is logged in the new exp schema — the gap the V6.0 trade log never had.
PERSONAS = [
    FlowPersona(),
    TechnicalPersona(),
    WhalePersona(),
    OnChainPersona(),
    SentimentPersona(),
    RegimePersona(),
]

DEFAULT_MAX_POSITIONS = 5   # gap-7 portfolio capacity (learnable; V3.1.73 lift was 3→5)


class Daemon:
    """The orchestration loop. One instance per process.

    Dependencies are injectable so tests drive a full cycle with mocked
    exec/tracker and a tmp exp dir — no live WEEX, no real corpus IO.
    """

    def __init__(
        self,
        cycle_seconds: int = 120,
        mode: str = "UAT",
        ctx: Optional[GlobalContext] = None,
        exec_client: Optional[Any] = None,
        tracker: Optional[Any] = None,
        bandit: Optional[ContextualBandit] = None,
        personas: Optional[List[Any]] = None,
        strategies: Optional[List[Strategy]] = None,
        exp_dir: str = exp.EXP_DIR,
        state_dir: str = "v4",
        seed_corpus: bool = True,
        learned_params: Optional[Dict[str, Any]] = None,
    ):
        self.cycle_seconds = cycle_seconds
        self.mode = mode                       # "UAT" or "CMT" — kept separate downstream
        self.ctx = ctx if ctx is not None else GlobalContext()
        self.exec = exec_client if exec_client is not None else ExecutionClient()
        self.tracker = tracker if tracker is not None else PositionTracker()
        self.bandit = bandit if bandit is not None else ContextualBandit()
        self.personas = personas if personas is not None else PERSONAS
        self.strategies = strategies if strategies is not None else STRATEGIES
        self.judge = JudgePersona()
        self.exp_dir = exp_dir
        self.state_dir = state_dir
        self.learned_params = learned_params
        self.risk = RiskGate(position_pct=self._learned_position_pct())
        self._startup(seed_corpus=seed_corpus, skip_load=(learned_params is not None))

    # ── startup: learned params + bandit cold-start seeding (gap 8) ────────────

    def _startup(self, seed_corpus: bool = True, skip_load: bool = False) -> None:
        if not skip_load:
            self.learned_params = load_learned_params()
            self.risk = RiskGate(position_pct=self._learned_position_pct())
        n = 0
        if seed_corpus:
            try:
                records = corpus.load_best_corpus()
                n = self.bandit.seed_from_corpus(records) if records else 0
            except Exception as e:
                log.warning("[DAEMON] corpus seeding failed: %s", e)
        if self.learned_params is None:
            log.info("[DAEMON] cold start — no learned_params, bandit seeded from %d outcomes", n)
        else:
            log.info("[DAEMON] warm start — learned_params loaded, bandit seeded from %d outcomes", n)

    def _learned_position_pct(self) -> float:
        if self.learned_params:
            pct = self.learned_params.get("position_pct")
            if pct:
                return float(pct)
        return DEFAULT_POSITION_PCT

    def _max_positions(self) -> int:
        if self.learned_params:
            port = self.learned_params.get("portfolio") or {}
            if port.get("max_positions"):
                return int(port["max_positions"])
        return DEFAULT_MAX_POSITIONS

    def _judge_floor(self, pair: str) -> float:
        if self.learned_params and self.learned_params.get("raw_judge_min_confidence"):
            return float(self.learned_params["raw_judge_min_confidence"])
        return PAIR_RAW_JUDGE_FLOOR.get(bare_pair(pair), JUDGE_CONF_FLOOR)

    # ── bandit decision-gate ───────────────────────────────────────────────────

    def _bandit_gate(
        self, decision: JudgeDecision, pair: str, regime: str,
    ) -> Tuple[JudgeDecision, float, float, bool]:
        """Scale JUDGE conf by the arm's prob-of-profit; warm arms may veto.

        Returns (gated_decision, raw_conf, scaled_conf, veto). Cold arms are
        pass-through (scaled == raw, no veto) so a fresh restart can't silently
        veto every entry (gap 8). A warm arm whose scaled conf falls below the
        JUDGE floor downgrades the action to WAIT.
        """
        raw = float(decision.confidence)
        if decision.action not in ("LONG", "SHORT"):
            return decision, raw, raw, False
        direction = decision.action
        scaled = self.bandit.scaled_confidence(raw, pair, direction, regime)
        floor = self._judge_floor(pair)
        veto = self.bandit.is_warm(pair, direction, regime) and scaled < floor
        gated = JudgeDecision(
            action="WAIT" if veto else decision.action,
            confidence=scaled,
            reasoning=decision.reasoning + (
                f" | bandit veto (scaled {scaled:.2f} < {floor:.2f})" if veto
                else f" | bandit scaled {scaled:.2f}"),
            persona_breakdown=decision.persona_breakdown,
            lane_hint=decision.lane_hint,
        )
        return gated, raw, scaled, veto

    # ── the cycle ───────────────────────────────────────────────────────────────

    def cycle(self) -> None:
        # 1. context refresh (market + tracker passthrough + derived prices)
        self.ctx.attach_tracker(self.tracker)
        self.ctx.refresh()
        cd = self.ctx.as_dict()
        self._fill_missing_prices(cd)

        # 2 + 3. personas → JUDGE → bandit gate, per pair (store into ctx.judge)
        votes_by_pair: Dict[str, Dict[str, Any]] = {}
        meta: Dict[str, Tuple[float, float, bool]] = {}
        regimes = cd.get("regime") or {}
        for pair in PAIRS:
            votes = JudgePersona.votes_from_personas(self.personas, pair, cd)
            votes_by_pair[pair] = votes
            decision = self.judge.decide(pair, votes, cd)
            regime = regimes.get(pair, "NORMAL")
            gated, raw, scaled, veto = self._bandit_gate(decision, pair, regime)
            cd["judge"][pair] = gated
            meta[pair] = (raw, scaled, veto)

        entries = exits = exp_written = 0
        equity = float(cd.get("equity_usd") or self.ctx.equity_usd)

        # 4. per strategy: exits/holds on open lanes FIRST, then entry
        for strat in self.strategies:
            sym = strat.pair

            for lane, position in list(self.tracker.lanes_for_pair(sym).items()):
                key = f"{strip_lane_suffix(sym)}#{lane}"
                decision = strat.exit_signal(position, cd)
                if decision is None:
                    strat.hold_signal(position, cd)   # fire hold (adjust no-op today)
                    continue
                if decision.should_exit:
                    if self._do_close(key, position, cd, decision.reason):
                        exits += 1
                        exp_written += 1
                elif "PARTIAL_CLOSE" in (decision.reason or ""):
                    self._do_partial_close(key, position, cd)

            # entry — strategy reads the bandit-gated JUDGE decision from ctx.judge
            plan = strat.entry_signal(cd)
            raw, scaled, veto = meta.get(sym, (0.0, 0.0, False))
            gated = cd["judge"].get(sym)
            executed = False
            tracker_key: Optional[str] = None
            if plan is not None and self._capacity_ok():
                sized = self.risk.accept(plan, equity)
                if sized is not None:
                    resp = self.exec.place(sized)
                    if resp is None or resp.get("executed", True):
                        self.tracker.add(sized, resp or {})
                        tracker_key = f"{strip_lane_suffix(sized.pair)}#{sized.lane}"
                        executed = True
                        entries += 1
            elif plan is not None:
                log.info("[DAEMON] %s entry skipped — capacity %d reached",
                         sym, self._max_positions())

            rec = self._make_eval_record(sym, cd, votes_by_pair[sym], gated, plan,
                                         executed, tracker_key, raw, scaled, veto)
            exp.write_record(rec, self.exp_dir)
            exp_written += 1
            log.info("[WHY] %s %s", sym, rec["reasoning"])   # XAI: per-decision "why"
            if executed and tracker_key:
                pos = self.tracker.get(tracker_key)
                if isinstance(pos, dict):
                    pos["exp_entry"] = rec   # tracker persists it for the close join

        log.info("[DAEMON] cycle done pairs=%d entries=%d exits=%d exp_written=%d",
                 len(self.strategies), entries, exits, exp_written)
        try:
            self.bandit.save(os.path.join(self.state_dir, "bandit_state.json"))
        except Exception as e:
            log.debug("[DAEMON] bandit save skipped: %s", e)

    # ── close + partial-close handlers (daemon-side) ───────────────────────────

    def _do_close(self, key: str, position: Dict[str, Any], cd: Dict[str, Any],
                  reason: str) -> bool:
        sym = strip_lane_suffix(position.get("pair") or key)
        side = position.get("side", "LONG")
        price = float((cd.get("prices") or {}).get(sym)
                      or position.get("entry_price") or 0.0)
        try:
            self.exec.close(sym, "SELL" if side == "LONG" else "BUY")
        except Exception as e:
            log.warning("[DAEMON] exec.close(%s) error: %s", sym, e)
        closed = self.tracker.close(
            key, fill_price=price, fill_qty=float(position.get("qty") or 0.0), reason=reason)
        if not closed:
            return False
        pnl_usd = float(closed.get("pnl_usd") or 0.0)
        rec = exp.build_close_record(
            entry_ctx=self._entry_ctx_for_close(position, cd),
            pnl_usd=pnl_usd,
            pnl_pct=float(closed.get("pnl_pct") or 0.0),
            exit_reason=reason,
            hours_open=float(closed.get("hours_open") or 0.0),
            win=pnl_usd > 0.0,
        )
        exp.write_record(rec, self.exp_dir)
        return True

    def _do_partial_close(self, key: str, position: Dict[str, Any], cd: Dict[str, Any]) -> None:
        if position.get("partial_close_done"):
            return  # latch — fires exactly once
        sym = strip_lane_suffix(position.get("pair") or key)
        side = position.get("side", "LONG")
        lane = position.get("entry_lane", "slow")
        qty = float(position.get("qty") or 0.0)
        half = qty / 2.0
        entry = float(position.get("entry_price") or 0.0)
        tp = float(position.get("exit_target") or 0.0) or None
        pos_side = side if lane == "bigwick" else "BOTH"
        try:
            self.exec.close_partial(sym, side, half, pos_side)
            self.exec.move_stop_to_entry(sym, side, qty - half, entry, tp, pos_side)
        except Exception as e:
            log.warning("[DAEMON] partial-close exec error %s: %s", sym, e)
        position["partial_close_done"] = True            # in-place latch (this process)
        self.tracker.mark_partial_close(key, new_qty=qty - half, new_stop=entry)
        log.info("[DAEMON] PARTIAL_CLOSE %s closed 50%% (%.6f) SL→entry %.6f", key, half, entry)

    # ── exp-record assembly ─────────────────────────────────────────────────────

    def _make_eval_record(
        self, sym: str, cd: Dict[str, Any], votes: Dict[str, Any],
        gated: Optional[JudgeDecision], plan: Optional[Any],
        executed: bool, tracker_key: Optional[str],
        raw: float, scaled: float, veto: bool,
    ) -> Dict[str, Any]:
        if plan is not None:
            lane, direction, action = plan.lane, plan.direction, plan.direction
        else:
            action = getattr(gated, "action", "WAIT") if gated else "WAIT"
            direction = action if action in ("LONG", "SHORT") else ""
            lane = (getattr(gated, "lane_hint", "") if gated and action in ("LONG", "SHORT") else "")
        price = float((cd.get("prices") or {}).get(sym) or 0.0)
        return exp.build_eval_record(
            pair=sym, mode=self.mode, lane=lane or "", direction=direction, action=action,
            conviction=raw, conviction_scaled=scaled,
            fear_greed=cd.get("fear_greed"), btc_dominance=cd.get("btc_dominance_pct"),
            regime=(cd.get("regime") or {}).get(sym, "NORMAL"),
            entry_price=price, persona_votes=exp.serialize_votes(votes),
            executed=executed, tracker_key=tracker_key,
            judge_action=getattr(gated, "action", action) if gated else action,
            bandit_veto=veto,
            reasoning=self._compose_reasoning(votes, gated, raw, scaled, veto, plan),
        )

    @staticmethod
    def _compose_reasoning(votes, gated, raw: float, scaled: float, veto: bool, plan) -> str:
        """Human-readable ≤500-char 'why' for every decision (XAI — logged + stored).

        Decomposes into: action + conviction (raw→bandit-scaled), the non-NEUTRAL persona
        votes that drove it, any bandit veto, the executed lane, and the JUDGE's own note
        (HARD-BLOCK / capitulation / floor). This is the per-trade explanation retail sees.
        """
        action = getattr(gated, "action", "WAIT") if gated else "WAIT"
        contribs = []
        for name in ("flow", "technical", "whale", "onchain", "sentiment", "regime"):
            v = votes.get(name) if votes else None
            if v is not None and getattr(v, "direction", "NEUTRAL") != "NEUTRAL":
                contribs.append(f"{name[:2].upper()} {v.direction[0]}{int(round(v.confidence * 100))}%")
        votes_str = " ".join(contribs) if contribs else "all NEUTRAL"
        lane = f" [{plan.lane} {plan.direction}]" if plan is not None else ""
        veto_s = " VETO" if veto else ""
        jr = (getattr(gated, "reasoning", "") if gated else "") or ""
        why = f"{action} {raw:.2f}->{scaled:.2f}{veto_s}{lane} | {votes_str}" + (f" | {jr}" if jr else "")
        return why[:exp.REASONING_MAX_CHARS]

    def _entry_ctx_for_close(self, position: Dict[str, Any], cd: Dict[str, Any]) -> Dict[str, Any]:
        """The entry-context record for a closing position.

        Prefer the per-position `exp_entry` stashed at entry (full fidelity);
        fall back to reconstructing from the stored position + current context
        so positions opened before this code (or in tests) still produce a
        complete close record.
        """
        ec = position.get("exp_entry")
        if isinstance(ec, dict):
            return dict(ec)
        sym = strip_lane_suffix(position.get("pair") or "")
        side = position.get("side", "")
        conf = float(position.get("confidence") or 0.0)
        return exp.build_eval_record(
            pair=position.get("pair") or sym, mode=self.mode,
            lane=position.get("entry_lane", ""), direction=side, action=side or "WAIT",
            conviction=conf, conviction_scaled=conf,
            fear_greed=cd.get("fear_greed"), btc_dominance=cd.get("btc_dominance_pct"),
            regime=(cd.get("regime") or {}).get(sym, "NORMAL"),
            entry_price=float(position.get("entry_price") or 0.0),
            persona_votes=exp.votes_from_breakdown(position.get("persona_votes")),
            executed=True, tracker_key=position.get("tracker_key"),
        )

    # ── helpers ─────────────────────────────────────────────────────────────────

    def _capacity_ok(self) -> bool:
        return len(self.tracker.all()) < self._max_positions()

    def _fill_missing_prices(self, cd: Dict[str, Any]) -> None:
        """Fill any pair missing a price via exec.get_price (skips on 429 -1.0).

        A no-op when refresh() already derived prices from klines (the test +
        live-kline path). Pairs that still have no price are left out so the
        strategies / exit cascade skip them (they guard price<=0).
        """
        prices = cd.setdefault("prices", {})
        for strat in self.strategies:
            s = strat.pair
            if prices.get(s):
                continue
            try:
                px = self.exec.get_price(s)
                if isinstance(px, (int, float)) and px > 0:
                    prices[s] = float(px)
            except Exception:
                pass

    def run(self) -> None:
        log.info(
            "Daemon starting (cycle=%ds, strategies=%d, mode=%s)",
            self.cycle_seconds, len(self.strategies), self.mode,
        )
        while True:
            try:
                self.cycle()
            except Exception as e:
                log.exception("Cycle error: %s", e)
            time.sleep(self.cycle_seconds)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    Daemon().run()
