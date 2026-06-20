"""Session E acceptance: Daemon.cycle() end-to-end + new exp schema.

Verifies (no real network — exec mocked, tracker local-tmp, corpus seeding off):
1. Daemon.cycle() runs once on a synthetic context; personas/judge/strategies/
   risk/exec(mock)/tracker all fire (e2e).
2. One paper cycle writes a valid exp_*.jsonl record carrying EVERY required
   field with correct types — incl all persona votes (Oc/onchain), on-close
   pnl/win (e2e).
3. ctx.attach_tracker + ctx.refresh() populate open_positions + prices
   (integration).
4. Bandit decision-gate invoked + conf-scaling applied on a warm arm
   (integration); a warm low-prob arm vetoes (integration).
5. PARTIAL_CLOSE: a position over the trigger closes 50% + moves SL→entry
   exactly once; partial_close_done latches (e2e).
6. Cold start logs + seeds the bandit, never trusting un-warm arms (unit).
"""

from __future__ import annotations
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from smt.daemon import Daemon
from smt.context.global_context import GlobalContext
from smt.core.tracker import PositionTracker
from smt.core.trade_plan import TradePlan
from smt.core.experience import REQUIRED_FIELDS, PERSONA_NAMES
from smt.learning.bandit import ContextualBandit


# ── candle / context helpers ──────────────────────────────────────────────────

def _c(o, h, l, c, v=1000) -> List:
    return [0, str(o), str(h), str(l), str(c), str(v)]


NEUTRAL = [_c(50000, 50100, 49900, 50050)] * 4   # no wick → bigwick abstains; close=50050


def _mock_exec() -> MagicMock:
    ex = MagicMock(name="ExecutionClient")
    ex.place.return_value = {"executed": True, "sl_ok": True, "tp_ok": True}
    ex.close.return_value = {"closed": True}
    ex.close_partial.return_value = {"closed_partial": True}
    ex.move_stop_to_entry.return_value = {"sl_ok": True, "tp_ok": True}
    ex.get_price.return_value = 0.0   # never used — refresh derives prices from klines
    return ex


def _tracker(tmp_path) -> PositionTracker:
    """Real tracker (local tmp state) with spies on the mutating methods so we
    keep real behavior (latch, pnl) AND can assert the daemon called them."""
    t = PositionTracker(state_path=str(tmp_path / "state.json"))
    t.add = MagicMock(wraps=t.add)
    t.close = MagicMock(wraps=t.close)
    t.mark_partial_close = MagicMock(wraps=t.mark_partial_close)
    return t


def _daemon(tmp_path, ctx, tracker=None, bandit=None, **kw) -> Daemon:
    return Daemon(
        ctx=ctx,
        exec_client=_mock_exec(),
        tracker=tracker if tracker is not None else _tracker(tmp_path),
        bandit=bandit if bandit is not None else ContextualBandit(seed=1),
        exp_dir=str(tmp_path / "rl"),
        state_dir=str(tmp_path),
        seed_corpus=False,
        **kw,
    )


def _ctx_with_btc_long(eth_price: float = None) -> GlobalContext:
    """Context where personas vote BTC LONG (flow+tech injected) → JUDGE LONG."""
    ctx = GlobalContext()
    ctx.klines = {"BTCUSDT#1h": NEUTRAL, "BTCUSDT_1h": NEUTRAL}
    ctx.flow_signal = {"BTCUSDT": {"direction": "LONG", "confidence": 0.90}}
    ctx.technical_signal = {"BTCUSDT": {"direction": "LONG", "confidence": 0.80}}
    if eth_price is not None:
        ctx.prices = {"ETHUSDT": eth_price}
    return ctx


def _read_exp(tmp_path) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    for f in (tmp_path / "rl").glob("exp_*.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))
    return recs


def _open_position(pair, side, lane, entry, qty, opened_minutes_ago, **extra) -> Dict[str, Any]:
    pos = {
        "pair": pair, "side": side, "entry_lane": lane, "entry_price": entry,
        "exit_target": entry * (1.04 if side == "LONG" else 0.96),
        "exit_stop": entry * (0.99 if side == "LONG" else 1.01),
        "qty": qty, "leverage": 20, "confidence": 0.70,
        "persona_votes": {"flow": 0.36, "technical": 0.24},
        "opened_at": (datetime.now(timezone.utc) - timedelta(minutes=opened_minutes_ago)).isoformat(),
        "peak_pnl_pct": 0.0, "partial_close_done": False,
        "_against_consec": 0, "_drop_consec": 0, "_np_consec": 0, "_cons_consec": 0,
        "tracker_key": f"{pair}#{lane}",
    }
    pos.update(extra)
    return pos


# ── 3. ctx.attach_tracker + refresh populate open_positions + prices ──────────

def test_context_attach_tracker_populates_positions_and_prices(tmp_path):
    tracker = PositionTracker(state_path=str(tmp_path / "state.json"))
    tracker.positions["BTCUSDT#fast"] = _open_position("BTCUSDT", "LONG", "fast", 50000.0, 0.1, 5)
    ctx = GlobalContext()
    ctx.klines = {"BTCUSDT#1h": NEUTRAL}
    ctx.attach_tracker(tracker)
    ctx.refresh()
    assert "BTCUSDT#fast" in ctx.open_positions          # tracker passthrough
    assert ctx.open_positions == tracker.all()
    assert ctx.prices.get("BTCUSDT") == pytest.approx(50050.0)   # derived from kline close


# ── 1 + 2. Full cycle: all components fire + valid exp record (e2e) ───────────

def test_daemon_cycle_runs_without_network_and_fires_all(tmp_path):
    tracker = _tracker(tmp_path)
    # Pre-existing ETH position closes via MAX_HOLD (opened 8h ago, fast lane).
    tracker.positions["ETHUSDT#fast"] = _open_position(
        "ETHUSDT", "LONG", "fast", 3000.0, 0.5, opened_minutes_ago=8 * 60)
    ctx = _ctx_with_btc_long(eth_price=3005.0)   # +0.17% < partial trigger → MAX_HOLD wins
    daemon = _daemon(tmp_path, ctx, tracker=tracker)

    daemon.cycle()

    # exec(mock) + risk + tracker all fired
    assert daemon.exec.place.called, "BTC entry must place an order"
    placed = daemon.exec.place.call_args.args[0]
    assert isinstance(placed, TradePlan) and placed.qty > 0     # risk sized it
    assert placed.est_profit_net > placed.est_fees              # fee floor honored
    assert tracker.add.called
    assert daemon.exec.close.called, "ETH MAX_HOLD must close the position"
    assert tracker.close.called

    recs = _read_exp(tmp_path)
    # personas/judge/strategies fired for all 8 pairs → 8 eval records + 1 close
    evals = [r for r in recs if r["event"] == "eval"]
    closes = [r for r in recs if r["event"] == "close"]
    assert len(evals) == 8
    assert len(closes) == 1

    btc = next(r for r in evals if r["pair"] == "BTCUSDT")
    assert btc["executed"] is True
    assert btc["action"] == "LONG" and btc["direction"] == "LONG"
    assert btc["lane"] == "fast"
    assert btc["conviction"] > 0.55
    assert btc["persona_votes"]["flow"]["signal"] == "LONG"
    # XAI: a populated, capped "why" decomposing the decision
    assert isinstance(btc["reasoning"], str) and 0 < len(btc["reasoning"]) <= 500
    assert "LONG" in btc["reasoning"] and "FL" in btc["reasoning"]


def test_daemon_cycle_writes_valid_exp_record_all_required_fields(tmp_path):
    tracker = _tracker(tmp_path)
    tracker.positions["ETHUSDT#fast"] = _open_position(
        "ETHUSDT", "LONG", "fast", 3000.0, 0.5, opened_minutes_ago=8 * 60)
    ctx = _ctx_with_btc_long(eth_price=3005.0)
    daemon = _daemon(tmp_path, ctx, tracker=tracker)
    daemon.cycle()

    recs = _read_exp(tmp_path)
    close = next(r for r in recs if r["event"] == "close")   # carries entry + on-close fields

    # Every REQUIRED field present on the self-contained close record.
    for field in REQUIRED_FIELDS:
        assert field in close, f"missing required exp field: {field}"

    # Persona votes: all personas incl onchain (Oc) — the V6.0 logging gap.
    pv = close["persona_votes"]
    assert isinstance(pv, dict)
    for name in PERSONA_NAMES:
        assert name in pv, f"persona {name} missing from exp"
        assert set(pv[name]) >= {"signal", "confidence"}
        assert isinstance(pv[name]["confidence"], (int, float))
    assert "onchain" in pv

    # Types on the entry-side fields.
    assert isinstance(close["lane"], str) and close["lane"] in ("fast", "bigwick", "slow")
    assert isinstance(close["conviction"], (int, float))
    assert isinstance(close["fear_greed"], int)
    assert isinstance(close["btc_dominance"], float)
    assert isinstance(close["regime"], str)
    assert isinstance(close["direction"], str)
    assert close["action"] in ("LONG", "SHORT", "WAIT", "BLOCK")
    assert isinstance(close["entry_price"], float) and close["entry_price"] > 0
    assert isinstance(close["ts"], str)
    assert close["mode"] in ("UAT", "CMT")

    # Types on the on-close fields.
    assert isinstance(close["pnl_usd"], float)
    assert isinstance(close["pnl_pct"], float)
    assert isinstance(close["exit_reason"], str) and "MAX_HOLD" in close["exit_reason"]
    assert isinstance(close["hours_open"], float)
    assert isinstance(close["win"], bool)
    assert isinstance(close["reasoning"], str) and len(close["reasoning"]) <= 500


# ── 4. Bandit decision-gate: warm arm scales conviction (integration) ─────────

def test_daemon_bandit_gate_warm_arm_scales_confidence(tmp_path):
    bandit = ContextualBandit(seed=1)
    for _ in range(40):                       # warm + high prob-of-profit (≈0.976)
        bandit.update("BTCUSDT", "LONG", "NORMAL", True)
    assert bandit.is_warm("BTCUSDT", "LONG", "NORMAL")

    ctx = _ctx_with_btc_long()
    daemon = _daemon(tmp_path, ctx, bandit=bandit)
    daemon.cycle()

    btc = next(r for r in _read_exp(tmp_path) if r["pair"] == "BTCUSDT" and r["event"] == "eval")
    prob = bandit.prob_of_profit("BTCUSDT", "LONG", "NORMAL")
    assert btc["conviction_scaled"] == pytest.approx(btc["conviction"] * prob)
    assert btc["conviction_scaled"] < btc["conviction"]   # scaling applied
    assert btc["bandit_veto"] is False                    # still cleared the floor
    assert btc["executed"] is True


def test_daemon_bandit_gate_warm_low_prob_vetoes(tmp_path):
    bandit = ContextualBandit(seed=1)
    for _ in range(40):                       # warm + low prob-of-profit (≈0.05)
        bandit.update("BTCUSDT", "LONG", "NORMAL", False)
    ctx = _ctx_with_btc_long()
    daemon = _daemon(tmp_path, ctx, bandit=bandit)
    daemon.cycle()

    btc = next(r for r in _read_exp(tmp_path) if r["pair"] == "BTCUSDT" and r["event"] == "eval")
    assert btc["bandit_veto"] is True
    assert btc["action"] == "WAIT"
    assert btc["executed"] is False
    assert not daemon.exec.place.called      # vetoed → no order


# ── 5. PARTIAL_CLOSE path: 50% close + SL→entry, fires once (e2e) ─────────────

def test_daemon_partial_close_fires_once_and_latches(tmp_path):
    tracker = _tracker(tmp_path)
    # SOL slow position +0.6% → over the 0.45% partial trigger.
    tracker.positions["SOLUSDT#slow"] = _open_position(
        "SOLUSDT", "LONG", "slow", 100.0, 10.0, opened_minutes_ago=30, exit_target=106.0)
    ctx = GlobalContext()
    ctx.prices = {"SOLUSDT": 100.6}
    daemon = _daemon(tmp_path, ctx, tracker=tracker)

    daemon.cycle()
    daemon.cycle()   # second cycle must NOT re-fire the partial (latch)

    assert daemon.exec.close_partial.call_count == 1
    assert daemon.exec.move_stop_to_entry.call_count == 1
    assert tracker.mark_partial_close.call_count == 1
    pos = tracker.positions["SOLUSDT#slow"]
    assert pos["partial_close_done"] is True
    assert pos["qty"] == pytest.approx(5.0)         # halved
    assert pos["exit_stop"] == pytest.approx(100.0)  # SL moved to entry
    assert not daemon.exec.close.called             # partial ≠ full close


# ── 6. Cold start: logs + seeds, never auto-trusts un-warm arms (unit) ────────

def test_daemon_cold_start_no_learned_params(tmp_path, caplog):
    import logging
    with caplog.at_level(logging.INFO, logger="smt.daemon"):
        daemon = _daemon(tmp_path, GlobalContext())
    assert daemon.learned_params is None
    assert any("cold start" in r.getMessage() for r in caplog.records)
    # A cold (unseeded) arm must be pass-through, never a silent veto (gap 8).
    assert daemon.bandit.scaled_confidence(0.80, "BTCUSDT", "LONG", "NORMAL") == pytest.approx(0.80)


def test_daemon_uses_default_position_pct_on_cold_start(tmp_path):
    daemon = _daemon(tmp_path, GlobalContext())
    from smt.core.risk import DEFAULT_POSITION_PCT
    assert daemon.risk.position_pct == pytest.approx(DEFAULT_POSITION_PCT)
