"""Strategy ABC: per-pair behavioral contract + shared helpers.

Each pair implements its own subclass (smt/pairs/btc.py etc.). The
daemon loops over the 8 instances per cycle and calls
entry_signal / exit_signal / hold_signal — that's the entire surface.

The pair Strategy is responsible for:
- per-pair regime read + entry logic (drawing on personas + context)
- per-pair exit rules (when to flatten, when to trail)
- per-pair hold/adjust logic (nudge TP/SL on open positions)

It is NOT responsible for execution, persistence, fee accounting, or
risk sizing — those live in smt/core/.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from smt.core.exit_cascade import ExitCascade
from smt.core.risk import ROUND_TRIP_FEE_PCT
from smt.core.trade_plan import TradePlan, ExitDecision, HoldDecision

_cascade = ExitCascade()


class Strategy(ABC):
    """Per-pair trading strategy. One subclass per WEEX pair."""

    pair: str = ""
    CONFIG: Dict[str, Any] = {}

    @abstractmethod
    def entry_signal(self, context: Dict[str, Any]) -> Optional[TradePlan]:
        """Return TradePlan if conditions warrant a new position, else None."""
        ...

    @abstractmethod
    def exit_signal(
        self, position: Dict[str, Any], context: Dict[str, Any]
    ) -> Optional[ExitDecision]:
        """Return ExitDecision if we should close, else None."""
        ...

    @abstractmethod
    def hold_signal(
        self, position: Dict[str, Any], context: Dict[str, Any]
    ) -> HoldDecision:
        """Always non-None — keep/adjust for an open position."""
        ...

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _pair_name(self) -> str:
        """Short name for config lookups. 'BTCUSDT' → 'BTC'."""
        return self.pair.replace("USDT", "").replace("usdt", "").upper()

    def _compute_atr(self, klines: Dict[str, Any], period: int = 14) -> float:
        """14-period ATR as % of close from 1h klines. Falls back to CONFIG ref."""
        sym = self.pair
        candles: List = []
        for key in (f"{sym}#1h", f"{sym}_1h", f"{sym}:1h"):
            c = klines.get(key) or []
            if len(c) >= period + 1:
                candles = c
                break
        if not candles:
            return float(self.CONFIG.get("ref_atr_1h_pct", 0.60))
        trs = []
        for i in range(1, min(len(candles), period + 1)):
            try:
                h, l, pc = float(candles[i][2]), float(candles[i][3]), float(candles[i - 1][4])
                trs.append(max(h - l, abs(h - pc), abs(l - pc)))
            except (IndexError, ValueError):
                pass
        if not trs:
            return float(self.CONFIG.get("ref_atr_1h_pct", 0.60))
        close = float(candles[-1][4]) if candles else 0.0
        return (sum(trs) / len(trs) / close * 100.0) if close > 0 else float(self.CONFIG.get("ref_atr_1h_pct", 0.60))

    def _bigwick_direction(self, klines: Dict[str, Any]) -> Optional[str]:
        """Detect wick-up (→ SHORT) or wick-down (→ LONG) on last closed candle.

        Wick-up: upper shadow > wick_thresh × range → buying exhaustion → SHORT.
        Wick-down: lower shadow > wick_thresh × range → selling exhaustion → LONG.
        Returns "LONG", "SHORT", or None.
        """
        bw = self.CONFIG.get("bigwick", {})
        interval = bw.get("interval", "1h")
        wick_thresh = float(bw.get("wick_thresh", 0.60))
        min_range_pct = float(bw.get("min_range_pct", 0.40))
        sym = self.pair
        candles: List = []
        for key in (f"{sym}#{interval}", f"{sym}_{interval}", f"{sym}:{interval}"):
            c = klines.get(key) or []
            if len(c) >= 2:
                candles = c
                break
        if not candles:
            return None
        last = candles[-2]  # last CLOSED candle
        try:
            o, h, l, c = float(last[1]), float(last[2]), float(last[3]), float(last[4])
        except (IndexError, ValueError):
            return None
        rng = h - l
        if rng == 0:
            return None
        mid = (o + c) / 2.0 or 1.0
        if rng / mid * 100.0 < min_range_pct:
            return None
        if (h - max(o, c)) / rng >= wick_thresh:
            return "SHORT"
        if (min(o, c) - l) / rng >= wick_thresh:
            return "LONG"
        return None

    def _build_entry_plan(
        self,
        lane: str,
        direction: str,
        price: float,
        tp_pct: float,
        sl_pct: float,
        hold_max_h: float,
        leverage: int,
        confidence: float = 0.0,
        persona_votes: Optional[Dict] = None,
        equity_usd: float = 40_000.0,
    ) -> Optional[TradePlan]:
        """Construct TradePlan. Returns None if fee floor fails.

        # TODO Session D: position_pct learnable via smt.learning.optimizer
        """
        if price <= 0 or tp_pct <= 0 or sl_pct <= 0:
            return None
        position_pct = 0.02  # TODO Session D: learnable via smt.learning.optimizer
        notional_usd = equity_usd * position_pct * leverage
        qty = notional_usd / price
        if direction == "LONG":
            exit_target = price * (1.0 + tp_pct / 100.0)
            exit_stop = price * (1.0 - sl_pct / 100.0)
        else:
            exit_target = price * (1.0 - tp_pct / 100.0)
            exit_stop = price * (1.0 + sl_pct / 100.0)
        est_fees = notional_usd * ROUND_TRIP_FEE_PCT
        est_profit_net = notional_usd * tp_pct / 100.0 - est_fees
        if est_profit_net <= est_fees:
            return None
        return TradePlan(
            pair=self.pair,
            lane=lane,  # type: ignore[arg-type]
            direction=direction,  # type: ignore[arg-type]
            entry_price=price,
            exit_target=exit_target,
            exit_stop=exit_stop,
            hold_max=hold_max_h,
            qty=qty,
            leverage=leverage,
            est_fees=est_fees,
            est_profit_net=est_profit_net,
            est_time_hours=hold_max_h * 0.5,
            decision_confidence=confidence,
            persona_votes=persona_votes or {},
        )

    def _exit_via_cascade(
        self, position: Dict[str, Any], context: Dict[str, Any]
    ) -> Optional[ExitDecision]:
        return _cascade.evaluate(position, context, self._pair_name())

    def _hold_default(self, position: Dict[str, Any], context: Dict[str, Any]) -> HoldDecision:
        return HoldDecision(should_hold=True, reasoning="hold — exit cascade monitoring")

    # ── Session C: JUDGE-driven fast-lane entry ──────────────────────────────

    def _judge_fast_entry(
        self, context: Dict[str, Any], price: float,
    ) -> Optional[TradePlan]:
        """Build a fast-lane TradePlan when JUDGE says LONG/SHORT.

        Reads context["judge"][pair] (a JudgeDecision). Returns None if no
        decision is present, action != LONG/SHORT, or fee floor fails.

        Slow-lane routing requires research-data feeds (weekly ETF inflow,
        DefiLlama TVL Δ, etc.) that arrive in Session E. Until then JUDGE
        routes to fast; Session E will pick lane via JudgeDecision.lane_hint
        and the per-pair slow trigger evaluation.
        """
        judge_map = context.get("judge") or {}
        decision = judge_map.get(self.pair) or judge_map.get(self._pair_name())
        if not decision:
            return None
        action = getattr(decision, "action", None) or (
            decision.get("action") if isinstance(decision, dict) else None
        )
        if action not in ("LONG", "SHORT"):
            return None
        conf = float(getattr(decision, "confidence", None)
                     if not isinstance(decision, dict)
                     else decision.get("confidence", 0.0))
        breakdown = getattr(decision, "persona_breakdown", None) or (
            decision.get("persona_breakdown") if isinstance(decision, dict) else {}
        ) or {}
        lane_hint = getattr(decision, "lane_hint", None) or (
            decision.get("lane_hint") if isinstance(decision, dict) else "fast"
        ) or "fast"
        # Session C: only fast is wired; bigwick is already live and slow needs
        # research-data feeds. Pin lane_hint=slow → fast until Session E.
        lane = "slow" if lane_hint == "slow" else "fast"

        # Per-direction leverage (Session B per-direction knob).
        lev_key = f"{action.lower()}_leverage"
        leverage = int(self.CONFIG.get(lev_key, self.CONFIG.get("base_leverage", 10)))

        # Fast-lane TP = pair tp_cap; SL = tighter end of sl_range.
        tp_pct = float(self.CONFIG["tp_cap_pct"])
        sl_range = self.CONFIG.get("sl_range_pct", (0.40, 1.00))
        sl_pct = float(sl_range[0] if isinstance(sl_range, (list, tuple)) and sl_range else 0.50)
        if lane == "slow":
            hold_max_h = float(self.CONFIG.get("slow_max_hold_h", 4.0))
        else:
            hold_max_h = float(self.CONFIG.get("fast_max_hold_h", 0.5))

        return self._build_entry_plan(
            lane=lane,
            direction=action,
            price=price,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            hold_max_h=hold_max_h,
            leverage=leverage,
            confidence=conf,
            persona_votes=breakdown,
            equity_usd=float(context.get("equity_usd") or 40_000.0),
        )
