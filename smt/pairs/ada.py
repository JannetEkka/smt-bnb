"""ADA strategy — ground truth: docs/research/ada.md

Lanes:
- FAST: macro policy headline + liquidity sweep wick; horizon 1-24h.
- BIGWICK: 1H candle wick rejection (wick > 65% range); bidirectional; 0.5-2.5h.
  ADA hard_block_long_bearish applies: TRENDING_DOWN/CRASH blocks LONG entry.
- SLOW: Koios DRep-count 7d Δ ≥+15% + stealth accumulation; horizon 3mo-1yr.

Per-pair config: TP 5.0% · SL 0.60-1.80% · ADX 14 · thin · slow 7.0h /
fast 1.5h · min 45/15min · leverage 15× · ref 1H ATR 0.60%.

Patterns (docs/research/ada.md):
- "Whale-First" frontrunning — Whale activity precedes every regime shift.
  Whales sell during "Extreme Greed"; whales accumulate during "Boredom".
- Sentiment is CONTRA — peaks mark tops; boredom marks bottoms.
- "Liquidity Sweep" — recurring 20%+ wick shorts (thin book).
- Macro-sensitive — moves on US policy / regulatory headlines,
  not native tech.
- Current state (Feb 2026 →): stealth accumulation mimicking pre-Nov-2024
  pre-breakout phase (whale address count rising while flat sentiment).

V6.0.7 audit: ADA LONG BEARISH HARD-BLOCK (n=31 WR=7.1%); ADA WHALE -17% edge.

TODO Session B: port from archive/v6.0/v4/smt_nightly_trade_v3_1.py
- V6.0.6 SLOW-HOLD: Koios /drep_list 7d DRep-count Δ ≥+15% (public, no key).
- V6.0.6 B.5+ HYBRID + B.11 ADA flush bucket (thresholds in
  PAIR_PLAYBOOK_FLUSH dict).
- V6.0.7 PAIR_DIR_REGIME_HARD_BLOCK ADA LONG BEARISH preserved.
- V6.0.7 consolidation_exit_min ADA = 9999 (effectively SKIP — multi-hour
  ranges normal in thin books).
"""

from __future__ import annotations
from typing import Any, Dict, Optional

from smt.pairs.base import Strategy
from smt.core.trade_plan import TradePlan, ExitDecision, HoldDecision


class ADAStrategy(Strategy):
    pair = "ADAUSDT"
    CONFIG: Dict[str, Any] = {
        "tp_cap_pct": 5.0,
        "sl_range_pct": (0.60, 1.80),
        "adx_floor": 14,
        "book": "thin",
        "slow_max_hold_h": 7.0,
        "fast_max_hold_h": 1.5,
        "slow_min_hold_min": 45,
        "fast_min_hold_min": 15,
        "base_leverage": 15,
        "long_leverage": 15,
        "short_leverage": 15,
        "ref_atr_1h_pct": 0.60,
        "consolidation_exit_min": 9999,      # V6.0.7 — effectively skip
        "hard_block_long_bearish": True,     # V6.0.7 audit n=31 WR=7.1%
        "bigwick": {
            "interval": "1h",
            "tp_cap_pct": 4.0,
            "sl_pct": 1.00,
            "max_hold_h": 2.5,
            "leverage_long": 10,
            "leverage_short": 12,
            "wick_thresh": 0.65,
            "min_range_pct": 0.40,
        },
        "fast": {
            "trigger": "macro_policy_headline + liquidity_sweep_wick",
            "horizon_hours": (1, 24),
        },
        "slow": {
            "trigger": "koios_drep_count_7d_delta + stealth_accumulation_pattern",
            "horizon_hours": (24 * 30 * 3, 24 * 365),
        },
    }

    def entry_signal(self, context: Dict[str, Any]) -> Optional[TradePlan]:
        klines = context.get("klines") or {}
        prices = context.get("prices") or {}
        regime = context.get("regime") or {}
        price = float(prices.get(self.pair) or 0.0)
        if price <= 0:
            return None

        bw = self.CONFIG["bigwick"]
        direction = self._bigwick_direction(klines)
        if direction:
            # V6.0.7: block LONG in bearish regimes (n=31 WR=7.1%)
            if direction == "LONG" and self.CONFIG.get("hard_block_long_bearish"):
                sym_regime = (
                    regime.get(self.pair) or regime.get("ADA") or "NORMAL"
                ).upper()
                if sym_regime in ("TRENDING_DOWN", "CRASH"):
                    direction = None

        if direction:
            lev = int(bw.get(f"leverage_{direction.lower()}", self.CONFIG["base_leverage"]))
            plan = self._build_entry_plan(
                lane="bigwick",
                direction=direction,
                price=price,
                tp_pct=float(bw["tp_cap_pct"]),
                sl_pct=float(bw["sl_pct"]),
                hold_max_h=float(bw["max_hold_h"]),
                leverage=lev,
                equity_usd=float(context.get("equity_usd") or 40_000.0),
            )
            if plan:
                return plan

        # Session C: JUDGE-driven fast lane (JUDGE already enforces
        # PAIR_DIR_REGIME_HARD_BLOCK so ADA LONG in BEARISH is BLOCK upstream).
        return self._judge_fast_entry(context, price)

    def exit_signal(
        self, position: Dict[str, Any], context: Dict[str, Any]
    ) -> Optional[ExitDecision]:
        return self._exit_via_cascade(position, context)

    def hold_signal(
        self, position: Dict[str, Any], context: Dict[str, Any]
    ) -> HoldDecision:
        return self._hold_default(position, context)
