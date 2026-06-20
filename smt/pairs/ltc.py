"""LTC strategy — ground truth: docs/research/ltc.md

Lanes:
- FAST: regulatory classification print + BTC stability catch-up; horizon 4-48h.
- BIGWICK: 1H candle wick rejection (wick > 60% range); bidirectional; 0.5-2.5h.
- SLOW: BTC dominance ≥62% + LTC < $60 + $80 ceiling SL; horizon 3mo-1yr.

Per-pair config: TP 4.0% · SL 0.45-1.20% · ADX 14 · thin · slow 6.0h /
fast 1.0h · min 40/12min · leverage 15× · ref 1H ATR 0.60%.

Patterns (docs/research/ltc.md):
- "Commodity" momentum trigger — moves on regulatory classification
  (CFTC), not partnerships.
- "Digital Silver" lag — waits for BTC stability then "catch-up" trade.
- Institutional floor vs retail boredom — 1k-10k LTC wallets buy when
  retail exits.
- Hard psych ceiling at $110-$115 — heavy distribution zone.
- "Risk-Off" sensitivity — used as liquidity proxy (dumped fast).

TODO Session B: port from archive/v6.0/v4/smt_nightly_trade_v3_1.py
- V6.0.6 B.1 LTC floor refresh [50, 55] / ceilings [60, 80] (was [60,90]/[110,147]).
- V6.0.6 SLOW-HOLD: CMC BTC-dominance ≥62% + LTC<$60 + $80 ceiling SL.
- V6.0.7c TODO: dynamic floor keyed on BTC-dominance pctile (queue item).
"""

from __future__ import annotations
from typing import Any, Dict, Optional

from smt.pairs.base import Strategy
from smt.core.trade_plan import TradePlan, ExitDecision, HoldDecision


class LTCStrategy(Strategy):
    pair = "LTCUSDT"
    CONFIG: Dict[str, Any] = {
        "tp_cap_pct": 4.0,
        "sl_range_pct": (0.45, 1.20),
        "adx_floor": 14,
        "book": "thin",
        "slow_max_hold_h": 6.0,
        "fast_max_hold_h": 1.0,
        "slow_min_hold_min": 40,
        "fast_min_hold_min": 12,
        "base_leverage": 15,
        "long_leverage": 15,
        "short_leverage": 15,
        "ref_atr_1h_pct": 0.60,
        "range_floors_usd": [50, 55],      # V6.0.6 B.1 refresh
        "range_ceilings_usd": [60, 80],    # V6.0.6 B.1 refresh
        "bigwick": {
            "interval": "1h",
            "tp_cap_pct": 3.5,
            "sl_pct": 0.80,
            "max_hold_h": 2.5,
            "leverage_long": 10,
            "leverage_short": 12,
            "wick_thresh": 0.60,
            "min_range_pct": 0.40,
        },
        "fast": {
            "trigger": "regulatory_classification_print + btc_stability_catchup",
            "horizon_hours": (4, 48),
        },
        "slow": {
            "trigger": "btc_dominance_above_62 + ltc_below_60",
            "horizon_hours": (24 * 30 * 3, 24 * 365),
        },
    }

    def entry_signal(self, context: Dict[str, Any]) -> Optional[TradePlan]:
        klines = context.get("klines") or {}
        prices = context.get("prices") or {}
        price = float(prices.get(self.pair) or 0.0)
        if price <= 0:
            return None

        bw = self.CONFIG["bigwick"]
        direction = self._bigwick_direction(klines)
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

        # Session C: JUDGE-driven fast lane (slow routing in Session E).
        return self._judge_fast_entry(context, price)

    def exit_signal(
        self, position: Dict[str, Any], context: Dict[str, Any]
    ) -> Optional[ExitDecision]:
        return self._exit_via_cascade(position, context)

    def hold_signal(
        self, position: Dict[str, Any], context: Dict[str, Any]
    ) -> HoldDecision:
        return self._hold_default(position, context)
