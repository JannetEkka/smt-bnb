"""BNB strategy — ground truth: docs/research/bnb.md

Lanes:
- FAST: V-bottom macro dip + launchpool anchor; horizon 1-24h.
- BIGWICK: 1H candle wick rejection (wick > 60% range); bidirectional; 0.5-2.5h.
- SLOW: BSC TVL 7d Δ ≥+10% + price > 200d EMA; horizon 3mo-1yr.

Per-pair config: TP 5.0% · SL 0.50-1.50% · ADX 12 · thick · slow 4.0h /
fast 0.75h · min 35/10min · leverage 18× · ref 1H ATR 0.60%.

Patterns (docs/research/bnb.md):
- "Launchpool Anchor" — whales accumulate ahead of announcements;
  Flow flips from selling to "locking" → forced upward price.
- "Exchange Proxy" — first to dump on macro tension (exchange risk).
- "Burn vs Bleed" — quarterly burns create deflationary lag; higher
  lows on long-term charts vs ADA/LTC.
- "V-Bottom Specialist" — fastest recovery velocity after crashes.

TODO Session B: port from archive/v6.0/v4/smt_nightly_trade_v3_1.py
- V6.0.6 SLOW-HOLD: DefiLlama BSC TVL 7d Δ ≥+10% + price>200d EMA (no key).
- B.3 BNB Launchpool playbook.
- V6.0.7b BNB FAST_PATH threshold raised 0.28→0.40 (was over-firing).
"""

from __future__ import annotations
from typing import Any, Dict, Optional

from smt.pairs.base import Strategy
from smt.core.trade_plan import TradePlan, ExitDecision, HoldDecision


class BNBStrategy(Strategy):
    pair = "BNBUSDT"
    CONFIG: Dict[str, Any] = {
        "tp_cap_pct": 5.0,
        "sl_range_pct": (0.50, 1.50),
        "adx_floor": 12,
        "book": "thick",
        "slow_max_hold_h": 4.0,
        "fast_max_hold_h": 0.75,
        "slow_min_hold_min": 35,
        "fast_min_hold_min": 10,
        "base_leverage": 18,
        "long_leverage": 18,
        "short_leverage": 18,
        "ref_atr_1h_pct": 0.60,
        "fast_path_breakout_thresh": 0.40,   # V6.0.7b
        "bigwick": {
            "interval": "1h",
            "tp_cap_pct": 4.0,
            "sl_pct": 0.90,
            "max_hold_h": 2.5,
            "leverage_long": 12,
            "leverage_short": 15,
            "wick_thresh": 0.60,
            "min_range_pct": 0.40,
        },
        "fast": {
            "trigger": "v_bottom_macro_dip + launchpool_anchor",
            "horizon_hours": (1, 24),
        },
        "slow": {
            "trigger": "bsc_tvl_7d_delta + price_above_200d_ema",
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
