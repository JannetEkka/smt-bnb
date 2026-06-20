"""BTC strategy — ground truth: docs/research/btc.md

Lanes:
- FAST: futures OI spike + funding rate extrema → 1H/4H RSI wick scalps;
  horizon 4h-3d.
- BIGWICK: 1H candle wick rejection (wick > 60% range); bidirectional; 0.5-2.5h.
- SLOW: weekly ETF net inflow + 1k+ BTC wallet count; institutional
  floor accumulation; horizon 3mo-2yr.

Per-pair config (V5.0.7 baseline; modified by regime / vol / lane):
- TP cap 4.0% · SL 0.40-1.00% · ADX floor 14 · thick book
- slow max hold 4.0h · fast max hold 0.5h
- min hold: slow 30min / fast 8min
- base leverage 20× · ref 1H ATR 0.50%

Patterns (docs/research/btc.md):
- "Halving Lag" 4-5mo post-halving sideways chop / distribution.
- Institutional ETF floor — wick depth bid backstop.
- Macro-dominance — moves on macro liquidity, not crypto news.

TODO Session B: port entry/exit from archive/v6.0/v4/smt_nightly_trade_v3_1.py
- WhalePersona BTC branches: 6235-6691
- check_pair_breakout BTC FAST_PATH thresholds
- V6.0.6 SLOW-HOLD: Binance spot-perp 7d basis + 1H RSI<45 + price>200d EMA
"""

from __future__ import annotations
from typing import Any, Dict, Optional

from smt.pairs.base import Strategy
from smt.core.trade_plan import TradePlan, ExitDecision, HoldDecision


class BTCStrategy(Strategy):
    pair = "BTCUSDT"
    CONFIG: Dict[str, Any] = {
        "tp_cap_pct": 4.0,
        "sl_range_pct": (0.40, 1.00),
        "adx_floor": 14,
        "book": "thick",
        "slow_max_hold_h": 4.0,
        "fast_max_hold_h": 0.5,
        "slow_min_hold_min": 30,
        "fast_min_hold_min": 8,
        "base_leverage": 20,
        "long_leverage": 20,
        "short_leverage": 20,
        "ref_atr_1h_pct": 0.50,
        "bigwick": {
            "interval": "1h",
            "tp_cap_pct": 3.5,
            "sl_pct": 0.70,
            "max_hold_h": 2.5,
            "leverage_long": 15,
            "leverage_short": 18,
            "wick_thresh": 0.60,
            "min_range_pct": 0.30,
        },
        "fast": {
            "trigger": "futures_oi_spike + funding_extrema + rsi_wick",
            "horizon_hours": (4, 72),
        },
        "slow": {
            "trigger": "weekly_etf_net_inflow + 1k_btc_wallet_count_delta",
            "horizon_hours": (24 * 90, 24 * 365 * 2),
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
