"""XRP strategy — ground truth: docs/research/xrp.md

Lanes:
- FAST: SEC/Senate news + OI spikes; scalp regulatory wicks; 15min-12h.
- BIGWICK: 1H candle wick rejection (wick > 60% range); bidirectional; 0.5-2.0h.
- SLOW: net ETF inflows + corporate custody; Weekly RSI<40 entry; 1mo-1yr.

Per-pair config: TP 5.0% · SL 0.50-1.50% · ADX 14 · thin · slow 6.0h /
fast 1.0h · min 35/10min · leverage 15× · ref 1H ATR 0.60%.

Present state (May 2026 per research):
- Compressed wedge $1.34-$1.45.
- CLARITY Act advanced 15-9 Senate Banking on 2026-05-14.
- US spot XRP ETF inflows $25.8M on 2026-05-11 (cum AUM $1.37B).
- Stealth accumulation 2026-05-12 to 05-16: wallets holding 1M+ XRP rising.

TODO Session B: port from archive/v6.0/v4/smt_nightly_trade_v3_1.py
- V6.0.6 SLOW-HOLD: Gemini-grounded spot-XRP-ETF 7d net inflow ≥$25M +
  weekly RSI<40 + $1.30 floor SL.
- V6.0.9 PART 3: XRP regulatory_bias gate (POSITIVE/NEUTRAL/NEGATIVE);
  NEGATIVE blocks LONG even at $25M+ inflow.
- B.9 XRP regulatory-event playbook + thin-book-flush bucket (V6.0.4 / V6.0.6).
"""

from __future__ import annotations
from typing import Any, Dict, Optional

from smt.pairs.base import Strategy
from smt.core.trade_plan import TradePlan, ExitDecision, HoldDecision


class XRPStrategy(Strategy):
    pair = "XRPUSDT"
    CONFIG: Dict[str, Any] = {
        "tp_cap_pct": 5.0,
        "sl_range_pct": (0.50, 1.50),
        "adx_floor": 14,
        "book": "thin",
        "slow_max_hold_h": 6.0,
        "fast_max_hold_h": 1.0,
        "slow_min_hold_min": 35,
        "fast_min_hold_min": 10,
        "base_leverage": 15,
        "long_leverage": 15,
        "short_leverage": 15,
        "ref_atr_1h_pct": 0.60,
        "slow_floor_usd": 1.30,           # V6.0.6 structural SL
        "slow_etf_inflow_min_usd": 25_000_000,  # 7d threshold
        "bigwick": {
            "interval": "1h",
            "tp_cap_pct": 4.0,
            "sl_pct": 0.90,
            "max_hold_h": 2.0,
            "leverage_long": 10,
            "leverage_short": 12,
            "wick_thresh": 0.60,
            "min_range_pct": 0.35,
        },
        "fast": {
            "trigger": "sec_senate_news_print + oi_spike + 1h_funding_extrema",
            "horizon_hours": (0.25, 12),
        },
        "slow": {
            "trigger": "etf_7d_net_inflow_min + weekly_rsi_below_40 + regulatory_bias_not_negative",
            "horizon_hours": (24 * 30, 24 * 365),
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
