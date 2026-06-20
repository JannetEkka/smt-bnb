"""SOL strategy — ground truth: docs/research/sol.md

Lanes:
- FAST: BTC beta ×2 + Firedancer/tech catalyst; horizon 4-48h.
- BIGWICK: 1H candle wick rejection (wick > 60% range); bidirectional; 0.5-2.0h.
- SLOW: Solana DEX 7d vol ≥ 30d p80 + price > 200d EMA; horizon 3mo-1yr.

Per-pair config: TP 5.0% · SL 0.50-1.50% · ADX 10 · thick · slow 5.0h /
fast 0.75h · min 40/12min · leverage 18× · ref 1H ATR 0.80%.

Patterns (docs/research/sol.md):
- "Beta" — 2-3× BTC up, 1.5× BTC down. Highest-beta large-cap.
- "Firedancer / Tech Catalyst" — moves on validator + upgrade news.
- "Meme-Coin Liquidity Trap" — SOL price proxy for SOL-meme volume;
  dies when meme volume dies.
- $110-$120 "Institutional Floor" — recurring whale defense zone.

V6.0.7 audit: SOL SENT silenced at 0.25× (-25% edge_wr); rely on
TECH/FLOW + B.4 mean-revert at $110-$120 floor.

TODO Session B: port from archive/v6.0/v4/smt_nightly_trade_v3_1.py
- V6.0.6 SLOW-HOLD: DefiLlama Solana DEX 7d vol ≥ 30d p80 + price>200d EMA.
- B.4 SOL mean-revert at $110-$120 institutional floor.
- V6.0.7 partial_close trigger mult SOL TRENDING 2.2 / RANGING 1.1 →
  effective trigger 1.0%/0.5%.
- V6.0.7 consolidation_exit_min SOL = 30 (event-pair fast reset).
"""

from __future__ import annotations
from typing import Any, Dict, Optional

from smt.pairs.base import Strategy
from smt.core.trade_plan import TradePlan, ExitDecision, HoldDecision


class SOLStrategy(Strategy):
    pair = "SOLUSDT"
    CONFIG: Dict[str, Any] = {
        "tp_cap_pct": 5.0,
        "sl_range_pct": (0.50, 1.50),
        "adx_floor": 10,
        "book": "thick",
        "slow_max_hold_h": 5.0,
        "fast_max_hold_h": 0.75,
        "slow_min_hold_min": 40,
        "fast_min_hold_min": 12,
        "base_leverage": 18,
        "long_leverage": 18,
        "short_leverage": 18,
        "ref_atr_1h_pct": 0.80,
        "institutional_floor_band_usd": (110, 120),
        "partial_close_trigger_mult_trending": 2.2,   # V6.0.7
        "partial_close_trigger_mult_ranging": 1.1,    # V6.0.7
        "consolidation_exit_min": 30,                 # V6.0.7
        "bigwick": {
            "interval": "1h",
            "tp_cap_pct": 4.0,
            "sl_pct": 1.00,
            "max_hold_h": 2.0,
            "leverage_long": 12,
            "leverage_short": 15,
            "wick_thresh": 0.60,
            "min_range_pct": 0.40,
        },
        "fast": {
            "trigger": "btc_beta_x2 + firedancer_catalyst",
            "horizon_hours": (4, 48),
        },
        "slow": {
            "trigger": "sol_dex_vol_7d_above_p80 + price_above_200d_ema",
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
