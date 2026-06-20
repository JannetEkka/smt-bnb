"""ETH strategy — ground truth: docs/research/eth.md

Lanes:
- FAST: mainnet gas spikes + on-chain liquidation pools → 1H-24H.
- BIGWICK: 1H candle wick rejection (wick > 60% range); bidirectional; 0.5-2.0h.
- SLOW: net staking inflows + whale staking concentration; spot
  accumulate on weekly RSI<40; horizon 6mo-2yr.

Per-pair config: TP 4.0% · SL 0.45-1.20% · ADX 16 · thick · slow 4.0h /
fast 0.5h · min 30/8min · leverage 20× · ref 1H ATR 0.60%.

Patterns (docs/research/eth.md):
- "L2 Value Drain" — mainnet bleeds when L2s absorb volume.
- "Staking-Yield Floor" — 4-5% native yield = whale step-in level.
- Delayed ETF Effect — Grayscale outflows absorbed early buy pressure.

TODO Session B: port from archive/v6.0/v4/smt_nightly_trade_v3_1.py
- V6.0.6 SLOW-HOLD: DefiLlama Lido+RocketPool+EigenLayer 7d TVL Δ ≥+1.0% +
  weekly RSI<40 + structural SL = 200d EMA.
"""

from __future__ import annotations
from typing import Any, Dict, Optional

from smt.pairs.base import Strategy
from smt.core.trade_plan import TradePlan, ExitDecision, HoldDecision


class ETHStrategy(Strategy):
    pair = "ETHUSDT"
    CONFIG: Dict[str, Any] = {
        "tp_cap_pct": 4.0,
        "sl_range_pct": (0.45, 1.20),
        "adx_floor": 16,
        "book": "thick",
        "slow_max_hold_h": 4.0,
        "fast_max_hold_h": 0.5,
        "slow_min_hold_min": 30,
        "fast_min_hold_min": 8,
        "base_leverage": 20,
        "long_leverage": 20,
        "short_leverage": 20,
        "ref_atr_1h_pct": 0.60,
        "bigwick": {
            "interval": "1h",
            "tp_cap_pct": 3.5,
            "sl_pct": 0.80,
            "max_hold_h": 2.0,
            "leverage_long": 15,
            "leverage_short": 18,
            "wick_thresh": 0.60,
            "min_range_pct": 0.35,
        },
        "fast": {
            "trigger": "gas_spike + onchain_liquidation_pool",
            "horizon_hours": (1, 24),
        },
        "slow": {
            "trigger": "net_staking_inflow + whale_staking_concentration",
            "horizon_hours": (24 * 30 * 6, 24 * 365 * 2),
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
