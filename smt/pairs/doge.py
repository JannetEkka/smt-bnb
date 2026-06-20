"""DOGE strategy — ground truth: docs/research/doge.md

Lanes:
- FAST: Musk/political proxy + 200d EMA cross; horizon 1-24h.
- BIGWICK: 1H candle wick rejection (wick > 65% range); bidirectional; 0.5-2.0h.
  B.2 200d-MA block: LONG entry blocked when price < 200d EMA.
- SLOW: price above 200d EMA + accumulation phase; horizon 1mo-1yr.

Per-pair config: TP 6.0% · SL 0.70-2.00% · ADX 14 · thin · slow 7.0h /
fast 1.5h · min 45/15min · leverage 15× · ref 1H ATR 0.80%.

Patterns (docs/research/doge.md):
- "Musk-Political Proxy" — strongest regimes tied to political hype
  cycles, not network utility.
- "Liquidity Vacuum" — leads up in euphoria (most retail leverage);
  leads down in panic (drops harder than ADA/BNB).
- "Whale Exit vs Retail Entry" — recurring asymmetric distribution
  (whales sell into euphoria, retail buys the wick).
- 200-day EMA "Death/Life Line" — BINARY: above EMA = explosive, below
  EMA = 6+ month slow bleed.

V6.0.7 audit: DOGE "all personas broken" — lowered all weights,
rely on B.2 200d-MA playbook as primary alpha pending V6.0.12 re-audit.

TODO Session B: port from archive/v6.0/v4/smt_nightly_trade_v3_1.py
- B.2 DOGE 200d-MA playbook (live since V5.2.5).
- V6.0.7 partial_close trigger mult DOGE TRENDING 2.5 / RANGING 1.25 →
  effective trigger 1.0%/0.5%.
- V6.0.7 consolidation_exit_min DOGE = 30 (meme fast reset).
- V6.0.7b DOGE FAST_PATH threshold raised 0.35→0.50 (was over-firing).
- V6.0.8 OnChain DOGE conf softened ×0.85 (all-personas-broken caution).
"""

from __future__ import annotations
from typing import Any, Dict, Optional

from smt.pairs.base import Strategy
from smt.core.trade_plan import TradePlan, ExitDecision, HoldDecision


class DOGEStrategy(Strategy):
    pair = "DOGEUSDT"
    CONFIG: Dict[str, Any] = {
        "tp_cap_pct": 6.0,
        "sl_range_pct": (0.70, 2.00),
        "adx_floor": 14,
        "book": "thin",
        "slow_max_hold_h": 7.0,
        "fast_max_hold_h": 1.5,
        "slow_min_hold_min": 45,
        "fast_min_hold_min": 15,
        "base_leverage": 15,
        "long_leverage": 15,
        "short_leverage": 15,
        "ref_atr_1h_pct": 0.80,
        "partial_close_trigger_mult_trending": 2.5,    # V6.0.7
        "partial_close_trigger_mult_ranging": 1.25,    # V6.0.7
        "consolidation_exit_min": 30,                  # V6.0.7
        "fast_path_breakout_thresh": 0.50,             # V6.0.7b
        "onchain_conf_soften": 0.85,                   # V6.0.8
        "primary_alpha": "B.2 200d-MA playbook",
        "bigwick": {
            "interval": "1h",
            "tp_cap_pct": 5.0,
            "sl_pct": 1.20,
            "max_hold_h": 2.0,
            "leverage_long": 10,
            "leverage_short": 12,
            "wick_thresh": 0.65,
            "min_range_pct": 0.40,
        },
        "fast": {
            "trigger": "musk_political_proxy + 200d_ema_cross",
            "horizon_hours": (1, 24),
        },
        "slow": {
            "trigger": "above_200d_ema + accumulation_phase",
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
        if direction == "LONG" and self._below_200d_ema(klines, price):
            direction = None  # B.2 200d-MA block (V5.2.5)

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

        # Session C: JUDGE-driven fast lane. B.2 200d-MA gate applies here too.
        plan = self._judge_fast_entry(context, price)
        if plan is not None and plan.direction == "LONG" and self._below_200d_ema(klines, price):
            return None
        return plan

    def _below_200d_ema(self, klines: Dict[str, Any], price: float) -> bool:
        for k in (f"{self.pair}#1d", f"{self.pair}_1d", f"{self.pair}:1d"):
            candles = klines.get(k)
            if not candles or len(candles) < 200:
                continue
            try:
                ema200 = sum(float(c[4]) for c in candles[-200:]) / 200.0
            except (IndexError, ValueError):
                return False
            return price < ema200
        return False  # no 200d daily data → skip the gate (symmetric with bigwick)

    def exit_signal(
        self, position: Dict[str, Any], context: Dict[str, Any]
    ) -> Optional[ExitDecision]:
        return self._exit_via_cascade(position, context)

    def hold_signal(
        self, position: Dict[str, Any], context: Dict[str, Any]
    ) -> HoldDecision:
        return self._hold_default(position, context)
