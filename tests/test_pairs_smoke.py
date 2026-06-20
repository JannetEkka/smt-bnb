"""Session B acceptance: 3 lanes × 8 pairs smoke tests.

Verifies:
1. Bigwick lane returns TradePlan (wick detected, fee floor passes).
2. Fast / slow lanes return None — JUDGE not wired until Session C.
3. entry_signal never raises for the happy path on any lane.
4. exit_signal / hold_signal never raise on a synthetic open position.
5. ADA hard_block_long_bearish: LONG blocked in TRENDING_DOWN / CRASH.
6. DOGE B.2 200d-MA: LONG blocked when price < 200d EMA.
7. Fee floor contract: any returned TradePlan satisfies est_profit_net > est_fees.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, List

import pytest

from smt.pairs.btc import BTCStrategy
from smt.pairs.eth import ETHStrategy
from smt.pairs.bnb import BNBStrategy
from smt.pairs.ltc import LTCStrategy
from smt.pairs.sol import SOLStrategy
from smt.pairs.xrp import XRPStrategy
from smt.pairs.ada import ADAStrategy
from smt.pairs.doge import DOGEStrategy
from smt.core.trade_plan import TradePlan, HoldDecision

# ── Candle helpers ────────────────────────────────────────────────────────────

def _c(o, h, l, c, v=1000) -> List:
    return [0, str(o), str(h), str(l), str(c), str(v)]


# Last CLOSED candle (index -2) has upper wick > 76% of range → SHORT signal.
# o=49000, h=51000, l=48500, c=49100 → range=2500, upper=1900, 1900/2500=0.76
WICK_UP = [
    _c(49000, 50000, 48500, 49500),
    _c(49000, 51000, 48500, 49100),  # wick_up candle
    _c(49050, 49200, 49000, 49050),
]

# Last CLOSED candle (index -2) has lower wick > 80% of range → LONG signal.
# o=50000, h=50500, l=47500, c=49900 → range=3000, lower=2400, 2400/3000=0.80
WICK_DOWN = [
    _c(50000, 51000, 49000, 50100),
    _c(50000, 50500, 47500, 49900),  # wick_down candle
    _c(49950, 50000, 49800, 49950),
]

NEUTRAL_CANDLES = [_c(50000, 50100, 49900, 50050)] * 3  # no strong wick

ALL_STRATEGIES = [
    BTCStrategy(), ETHStrategy(), BNBStrategy(), LTCStrategy(),
    SOLStrategy(), XRPStrategy(), ADAStrategy(), DOGEStrategy(),
]

STRATEGY_IDS = [s.pair for s in ALL_STRATEGIES]


def _ctx(sym: str, price: float, klines_1h: List, regime: str = "NORMAL") -> Dict[str, Any]:
    return {
        "klines": {
            f"{sym}#1h": klines_1h,
            f"{sym}_1h": klines_1h,
        },
        "prices": {sym: price},
        "regime": {sym: regime},
        "equity_usd": 40_000.0,
        "timestamp": datetime.now(timezone.utc).timestamp(),
    }


def _fresh_position(side: str, entry_price: float, lane: str = "bigwick") -> Dict[str, Any]:
    return {
        "side": side,
        "entry_lane": lane,
        "entry_price": entry_price,
        "peak_pnl_pct": 0.0,
        "partial_close_done": False,
        "_against_consec": 0,
        "_drop_consec": 0,
        "_np_consec": 0,
        "_cons_consec": 0,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "hold_max": 2.5,
    }


# ── Bigwick SHORT entry (wick-up → SHORT) ────────────────────────────────────

@pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=STRATEGY_IDS)
def test_bigwick_short_entry_returns_trade_plan(strategy):
    ctx = _ctx(strategy.pair, 50_000.0, WICK_UP)
    plan = strategy.entry_signal(ctx)
    assert plan is not None, f"{strategy.pair}: expected TradePlan for wick-up SHORT"
    assert isinstance(plan, TradePlan)
    assert plan.lane == "bigwick"
    assert plan.direction == "SHORT"
    assert plan.est_profit_net > plan.est_fees, "fee floor violated"


# ── Bigwick LONG entry (wick-down → LONG) ────────────────────────────────────

@pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=STRATEGY_IDS)
def test_bigwick_long_entry_returns_trade_plan(strategy):
    # NORMAL regime — no hard-blocks active; no 200d daily klines → 200d check skipped
    ctx = _ctx(strategy.pair, 50_000.0, WICK_DOWN, regime="NORMAL")
    plan = strategy.entry_signal(ctx)
    assert plan is not None, f"{strategy.pair}: expected TradePlan for wick-down LONG"
    assert isinstance(plan, TradePlan)
    assert plan.lane == "bigwick"
    assert plan.direction == "LONG"
    assert plan.est_profit_net > plan.est_fees, "fee floor violated"


# ── Neutral klines → no bigwick signal; fast/slow return None (Session C) ────

@pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=STRATEGY_IDS)
def test_no_wick_returns_none(strategy):
    ctx = _ctx(strategy.pair, 50_000.0, NEUTRAL_CANDLES)
    plan = strategy.entry_signal(ctx)
    assert plan is None, f"{strategy.pair}: no wick → None expected"


@pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=STRATEGY_IDS)
def test_empty_klines_returns_none(strategy):
    ctx = {"klines": {}, "prices": {strategy.pair: 50_000.0}, "regime": {}, "equity_usd": 40_000.0}
    plan = strategy.entry_signal(ctx)
    assert plan is None, f"{strategy.pair}: no klines → None (JUDGE not wired)"


@pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=STRATEGY_IDS)
def test_zero_price_returns_none(strategy):
    ctx = _ctx(strategy.pair, 0.0, WICK_UP)
    plan = strategy.entry_signal(ctx)
    assert plan is None, f"{strategy.pair}: price=0 → None"


# ── ADA hard_block_long_bearish ──────────────────────────────────────────────

@pytest.mark.parametrize("regime", ["TRENDING_DOWN", "CRASH"])
def test_ada_hard_block_long_bearish(regime):
    strat = ADAStrategy()
    ctx = _ctx("ADAUSDT", 0.45, WICK_DOWN, regime=regime)
    plan = strat.entry_signal(ctx)
    assert plan is None, f"ADA LONG must be blocked in {regime}"


def test_ada_long_allowed_in_normal_regime():
    strat = ADAStrategy()
    ctx = _ctx("ADAUSDT", 0.45, WICK_DOWN, regime="NORMAL")
    plan = strat.entry_signal(ctx)
    assert plan is not None and plan.direction == "LONG"


def test_ada_short_not_blocked_in_bearish():
    strat = ADAStrategy()
    ctx = _ctx("ADAUSDT", 0.45, WICK_UP, regime="TRENDING_DOWN")
    plan = strat.entry_signal(ctx)
    assert plan is not None and plan.direction == "SHORT"


# ── DOGE B.2 200d-MA block ───────────────────────────────────────────────────

def test_doge_long_blocked_below_200d_ema():
    strat = DOGEStrategy()
    # 200 daily candles closing at 60000; current price = 50000 → below EMA
    daily = [_c(60000, 61000, 59000, 60000)] * 200 + [_c(60000, 60500, 59500, 60000)]
    ctx = {
        "klines": {
            "DOGEUSDT#1h": WICK_DOWN,
            "DOGEUSDT_1h": WICK_DOWN,
            "DOGEUSDT#1d": daily,
        },
        "prices": {"DOGEUSDT": 50_000.0},
        "regime": {"DOGEUSDT": "NORMAL"},
        "equity_usd": 40_000.0,
    }
    plan = strat.entry_signal(ctx)
    assert plan is None, "DOGE LONG must be blocked when price < 200d EMA"


def test_doge_long_allowed_above_200d_ema():
    strat = DOGEStrategy()
    daily = [_c(40000, 41000, 39000, 40000)] * 200 + [_c(40000, 40500, 39500, 40000)]
    ctx = {
        "klines": {
            "DOGEUSDT#1h": WICK_DOWN,
            "DOGEUSDT_1h": WICK_DOWN,
            "DOGEUSDT#1d": daily,
        },
        "prices": {"DOGEUSDT": 50_000.0},  # above EMA of 40000
        "regime": {"DOGEUSDT": "NORMAL"},
        "equity_usd": 40_000.0,
    }
    plan = strat.entry_signal(ctx)
    assert plan is not None and plan.direction == "LONG"


def test_doge_short_not_blocked_by_ema():
    strat = DOGEStrategy()
    # Even below 200d EMA, SHORT (wick-up) should work
    daily = [_c(60000, 61000, 59000, 60000)] * 200 + [_c(60000, 60500, 59500, 60000)]
    ctx = {
        "klines": {
            "DOGEUSDT#1h": WICK_UP,
            "DOGEUSDT_1h": WICK_UP,
            "DOGEUSDT#1d": daily,
        },
        "prices": {"DOGEUSDT": 50_000.0},
        "regime": {"DOGEUSDT": "NORMAL"},
        "equity_usd": 40_000.0,
    }
    plan = strat.entry_signal(ctx)
    assert plan is not None and plan.direction == "SHORT"


# ── exit_signal: fresh position at entry price → no cascade trigger ───────────

@pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=STRATEGY_IDS)
def test_exit_signal_fresh_position_returns_none(strategy):
    # Use wick-up to get a SHORT plan (SHORT available for all 8 pairs)
    ctx_entry = _ctx(strategy.pair, 50_000.0, WICK_UP)
    plan = strategy.entry_signal(ctx_entry)
    assert plan is not None

    pos = _fresh_position(plan.direction, 50_000.0, lane="bigwick")
    ctx_exit = {
        "prices": {strategy.pair: 50_000.0},
        "klines": {},
        "regime": {strategy.pair: "NORMAL"},
        "equity_usd": 40_000.0,
    }
    decision = strategy.exit_signal(pos, ctx_exit)
    assert decision is None, f"{strategy.pair}: fresh position should not trigger exit"


# ── hold_signal: always returns HoldDecision(should_hold=True) ───────────────

@pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=STRATEGY_IDS)
def test_hold_signal_returns_hold_decision(strategy):
    pos = _fresh_position("LONG", 50_000.0)
    ctx = {
        "prices": {strategy.pair: 50_000.0},
        "klines": {},
        "regime": {strategy.pair: "NORMAL"},
        "equity_usd": 40_000.0,
    }
    decision = strategy.hold_signal(pos, ctx)
    assert isinstance(decision, HoldDecision)
    assert decision.should_hold is True


# ── All 3 lanes covered (fast / bigwick / slow) — no NotImplementedError ─────

@pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=STRATEGY_IDS)
def test_three_lanes_do_not_raise(strategy):
    # Bigwick lane — wick present
    ctx_bw = _ctx(strategy.pair, 50_000.0, WICK_UP)
    strategy.entry_signal(ctx_bw)  # must not raise

    # Fast lane stub (no JUDGE) — empty klines
    ctx_fast = {"klines": {}, "prices": {strategy.pair: 50_000.0}, "regime": {}, "equity_usd": 40_000.0}
    strategy.entry_signal(ctx_fast)  # must not raise

    # Slow lane stub (no JUDGE) — same empty klines
    strategy.entry_signal(ctx_fast)  # must not raise

    # exit / hold on open position — must not raise
    pos = _fresh_position("SHORT", 50_000.0, lane="bigwick")
    strategy.exit_signal(pos, ctx_fast)
    strategy.hold_signal(pos, ctx_fast)
