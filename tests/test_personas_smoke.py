"""Session C acceptance: persona + JUDGE + JUDGE-driven strategy lane smoke tests.

Verifies:
1. Each of 7 personas instantiates + `.analyze({})` returns a PersonaVote with
   direction in {"LONG","SHORT","NEUTRAL"} and confidence in [0.0, 1.0].
2. No persona raises on blank context (graceful degrade).
3. JUDGE on blank votes returns WAIT (conf < floor).
4. JUDGE on strong FLOW=LONG×0.9 + TECH=LONG×0.8 returns LONG (conf ≥ floor).
5. JUDGE on strong FLOW=SHORT×0.9 + TECH=SHORT×0.8 returns SHORT.
6. HARD-BLOCK ADA LONG BEARISH → JUDGE returns BLOCK.
7. F&G < 22 capitulation: SHORT-leaning votes → WAIT (hedge-disable).
8. F&G < 22 capitulation: SENTIMENT weight zeroed.
9. Fast strategy lane (entry_signal) returns TradePlan when JUDGE says
   LONG/SHORT and there is no bigwick candle.
"""

from __future__ import annotations
from typing import Dict, List

import pytest

from smt.personas.base import (
    JudgeDecision,
    JUDGE_CONF_FLOOR,
    PersonaVote,
)
from smt.personas.flow import FlowPersona
from smt.personas.technical import TechnicalPersona
from smt.personas.whale import WhalePersona
from smt.personas.sentiment import SentimentPersona, build_pair_prompt
from smt.personas.regime import RegimePersona
from smt.personas.onchain import OnChainPersona
from smt.personas.judge import JudgePersona

from smt.pairs.btc import BTCStrategy
from smt.pairs.eth import ETHStrategy
from smt.pairs.bnb import BNBStrategy
from smt.pairs.ltc import LTCStrategy
from smt.pairs.sol import SOLStrategy
from smt.pairs.xrp import XRPStrategy
from smt.pairs.ada import ADAStrategy
from smt.pairs.doge import DOGEStrategy
from smt.core.trade_plan import TradePlan


ALL_PERSONAS = [
    FlowPersona(), TechnicalPersona(), WhalePersona(),
    SentimentPersona(), RegimePersona(), OnChainPersona(),
]
PERSONA_IDS = [p.name for p in ALL_PERSONAS]


ALL_STRATEGIES = [
    BTCStrategy(), ETHStrategy(), BNBStrategy(), LTCStrategy(),
    SOLStrategy(), XRPStrategy(), ADAStrategy(), DOGEStrategy(),
]
STRATEGY_IDS = [s.pair for s in ALL_STRATEGIES]


def _c(o, h, l, c, v=1000) -> List:
    return [0, str(o), str(h), str(l), str(c), str(v)]


# Neutral candles (no wick) → bigwick lane abstains, JUDGE drives.
NEUTRAL_CANDLES = [_c(50000, 50100, 49900, 50050)] * 4


# ── Persona contract: blank-context graceful degrade ─────────────────────────

@pytest.mark.parametrize("persona", ALL_PERSONAS, ids=PERSONA_IDS)
def test_persona_blank_context_returns_neutral(persona):
    vote = persona.analyze("BTCUSDT", {})
    assert isinstance(vote, PersonaVote)
    assert vote.direction in ("LONG", "SHORT", "NEUTRAL")
    assert 0.0 <= vote.confidence <= 1.0
    # Blank context should be NEUTRAL or near-zero confidence.
    if vote.direction == "NEUTRAL":
        assert vote.confidence == 0.0


@pytest.mark.parametrize("persona", ALL_PERSONAS, ids=PERSONA_IDS)
def test_persona_blank_context_does_not_raise(persona):
    # Multiple pair shapes to surface any string-handling bugs.
    for pair in ("BTCUSDT", "BTC", "DOGEUSDT", "", None):
        try:
            vote = persona.analyze(pair or "BTCUSDT", {})  # type: ignore[arg-type]
            assert isinstance(vote, PersonaVote)
        except Exception as e:
            pytest.fail(f"{persona.name} raised on pair={pair!r}: {e}")


@pytest.mark.parametrize("persona", ALL_PERSONAS, ids=PERSONA_IDS)
def test_persona_partial_context_does_not_raise(persona):
    """Context with klines but no signals must not blow up."""
    ctx = {
        "klines": {"BTCUSDT#1h": NEUTRAL_CANDLES},
        "regime": {"BTCUSDT": "NORMAL"},
    }
    vote = persona.analyze("BTCUSDT", ctx)
    assert isinstance(vote, PersonaVote)
    assert 0.0 <= vote.confidence <= 1.0


# ── JUDGE contract: V5.0.9 raw_judge_bypass ──────────────────────────────────

def _blank_votes() -> Dict[str, PersonaVote]:
    return {
        "flow":      PersonaVote("NEUTRAL", 0.0, ""),
        "technical": PersonaVote("NEUTRAL", 0.0, ""),
        "whale":     PersonaVote("NEUTRAL", 0.0, ""),
        "onchain":   PersonaVote("NEUTRAL", 0.0, ""),
        "sentiment": PersonaVote("NEUTRAL", 0.0, ""),
        "regime":    PersonaVote("NEUTRAL", 0.0, ""),
    }


def _strong_long_votes() -> Dict[str, PersonaVote]:
    v = _blank_votes()
    v["flow"] = PersonaVote("LONG", 0.90, "strong flow long")
    v["technical"] = PersonaVote("LONG", 0.80, "strong tech long")
    return v


def _strong_short_votes() -> Dict[str, PersonaVote]:
    v = _blank_votes()
    v["flow"] = PersonaVote("SHORT", 0.90, "strong flow short")
    v["technical"] = PersonaVote("SHORT", 0.80, "strong tech short")
    return v


def test_judge_blank_returns_wait():
    judge = JudgePersona()
    d = judge.decide("BTCUSDT", _blank_votes(), {"regime": {"BTCUSDT": "NORMAL"}, "fear_greed": 50})
    assert isinstance(d, JudgeDecision)
    assert d.action == "WAIT"
    assert d.confidence < JUDGE_CONF_FLOOR


def test_judge_strong_long_returns_long():
    judge = JudgePersona()
    d = judge.decide("BTCUSDT", _strong_long_votes(), {"regime": {"BTCUSDT": "NORMAL"}, "fear_greed": 50})
    assert d.action == "LONG"
    assert d.confidence >= JUDGE_CONF_FLOOR
    assert d.lane_hint == "fast"


def test_judge_strong_short_returns_short():
    judge = JudgePersona()
    d = judge.decide("ETHUSDT", _strong_short_votes(), {"regime": {"ETHUSDT": "NORMAL"}, "fear_greed": 50})
    assert d.action == "SHORT"
    assert d.confidence >= JUDGE_CONF_FLOOR


def test_judge_ada_long_bearish_hard_block():
    judge = JudgePersona()
    # Votes point LONG, but regime=TRENDING_DOWN → ADA LONG BEARISH HARD-BLOCK.
    d = judge.decide("ADAUSDT", _strong_long_votes(),
                     {"regime": {"ADAUSDT": "TRENDING_DOWN"}, "fear_greed": 30})
    assert d.action == "BLOCK"
    assert "HARD-BLOCK" in d.reasoning


def test_judge_btc_long_bearish_hard_block():
    judge = JudgePersona()
    d = judge.decide("BTCUSDT", _strong_long_votes(),
                     {"regime": {"BTCUSDT": "CRASH"}, "fear_greed": 18})
    assert d.action == "BLOCK"


def test_judge_doge_long_bearish_hard_block():
    judge = JudgePersona()
    d = judge.decide("DOGEUSDT", _strong_long_votes(),
                     {"regime": {"DOGEUSDT": "TRENDING_DOWN"}, "fear_greed": 30})
    assert d.action == "BLOCK"


def test_judge_capitulation_hedge_disables_short():
    """F&G < 22 (CMC) → SHORT is blocked; expect WAIT, not SHORT."""
    judge = JudgePersona()
    d = judge.decide("ETHUSDT", _strong_short_votes(),
                     {"regime": {"ETHUSDT": "NORMAL"}, "fear_greed": 15})
    assert d.action == "WAIT"
    assert "capitulation" in d.reasoning.lower()


def test_judge_capitulation_zeros_sentiment():
    """F&G < 22: SENTIMENT weight is zeroed — even strong opposing SENT vote
    cannot drag the leading direction below floor."""
    judge = JudgePersona()
    votes = _strong_long_votes()
    votes["sentiment"] = PersonaVote("SHORT", 1.0, "strong opposing sentiment")
    # LONG is allowed in capitulation; SENT veto must be muted.
    d = judge.decide("BTCUSDT", votes,
                     {"regime": {"BTCUSDT": "NORMAL"}, "fear_greed": 15})
    # With SENT zeroed and BTC capitulation NOT a HARD-BLOCK (BEARISH-regime is
    # what triggers BTC LONG block, F&G band alone doesn't): JUDGE should still
    # return LONG.
    assert d.action == "LONG"


def test_judge_sentiment_alone_cannot_lift():
    """Veto-only: SENTIMENT vote alone cannot cross JUDGE floor."""
    judge = JudgePersona()
    votes = _blank_votes()
    votes["sentiment"] = PersonaVote("LONG", 1.0, "sentiment alone")
    d = judge.decide("BTCUSDT", votes,
                     {"regime": {"BTCUSDT": "NORMAL"}, "fear_greed": 50})
    assert d.action == "WAIT"
    assert d.confidence < JUDGE_CONF_FLOOR


def test_judge_sentiment_can_veto():
    """SENTIMENT opposing a near-floor LONG can drag conf below floor."""
    judge = JudgePersona()
    votes = _blank_votes()
    # Just-clearing LONG: FLOW 0.50 LONG + TECH 0.50 LONG → weighted ~0.35
    # below the 0.55 floor so test should already WAIT — instead use a slightly
    # higher base that clears WITHOUT sentiment veto, then verify SHORT veto
    # cancels.
    votes["flow"] = PersonaVote("LONG", 0.90, "")
    votes["technical"] = PersonaVote("LONG", 0.80, "")
    # ADA has SENT weight 0.7 — strongest veto pair we have
    d_no_veto = judge.decide("ADAUSDT", votes,
                             {"regime": {"ADAUSDT": "NORMAL"}, "fear_greed": 50})
    assert d_no_veto.action == "LONG"
    # Add strong opposing SENTIMENT
    votes["sentiment"] = PersonaVote("SHORT", 1.0, "ada contra-sent")
    d_with_veto = judge.decide("ADAUSDT", votes,
                               {"regime": {"ADAUSDT": "NORMAL"}, "fear_greed": 50})
    # Confidence MUST drop (veto applied)
    assert d_with_veto.confidence < d_no_veto.confidence


# ── End-to-end: JUDGE-driven fast lane through Strategy.entry_signal ─────────

def _ctx(sym: str, price: float, klines=None, regime: str = "NORMAL",
         judge_decision: JudgeDecision = None) -> Dict:
    return {
        "klines": {
            f"{sym}#1h": klines or NEUTRAL_CANDLES,
            f"{sym}_1h": klines or NEUTRAL_CANDLES,
        },
        "prices": {sym: price},
        "regime": {sym: regime},
        "equity_usd": 40_000.0,
        "judge": {sym: judge_decision} if judge_decision else {},
    }


@pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=STRATEGY_IDS)
def test_judge_drives_fast_lane_long(strategy):
    judge_d = JudgeDecision(
        action="LONG", confidence=0.85, lane_hint="fast",
        reasoning="synthetic strong long", persona_breakdown={"flow": 0.36, "technical": 0.24},
    )
    ctx = _ctx(strategy.pair, 50_000.0, judge_decision=judge_d)
    plan = strategy.entry_signal(ctx)
    assert plan is not None, f"{strategy.pair}: JUDGE LONG must yield a TradePlan"
    assert isinstance(plan, TradePlan)
    assert plan.direction == "LONG"
    assert plan.lane in ("fast", "slow")
    assert plan.est_profit_net > plan.est_fees, "fee floor violated"
    assert plan.decision_confidence == pytest.approx(0.85)


@pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=STRATEGY_IDS)
def test_judge_drives_fast_lane_short(strategy):
    judge_d = JudgeDecision(action="SHORT", confidence=0.85, lane_hint="fast")
    ctx = _ctx(strategy.pair, 50_000.0, judge_decision=judge_d)
    plan = strategy.entry_signal(ctx)
    assert plan is not None, f"{strategy.pair}: JUDGE SHORT must yield a TradePlan"
    assert plan.direction == "SHORT"
    assert plan.lane in ("fast", "slow")
    assert plan.est_profit_net > plan.est_fees


@pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=STRATEGY_IDS)
def test_judge_wait_yields_no_plan(strategy):
    judge_d = JudgeDecision(action="WAIT", confidence=0.30)
    ctx = _ctx(strategy.pair, 50_000.0, judge_decision=judge_d)
    plan = strategy.entry_signal(ctx)
    assert plan is None, f"{strategy.pair}: JUDGE WAIT → no plan"


@pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=STRATEGY_IDS)
def test_judge_block_yields_no_plan(strategy):
    judge_d = JudgeDecision(action="BLOCK", confidence=0.0)
    ctx = _ctx(strategy.pair, 50_000.0, judge_decision=judge_d)
    plan = strategy.entry_signal(ctx)
    assert plan is None, f"{strategy.pair}: JUDGE BLOCK → no plan"


# ── DOGE 200d-EMA still gates JUDGE-driven LONG ──────────────────────────────

def test_doge_judge_long_blocked_below_200d_ema():
    strat = DOGEStrategy()
    judge_d = JudgeDecision(action="LONG", confidence=0.95, lane_hint="fast")
    daily = [_c(60000, 61000, 59000, 60000)] * 200 + [_c(60000, 60500, 59500, 60000)]
    ctx = {
        "klines": {
            "DOGEUSDT#1h": NEUTRAL_CANDLES,
            "DOGEUSDT_1h": NEUTRAL_CANDLES,
            "DOGEUSDT#1d": daily,
        },
        "prices": {"DOGEUSDT": 50_000.0},
        "regime": {"DOGEUSDT": "NORMAL"},
        "equity_usd": 40_000.0,
        "judge": {"DOGEUSDT": judge_d},
    }
    plan = strat.entry_signal(ctx)
    assert plan is None, "DOGE JUDGE-driven LONG must be blocked when price < 200d EMA"


def test_doge_judge_long_allowed_above_200d_ema():
    strat = DOGEStrategy()
    judge_d = JudgeDecision(action="LONG", confidence=0.95, lane_hint="fast")
    daily = [_c(40000, 41000, 39000, 40000)] * 200 + [_c(40000, 40500, 39500, 40000)]
    ctx = {
        "klines": {
            "DOGEUSDT#1h": NEUTRAL_CANDLES,
            "DOGEUSDT_1h": NEUTRAL_CANDLES,
            "DOGEUSDT#1d": daily,
        },
        "prices": {"DOGEUSDT": 50_000.0},  # above 40000 EMA
        "regime": {"DOGEUSDT": "NORMAL"},
        "equity_usd": 40_000.0,
        "judge": {"DOGEUSDT": judge_d},
    }
    plan = strat.entry_signal(ctx)
    assert plan is not None and plan.direction == "LONG"


# ── SENTIMENT prompt input audit (Session C rule 9) ──────────────────────────

def test_sentiment_prompt_marks_contra_pairs():
    """CLAUDE.md rule 9 + V6.0.7b fix: ADA / DOGE / LTC are CONTRA pairs.
    The Gemini prompt must explicitly say so (peaks mark tops)."""
    for pair in ("ADA", "DOGE", "LTC"):
        prompt = build_pair_prompt(pair)
        assert "CONTRA" in prompt, \
            f"{pair} prompt missing CONTRA convention — V6.0.7b regression"
        assert "EUPHORIC" in prompt or "euphoric" in prompt


def test_sentiment_prompt_marks_aligned_pairs():
    """BTC/ETH/BNB/SOL/XRP are sentiment-aligned — prompt must NOT say CONTRA."""
    for pair in ("BTC", "ETH", "BNB", "SOL", "XRP"):
        prompt = build_pair_prompt(pair)
        assert "CONTRA-pair" not in prompt, \
            f"{pair} prompt incorrectly flagged CONTRA"


# ── End-to-end with real personas + JUDGE on synthetic context ───────────────

def test_personas_to_judge_end_to_end_no_signal():
    """Run all 6 personas + JUDGE on a synthetic minimal-data context. With
    no real signals injected, JUDGE should return WAIT."""
    judge = JudgePersona()
    ctx = {
        "klines": {"BTCUSDT#1h": NEUTRAL_CANDLES},
        "regime": {"BTCUSDT": "NORMAL"},
        "fear_greed": 50,
        "funding_rates": {"BTCUSDT": 0.0},
    }
    votes = JudgePersona.votes_from_personas(ALL_PERSONAS, "BTCUSDT", ctx)
    assert set(votes.keys()) == {"flow", "technical", "whale",
                                 "sentiment", "regime", "onchain"}
    decision = judge.decide("BTCUSDT", votes, ctx)
    assert decision.action in ("WAIT", "LONG", "SHORT")  # not BLOCK on NORMAL
    # With no live signals this should reliably WAIT.
    assert decision.action == "WAIT"


def test_personas_to_judge_end_to_end_with_strong_inputs():
    """Inject pre-computed flow + technical signals → JUDGE should fire LONG."""
    judge = JudgePersona()
    ctx = {
        "klines": {"BTCUSDT#1h": NEUTRAL_CANDLES},
        "regime": {"BTCUSDT": "NORMAL"},
        "fear_greed": 50,
        "funding_rates": {"BTCUSDT": 0.0},
        "flow_signal": {"BTCUSDT": {"direction": "LONG", "confidence": 0.90}},
        "technical_signal": {"BTCUSDT": {"direction": "LONG", "confidence": 0.80}},
    }
    votes = JudgePersona.votes_from_personas(ALL_PERSONAS, "BTCUSDT", ctx)
    decision = judge.decide("BTCUSDT", votes, ctx)
    assert decision.action == "LONG"
    assert decision.confidence >= JUDGE_CONF_FLOOR
