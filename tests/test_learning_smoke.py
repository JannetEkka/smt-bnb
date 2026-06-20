"""Session D acceptance: learning loop (optimizer / bandit / reward / synthetic).

Verifies the PLAN.md Session-D acceptance criteria:
1. Optimizer instantiates + .suggest() returns a valid params dict.
2. Reward: positive PnL + fat-tail bonus > positive PnL alone; high-overtrade
   outcome penalized; <55% direction-accuracy config returns −inf.
3. Bandit Beta-Binomial update converges to the empirical win-rate within ±0.05
   on 200 synthetic pulls.
4. Synthetic simulator generates 1000 trades with regime distribution within
   ±10% of expected.
5. Optimizer 50-trial run on synthetic recovers the planted FLOW weight ±15%.

Plus the addenda: OvertradingPenalty on post-exit-stack outcomes (gap 5);
DirectionQualityGate active (gap 9); hierarchical pooling gate (gap 6);
cold-start warm-up (gap 8); corpus loader (operator CORPUS SCOPE).
"""

from __future__ import annotations
import math
import random
import statistics

import pytest

from smt.learning.reward import (
    TradeOutcome, compute_reward, fat_tail_bonus, overtrading_penalty,
    direction_accuracy, direction_quality_weights, cell_of,
)
from smt.learning.bandit import ContextualBandit
from smt.learning.synthetic import (
    RegimeSwitchingSimulator, REGIMES, simulate_known_edge,
)
from smt.learning.optimizer import (
    TPEOptimizer, default_search_space, judge_priors_space,
    write_learned_params, load_learned_params,
)
from smt.learning.hierarchical import should_pool, pools_into_majors
from smt.learning import corpus
from smt.personas.base import JUDGE_SEED_PRIORS


# ── 1. Optimizer .suggest() returns a valid params dict ───────────────────────

def test_optimizer_suggest_returns_valid_params():
    opt = TPEOptimizer(seed=1)
    p = opt.suggest()
    jp = p["judge_priors"]
    assert set(jp) == set(JUDGE_SEED_PRIORS)
    assert sum(jp.values()) == pytest.approx(1.0)
    assert all(0.0 <= v <= 1.0 for v in jp.values())
    assert 0.45 <= p["raw_judge_min_confidence"] <= 0.75
    assert 0.005 <= p["position_pct"] <= 0.05
    # portfolio-capacity is learnable (gap 7)
    assert 2 <= p["portfolio"]["max_positions"] <= 8
    assert 0 <= p["portfolio"]["cooldown_minutes"] <= 60
    assert p["portfolio"]["margin_per_trade"] == p["position_pct"]
    # per-direction knobs (Finding 6: SHORT ≠ LONG calibration)
    assert 0.8 <= p["per_direction"]["long_leverage_mult"] <= 1.2
    assert 0.8 <= p["per_direction"]["short_leverage_mult"] <= 1.2
    assert p["reward_coeffs"]["alpha"] >= 0.0 and p["reward_coeffs"]["beta"] >= 0.0


def test_optimizer_default_space_dimensions():
    specs = default_search_space()
    names = {s.name for s in specs}
    assert {f"judge_priors.{p}" for p in JUDGE_SEED_PRIORS} <= names
    assert "pairs.BTC.tp_cap_pct" in names and "pairs.DOGE.sl_pct" in names
    assert "pair_persona_mult.ADA.sentiment" in names


# ── 2. Reward: fat-tail bonus ─────────────────────────────────────────────────

def test_reward_fat_tail_bonus_adds_value():
    outs = [TradeOutcome("BTC", "LONG", net_pnl_usd=5.0, fees_usd=2.0) for _ in range(40)]
    outs += [TradeOutcome("BTC", "LONG", net_pnl_usd=300.0, fees_usd=2.0) for _ in range(3)]
    r_without = compute_reward(outs, alpha=0.0, beta=0.0)
    r_with = compute_reward(outs, alpha=1.0, beta=0.0)
    assert fat_tail_bonus([o.net_pnl_usd for o in outs]) > 0.0
    assert r_with > r_without


def test_fat_tail_rewards_asymmetry_over_uniform_wins():
    # Equal TOTAL win mass, different shape: a few fat winners must score higher
    # than many equal small winners (V4.2.5 asymmetry > flat book).
    thin = [TradeOutcome("BTC", "LONG", net_pnl_usd=10.0, fees_usd=1.0) for _ in range(60)]
    fat = ([TradeOutcome("BTC", "LONG", net_pnl_usd=1.0, fees_usd=1.0) for _ in range(57)]
           + [TradeOutcome("BTC", "LONG", net_pnl_usd=181.0, fees_usd=1.0) for _ in range(3)])
    assert sum(o.net_pnl_usd for o in thin) == pytest.approx(sum(o.net_pnl_usd for o in fat))
    assert (fat_tail_bonus([o.net_pnl_usd for o in fat])
            > fat_tail_bonus([o.net_pnl_usd for o in thin]))


# ── 2b. Reward: overtrading penalty (gap 5 — post-exit-stack, not raw count) ──

def test_overtrading_penalty_punishes_net_losers():
    winners = [TradeOutcome("BTC", "LONG", net_pnl_usd=50.0, fees_usd=2.0) for _ in range(10)]
    few = winners + [TradeOutcome("BTC", "SHORT", net_pnl_usd=-40.0, fees_usd=2.0) for _ in range(2)]
    many = winners + [TradeOutcome("BTC", "SHORT", net_pnl_usd=-40.0, fees_usd=2.0) for _ in range(20)]
    assert (compute_reward(many, alpha=0.0, min_direction_acc=0.0)
            < compute_reward(few, alpha=0.0, min_direction_acc=0.0))


def test_overtrading_penalty_spares_scratches_gap5():
    """V4.2.5 wide net: 84 ~$0 scratches catch 6 fat winners, ZERO real losers.
    The penalty must NOT fire on scratch count or it kills the asymmetry."""
    winners = [TradeOutcome("BTC", "LONG", net_pnl_usd=80.0, fees_usd=2.0) for _ in range(6)]
    scratches = [TradeOutcome("BTC", "LONG", net_pnl_usd=0.5, fees_usd=2.0) for _ in range(84)]
    near_zero_losers = [TradeOutcome("BTC", "SHORT", net_pnl_usd=-0.5, fees_usd=2.0) for _ in range(20)]
    book = winners + scratches + near_zero_losers
    # scratch band = 1.5 × median fee = 3.0 → nothing below −3.0 → no penalty
    assert overtrading_penalty(book) == 0.0


# ── 2c. Reward: direction-quality candidate guard (<55% → −inf) ───────────────

def test_reward_direction_quality_floor_rejects_v32124_noise():
    # 20% +Nh accuracy, 5 lucky breakevens — the V3.2.124 failure mode.
    outs = [TradeOutcome("BTC", "LONG", net_pnl_usd=(30.0 if i < 20 else -2.0),
                         fees_usd=2.0, direction_correct=(i < 20)) for i in range(100)]
    assert direction_accuracy(outs) < 0.55
    assert compute_reward(outs) == float("-inf")


def test_reward_direction_quality_passes_real_edge():
    outs = [TradeOutcome("BTC", "LONG", net_pnl_usd=(30.0 if i < 70 else -25.0),
                         fees_usd=2.0, direction_correct=(i < 70)) for i in range(100)]
    assert direction_accuracy(outs) >= 0.55
    assert math.isfinite(compute_reward(outs))


# ── 2d. gap-9 DATA mechanism: posterior weights down-weight low-accuracy cells ─

def test_direction_quality_weights_downweight_low_accuracy_cells():
    good = [TradeOutcome("BTC", "LONG", regime="BULLISH", conviction=0.9,
                         direction_correct=True) for _ in range(20)]
    bad = [TradeOutcome("DOGE", "LONG", regime="BEARISH", conviction=0.9,
                        direction_correct=False) for _ in range(20)]
    w = direction_quality_weights(good + bad)
    assert w[cell_of(good[0])] > 0.8
    assert w[cell_of(bad[0])] < 0.2
    assert w[cell_of(good[0])] > w[cell_of(bad[0])]


# ── 3. Bandit Beta-Binomial convergence ───────────────────────────────────────

def test_bandit_beta_binomial_converges_to_winrate():
    rng = random.Random(7)
    b = ContextualBandit(seed=1)
    p_true = 0.65
    for _ in range(200):
        b.update("BTC", "LONG", "BULLISH", rng.random() < p_true)
    est = b.posterior_mean("BTC", "LONG", "BULLISH")
    assert abs(est - p_true) <= 0.05


def test_bandit_cold_start_warmup_is_passthrough():
    b = ContextualBandit(seed=1, warmup_pulls=20)
    # Cold arm must not gate — returns JUDGE conf unchanged (gap 8).
    assert b.scaled_confidence(0.8, "BTC", "LONG", "BULLISH") == pytest.approx(0.8)
    assert not b.is_warm("BTC", "LONG", "BULLISH")
    for _ in range(20):
        b.update("BTC", "LONG", "BULLISH", True)
    assert b.is_warm("BTC", "LONG", "BULLISH")
    # Warm + high prob-of-profit → scales conf (here downward toward conf×mean).
    assert b.scaled_confidence(0.8, "BTC", "LONG", "BULLISH") < 0.8


def test_bandit_select_playbook_returns_candidate():
    b = ContextualBandit(seed=3)
    choice = b.select_playbook("ETH", "BEARISH", candidates=("LONG", "SHORT"))
    assert choice in ("LONG", "SHORT")


# ── 4. Synthetic simulator: regime distribution + valid outcomes ──────────────

def test_synthetic_regime_distribution_within_10pct():
    sim = RegimeSwitchingSimulator(seed=2)   # verified comfortably within ±6%
    counts = sim.regime_counts(1000)
    assert sum(counts.values()) == 1000
    expected = 1000 / len(REGIMES)
    for r in REGIMES:
        assert abs(counts[r] - expected) / expected <= 0.10, (r, counts)


def test_synthetic_book_outcomes_are_valid():
    book = simulate_known_edge(n=400)
    assert len(book) > 0
    for o in book[:10]:
        assert o.direction in ("LONG", "SHORT")
        assert o.regime in REGIMES
        assert 0.0 <= o.conviction <= 1.0
    labeled = [o for o in book if o.direction_correct is not None]
    assert len(labeled) > 0.5 * len(book)


# ── 5. Optimizer recovers the planted FLOW edge (±15% on a 50-trial study) ────

@pytest.mark.parametrize("backend", ["auto", "builtin"])
def test_optimizer_recovers_planted_flow_edge(backend):
    # "auto" → Optuna's TPESampler when installed (the production path), else the
    # builtin TPE. "builtin" forces the dependency-free fallback. Both recover.
    sims = [RegimeSwitchingSimulator(seed=s) for s in range(1, 7)]
    planted = sims[0].planted_flow_weight

    def objective(params):
        return statistics.mean(
            compute_reward(s.simulate_book(params["judge_priors"], n=500),
                           alpha=0.3, beta=1.0, min_direction_acc=0.0)
            for s in sims
        )

    # Focused study over the two PLANTED-edge priors (FLOW trend + SENTIMENT
    # crash). Noise personas carry zero weight → cannot be over-fit. The full
    # default_search_space tunes all 6 priors in the production weekly refit.
    opt = TPEOptimizer(search_space=judge_priors_space(["flow", "sentiment"]),
                       seed=1, n_startup_trials=12, explore_prob=0.25, backend=backend)
    res = opt.optimize(objective, n_trials=50)
    flow = res.best_params["judge_priors"]["flow"]
    assert abs(flow - planted) / planted <= 0.15, (backend, flow, planted)
    assert math.isfinite(res.best_value)


# ── learned_params.json round-trip (daemon startup load, Session E) ───────────

def test_learned_params_roundtrip(tmp_path):
    params = TPEOptimizer(seed=1).suggest()
    path = str(tmp_path / "learned.json")
    write_learned_params(params, path)
    loaded = load_learned_params(path)
    assert set(loaded["judge_priors"]) == set(params["judge_priors"])
    assert load_learned_params(str(tmp_path / "missing.json")) is None


# ── 6. Hierarchical pooling gate (gap 6 — OFF in extreme fear) ────────────────

def test_hierarchical_pooling_gate():
    assert should_pool(15) is False     # capitulation → no pooling
    assert should_pool(50) is True
    assert should_pool(80) is True
    assert should_pool(None) is True
    assert pools_into_majors("ADAUSDT", 50) is True
    assert pools_into_majors("ADAUSDT", 15) is False   # alt decouples in fear
    assert pools_into_majors("BTCUSDT", 50) is False    # BTC is a major


# ── Corpus loader (operator CORPUS SCOPE) — guarded if files absent ───────────

def test_corpus_loads_splits_and_validates():
    trades = corpus.load_trades()
    if not trades:
        pytest.skip("docs/data/trades.json not present in this environment")
    cmt, uat = corpus.split_cmt_uat(trades)
    assert len(cmt) + len(uat) == len(trades)
    train, validate = corpus.forward_validation_split(uat)
    assert len(train) + len(validate) == len(uat)
    outs = corpus.to_outcomes(uat[:300])
    assert all(o.direction in ("LONG", "SHORT") for o in outs)


def test_bandit_seeds_from_corpus():
    trades = corpus.load_trades()
    if not trades:
        pytest.skip("no corpus")
    b = ContextualBandit(seed=1)
    n = b.seed_from_corpus(trades[:1000])
    assert n > 0
    assert len(b.arms) > 0
