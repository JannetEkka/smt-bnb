"""Session F acceptance: validation gates + faithfulness + ground-truth join.

Verifies the PLAN.md Session-F acceptance criteria:
1. Each gate computes on synthetic input; a deliberately OVERFIT study is REJECTED
   on DSR < 0 OR PBO > 0.20 OR FDR > 0.10; a genuine-edge study PASSES.
2. Conformal interval achieves ≈ target coverage on held-out synthetic.
3. Faithfulness flip detects a planted attribution (flip FLOW → decision moves;
   flip a zero-weight persona → decision unchanged).
4. Per-lane PBO: a slow-lane drawdown does not reject the fast lane.
5. Ground-truth join: an exp record with entry_price+ts gets a direction_correct.

Plus: per-persona reliability curves from persona_audit.json (CONFIRM FLOW /
DEMOTE SENTIMENT / WHALE-high-conviction-only), the input-cascade detector, the
forward-looking regime classifier, KDE CV bandwidth, and a backtest + regression
test (QA pyramid).
"""

from __future__ import annotations
import math
import os
import random

import pytest

from smt.learning.validation.dsr import deflated_sharpe
from smt.learning.validation.pbo import (
    probability_of_backtest_overfitting, per_lane_pbo, lanes_passing,
)
from smt.learning.validation.fdr import bh_fdr, candidate_fdr
from smt.learning.validation.cpcv import (
    bagged_cpcv_sharpe, combinatorial_purged_splits,
)
from smt.learning.validation.conformal import conformal_interval, empirical_coverage
from smt.learning.validation.kde import cv_bandwidth
from smt.learning.validation.gate import validate_candidate
from smt.learning.faithfulness import (
    counterfactual_persona_flip, persona_attribution, input_cascade_flag,
)
from smt.learning.groundtruth import (
    join_forward_returns, outcomes_with_ground_truth, direction_correct,
    persona_reliability_curves, recommended_weight_adjustments,
    ForwardRegimeClassifier,
)
from smt.learning.reward import direction_quality_weights
from smt.learning.synthetic import RegimeSwitchingSimulator
from smt.personas.base import JUDGE_SEED_PRIORS, PersonaVote
from smt.personas.judge import JudgePersona

PERSONA_AUDIT = "docs/data/persona_audit.json"


# ── helpers ───────────────────────────────────────────────────────────────────

def _genuine_returns(seed=42, n=750, mu=0.12):
    rng = random.Random(seed)
    return [rng.gauss(mu, 1.0) for _ in range(n)]


def _overfit_returns(seed=43, n=300):
    rng = random.Random(seed)
    return [rng.gauss(0.0, 1.0) for _ in range(n)]


def _genuine_matrix(seed=42, T=400, N=12):
    rng = random.Random(seed)
    # column 0 carries a real edge; the rest are noise.
    return [[(rng.gauss(0.15, 1.0) if c == 0 else rng.gauss(0.0, 1.0))
             for c in range(N)] for _ in range(T)]


def _overfit_matrix(seed=44, T=400, N=12):
    rng = random.Random(seed)
    return [[rng.gauss(0.0, 1.0) for _ in range(N)] for _ in range(T)]


# ── 1. DSR — deflate for trials + non-normality ───────────────────────────────

def test_dsr_rejects_overfit_passes_genuine():
    g = deflated_sharpe(_genuine_returns(), n_trials=5)
    o = deflated_sharpe(_overfit_returns(), n_trials=2000)
    assert g.dsr > 0 and g.passed and g.psr > 0.5
    assert o.dsr < 0 and not o.passed                 # below null-max benchmark
    # more trials on the SAME returns must lower DSR (the deflation works)
    few = deflated_sharpe(_genuine_returns(), n_trials=2)
    many = deflated_sharpe(_genuine_returns(), n_trials=5000)
    assert few.dsr > many.dsr


# ── 1b. PBO — CSCV ────────────────────────────────────────────────────────────

def test_pbo_rejects_overfit_passes_genuine():
    g = probability_of_backtest_overfitting(_genuine_matrix())
    o = probability_of_backtest_overfitting(_overfit_matrix())
    assert g.pbo <= 0.20 and g.passed
    assert o.pbo > 0.20 and not o.passed


# ── 1c. FDR — Benjamini-Hochberg ──────────────────────────────────────────────

def test_fdr_rejects_no_discovery_passes_genuine():
    genuine = [0.001, 0.002, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]   # two real discoveries
    overfit = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]  # nothing survives
    assert candidate_fdr(genuine) <= 0.10
    assert candidate_fdr(overfit) > 0.10
    res = bh_fdr(genuine)
    assert res.n_discoveries >= 1 and res.rejected[0] is True


# ── 1d. Combined gate — overfit rejected on DSR OR PBO OR FDR; genuine passes ──

def test_validation_gate_rejects_overfit_study():
    rep = validate_candidate(
        returns=_overfit_returns(), n_trials=2000,
        returns_matrix=_overfit_matrix(),
        per_cell_pvalues=[0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95],
    )
    assert rep.verdict == "REJECT" and not rep.passed
    assert rep.dsr < 0 and rep.pbo > 0.20 and rep.fdr > 0.10   # all three fire


def test_validation_gate_passes_genuine_study_with_activation_fields():
    rep = validate_candidate(
        returns=_genuine_returns(), n_trials=5,
        returns_matrix=_genuine_matrix(),
        per_cell_pvalues=[0.001, 0.002, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    )
    assert rep.verdict == "PASS" and rep.passed
    # activation-log fields exist + are finite floats
    for f in (rep.dsr, rep.pbo, rep.fdr):
        assert isinstance(f, float) and math.isfinite(f)


def test_validation_gate_single_breach_rejects():
    # genuine DSR + genuine FDR but an OVERFIT matrix → PBO alone must reject.
    rep = validate_candidate(
        returns=_genuine_returns(), n_trials=5,
        returns_matrix=_overfit_matrix(),
        per_cell_pvalues=[0.001, 0.002, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    )
    assert rep.verdict == "REJECT"
    assert any("PBO" in r for r in rep.reasons)


# ── 2. Conformal — calibrated coverage ────────────────────────────────────────

def test_conformal_interval_hits_target_coverage():
    rng = random.Random(7)
    calib = [rng.gauss(0.0, 1.0) for _ in range(800)]
    ci = conformal_interval(0.0, calib, confidence=0.90)
    held_out = [rng.gauss(0.0, 1.0) for _ in range(6000)]
    cov = empirical_coverage([(ci.lower, ci.upper)] * len(held_out), held_out)
    assert abs(cov - 0.90) <= 0.05


# ── 3. CPCV — purge + embargo kill leakage ────────────────────────────────────

def test_cpcv_purge_embargo_no_leak_and_full_bag():
    splits = combinatorial_purged_splits(
        120, n_groups=6, n_test_groups=2, embargo_frac=0.03, label_horizon=2)
    assert len(splits) == math.comb(6, 2)            # full combinatorial bag
    for train, test in splits:
        tset = set(test)
        # no train index within the purge window of any test index
        assert not any(any(abs(i - t) <= 2 for t in test) for i in train)
        # no train index inside the post-test embargo window
        embargo = math.ceil(0.03 * 120)
        assert not any(any(0 < (i - t) <= embargo for t in tset) for i in train)
    cp = bagged_cpcv_sharpe([random.Random(1).gauss(0.05, 1.0) for _ in range(120)])
    assert cp.n_paths == math.comb(6, 2)


# ── 4. Per-lane PBO — slow-lane drawdown must NOT reject the fast lane ─────────

def test_per_lane_pbo_slow_drawdown_does_not_reject_fast():
    per_lane = per_lane_pbo({"fast": _genuine_matrix(), "slow": _overfit_matrix()})
    passing = lanes_passing(per_lane)
    assert passing["fast"] is True               # healthy fast scalp book survives
    assert passing["slow"] is False              # slow lane correctly flagged
    assert per_lane["fast"].pbo <= 0.20 < per_lane["slow"].pbo


# ── 5. Faithfulness flip — planted attribution ────────────────────────────────

def test_faithfulness_flip_moves_decision_for_real_persona():
    judge = JudgePersona()
    ctx = {"fear_greed": 50, "regime": {"BNB": "RANGING"}}
    votes = {"technical": PersonaVote("LONG", 0.9)}      # alone: below the floor → WAIT
    fr = counterfactual_persona_flip(judge, "BNB", votes, ctx, "flow", "LONG", 0.95)
    assert fr.baseline_action == "WAIT" and fr.flipped_action == "LONG"
    assert fr.moved and fr.delta > 0


def test_faithfulness_zero_weight_persona_leaves_decision_unchanged():
    judge = JudgePersona()
    ctx = {"fear_greed": 50, "regime": {"BNB": "RANGING"}}
    # BNB SENTIMENT weight is 0.0 in PAIR_JUDGE_PERSONA_WEIGHTS → flipping it is inert.
    votes = {"technical": PersonaVote("LONG", 0.9), "flow": PersonaVote("LONG", 0.95)}
    fr = counterfactual_persona_flip(judge, "BNB", votes, ctx, "sentiment", "SHORT", 0.95)
    assert fr.baseline_action == fr.flipped_action == "LONG"
    assert fr.delta == 0.0 and not fr.moved
    attr = persona_attribution(judge, "BNB", votes, ctx)
    assert attr["sentiment"] == 0.0           # zero-weight → no influence
    assert attr["flow"] > attr["sentiment"]   # FLOW is the real lever


# ── 6. Input-cascade detector ─────────────────────────────────────────────────

def _rec(sigs):
    return {"persona_votes": {k: {"signal": v, "confidence": 0.8} for k, v in sigs.items()}}


def test_input_cascade_flags_correlated_inputs():
    # every persona keys off the same broken feed → unanimous, but not independent.
    recs = [_rec({"flow": d, "technical": d, "whale": d})
            for d in (["LONG", "SHORT"] * 30)]
    rep = input_cascade_flag(recs)
    assert rep.flagged and rep.mean_corr >= 0.9


def test_input_cascade_passes_independent_personas():
    rng = random.Random(1)
    recs = [_rec({p: rng.choice(["LONG", "SHORT"]) for p in ("flow", "technical", "whale")})
            for _ in range(300)]
    assert not input_cascade_flag(recs).flagged


# ── 7. Ground-truth +2h/+4h join ──────────────────────────────────────────────

def test_ground_truth_join_sets_direction_correct():
    exp = {"pair": "BTC", "direction": "LONG", "entry_price": 100.0,
           "ts": "2026-05-01T00:00:00+00:00", "regime": "BULLISH",
           "conviction": 0.7, "fear_greed": 55, "pnl_usd": 12.0}
    rose = lambda pair, ts: {"h2": 101.0, "h4": 104.0}     # noqa: E731
    joined = join_forward_returns([exp], rose)
    assert joined[0]["direction_correct"] is True
    assert joined[0]["direction_correct_h2"] is True
    fell = lambda pair, ts: {"h2": 99.0, "h4": 96.0}       # noqa: E731
    assert join_forward_returns([exp], fell)[0]["direction_correct"] is False
    # SHORT direction logic is mirrored
    assert direction_correct("SHORT", 100.0, 96.0) is True
    assert direction_correct("WAIT", 100.0, 96.0) is None


def test_ground_truth_join_feeds_direction_quality_weights():
    rng = random.Random(3)
    recs, fetch_table = [], {}
    for i in range(30):
        ts = f"2026-05-0{1 + i % 9}T00:00:00+00:00"
        up = i < 24                                   # 80% of LONGs actually rose
        recs.append({"pair": "BTC", "direction": "LONG", "entry_price": 100.0,
                     "ts": ts, "regime": "BULLISH", "conviction": 0.9,
                     "fear_greed": 55, "pnl_usd": 5.0, "_up": up})
    def fetch(pair, ts, _recs=recs):
        # find the record with this ts (test stub — deterministic)
        for r in _recs:
            if r["ts"] == ts:
                return {"h4": 105.0 if r["_up"] else 95.0}
        return None
    # give each record a distinct ts so the stub resolves uniquely
    for i, r in enumerate(recs):
        r["ts"] = f"2026-05-01T00:{i:02d}:00+00:00"
    joined = join_forward_returns(recs, lambda p, ts: {"h4": 105.0 if any(
        rr["ts"] == ts and rr["_up"] for rr in recs) else 95.0})
    outs = outcomes_with_ground_truth(joined)
    weights = direction_quality_weights(outs)
    assert weights                                    # a real +Nh posterior exists
    assert all(0.0 <= w <= 1.0 for w in weights.values())


# ── 8. Per-persona reliability curves (authoritative, from persona_audit.json) ─

@pytest.mark.skipif(not os.path.exists(PERSONA_AUDIT), reason="persona_audit.json absent")
def test_persona_reliability_curves_confirm_flow_demote_sentiment():
    curves = persona_reliability_curves()
    assert curves["flow"].acc_h4 > 0.60                       # FLOW is the edge
    assert curves["sentiment"].hc_acc_h4 < 0.50               # SENTIMENT anti-predictive @ high conv
    assert curves["whale"].hc_acc_h4 > curves["whale"].acc_h4 # WHALE strong only when confident
    assert curves["whale"].monotonic


# ── 9. Forward-looking regime classifier (deferred-from-C rewrite) ─────────────

def test_forward_regime_classifier_learns_forward_edge():
    rng = random.Random(7)
    X, y = [], []
    for _ in range(400):
        up = 1 if rng.random() < 0.5 else 0
        X.append([up * 1.5 + rng.gauss(0, 1.0), rng.gauss(0, 1)])  # feat[0] predicts +Nh
        y.append(up)
    clf = ForwardRegimeClassifier(seed=0).fit(X[:300], y[:300])
    assert clf.accuracy(X[300:], y[300:]) > 0.65          # learned a FORWARD edge


# ── 10. KDE CV bandwidth ──────────────────────────────────────────────────────

def test_kde_cv_bandwidth_positive_and_finite():
    rng = random.Random(5)
    bw = cv_bandwidth([rng.gauss(0, 1) for _ in range(80)])
    assert bw > 0.0 and math.isfinite(bw)


# ── QA pyramid: backtest + regression (Session F acceptance) ──────────────────

def test_backtest_synthetic_book_passes_validation_gates():
    """BACKTEST: replay the planted-edge synthetic market under JUDGE seed priors;
    the validated FLOW-heavy book must survive DSR + PBO + FDR."""
    sim = RegimeSwitchingSimulator(seed=1)
    configs = [
        JUDGE_SEED_PRIORS,
        {**JUDGE_SEED_PRIORS, "flow": 0.0},                   # ablate the real edge
        {"flow": 0.05, "technical": 0.05, "whale": 0.1,
         "onchain": 0.1, "sentiment": 0.6, "regime": 0.1},    # over-weight noise
        {p: 1 / 6 for p in JUDGE_SEED_PRIORS},
    ]
    books = [sim.simulate_book(c, n=600) for c in configs]
    returns = [o.net_pnl_usd for o in books[0]]
    matrix = [[books[c][t].net_pnl_usd for c in range(len(configs))]
              for t in range(len(books[0]))]
    pvals = []
    for s in range(1, 9):
        xs = [o.net_pnl_usd for o in RegimeSwitchingSimulator(seed=s).simulate_book(
            JUDGE_SEED_PRIORS, n=600)]
        mu = sum(xs) / len(xs)
        sd = (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
        from smt.learning.validation._stats import norm_cdf
        t = mu / (sd / math.sqrt(len(xs))) if sd > 0 else 0.0
        pvals.append(1.0 - norm_cdf(t))
    rep = validate_candidate(returns=returns, n_trials=8,
                             returns_matrix=matrix, per_cell_pvalues=pvals)
    assert rep.verdict == "PASS"


@pytest.mark.skipif(not os.path.exists(PERSONA_AUDIT), reason="persona_audit.json absent")
def test_regression_persona_weight_verdicts_locked():
    """REGRESSION: lock the known-good weight verdicts derived from the committed
    persona_audit.json so a future audit/threshold change can't silently drift them."""
    recs = recommended_weight_adjustments()
    assert recs["flow"] == "CONFIRM"
    assert recs["sentiment"] == "DEMOTE"
    assert recs["whale"] == "TRUST_HIGH_CONV_ONLY"
    assert recs["technical"] == "LEAVE"
