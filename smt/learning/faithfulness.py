"""Counterfactual persona-flip faithfulness check (Session F).

Holding the context fixed, flip ONE persona's vote and re-run JudgePersona.decide.
If the decision does NOT move in the predicted direction (or a zero-weight persona
DOES move it), the JUDGE attribution is unfaithful — fix the weighting before
shipping it, and never hand retail an attribution the flip can't confirm
(Session H consumes this).

Three tools:
- ``counterfactual_persona_flip`` — one flip, returns the signed swing + a
  ``moved`` verdict against the predicted direction.
- ``persona_attribution`` — flip each persona LONG vs SHORT; the magnitude of the
  JUDGE swing is that persona's causal influence. RUN THIS to confirm FLOW (large
  swing) and demote SENTIMENT / WHALE-low-conviction (small or veto-only swing)
  BEFORE shipping weights (AUTOPSY Findings 2/3, CLAUDE.md rule 9).
- ``input_cascade_flag`` — over a set of decisions, flag when personas agree only
  because their INPUTS are correlated (near-perfect vote correlation). This would
  have caught V3.2.124's 7%-accuracy "wide net of wrong calls" — unanimous, but
  unanimous because every persona keyed off the same broken feed.

The JUDGE is called read-only; never raises.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from smt.personas.base import JUDGE_SEED_PRIORS, PersonaVote

log = logging.getLogger("smt.learning.faithfulness")

_DEFAULT_TOL = 1e-9
_PERSONAS = list(JUDGE_SEED_PRIORS.keys())


@dataclass
class FlipResult:
    persona: str
    flipped_to: str
    baseline_action: str
    flipped_action: str
    baseline_score: float       # signed JUDGE confidence (+LONG / −SHORT)
    flipped_score: float
    delta: float                # flipped − baseline (signed)
    moved: bool                 # decision moved as the flip predicted
    magnitude: float            # |delta|


def _signed_score(decision: Any) -> float:
    """Signed JUDGE confidence: +conf for LONG, −conf for SHORT, 0 for BLOCK.

    For WAIT we still surface the *leaning* (the JUDGE computes a directional
    score even below the floor) by reading the sign of the persona_breakdown sum,
    so a flip that strengthens a direction without crossing the floor is still
    visible as a faithful movement.
    """
    action = getattr(decision, "action", "WAIT")
    conf = float(getattr(decision, "confidence", 0.0) or 0.0)
    if action == "LONG":
        return +conf
    if action == "SHORT":
        return -conf
    if action == "BLOCK":
        return 0.0
    breakdown = getattr(decision, "persona_breakdown", {}) or {}
    net = sum(float(v) for v in breakdown.values())
    sign = 1.0 if net > 0 else -1.0 if net < 0 else 0.0
    return sign * conf


def _predicted_sign(direction: str) -> float:
    return {"LONG": 1.0, "SHORT": -1.0}.get(str(direction).upper(), 0.0)


def counterfactual_persona_flip(
    judge: Any,
    pair: str,
    votes: Dict[str, PersonaVote],
    context: Dict[str, Any],
    persona: str,
    to_direction: str = "LONG",
    to_confidence: float = 0.9,
    tol: float = _DEFAULT_TOL,
) -> FlipResult:
    """Flip ONE persona to (to_direction, to_confidence); measure the JUDGE swing."""
    persona = persona.lower()
    base = judge.decide(pair, votes, context)
    flipped_votes = dict(votes or {})
    flipped_votes[persona] = PersonaVote(
        direction=to_direction, confidence=to_confidence, reasoning="counterfactual flip")
    flipped = judge.decide(pair, flipped_votes, context)

    base_s = _signed_score(base)
    flip_s = _signed_score(flipped)
    delta = flip_s - base_s
    sign = _predicted_sign(to_direction)
    if sign > 0:
        moved = delta > tol
    elif sign < 0:
        moved = delta < -tol
    else:
        moved = abs(delta) <= tol
    log.debug("[FAITHFULNESS] flip %s→%s on %s: %.3f→%.3f (Δ=%.3f moved=%s)",
              persona, to_direction, pair, base_s, flip_s, delta, moved)
    return FlipResult(
        persona=persona, flipped_to=to_direction.upper(),
        baseline_action=base.action, flipped_action=flipped.action,
        baseline_score=base_s, flipped_score=flip_s,
        delta=delta, moved=moved, magnitude=abs(delta),
    )


def persona_attribution(
    judge: Any,
    pair: str,
    votes: Dict[str, PersonaVote],
    context: Dict[str, Any],
    personas: Optional[Sequence[str]] = None,
    confidence: float = 0.9,
) -> Dict[str, float]:
    """Causal influence per persona = |JUDGE swing from flipping it LONG vs SHORT|.

    A persona the JUDGE genuinely weights (FLOW) yields a large swing; a
    zero-weight or veto-only persona yields ~0 — the flip leaves the decision
    unchanged, which is exactly the faithfulness signal.
    """
    personas = list(personas or _PERSONAS)
    out: Dict[str, float] = {}
    for p in personas:
        long_v = dict(votes or {})
        long_v[p.lower()] = PersonaVote("LONG", confidence, "attrib+")
        short_v = dict(votes or {})
        short_v[p.lower()] = PersonaVote("SHORT", confidence, "attrib-")
        long_s = _signed_score(judge.decide(pair, long_v, context))
        short_s = _signed_score(judge.decide(pair, short_v, context))
        out[p.lower()] = abs(long_s - short_s)
    return out


# ── Input-cascade detector ─────────────────────────────────────────────────────

@dataclass
class CascadeReport:
    flagged: bool
    mean_corr: float
    pair_corr: Dict[str, float] = field(default_factory=dict)
    cluster: List[str] = field(default_factory=list)


def _dir_value(signal: str) -> int:
    s = str(signal).upper()
    return 1 if s == "LONG" else -1 if s == "SHORT" else 0


def _persona_series(records: Sequence[Dict[str, Any]], persona: str) -> List[int]:
    out: List[int] = []
    for rec in records:
        votes = rec.get("persona_votes") or {}
        v = votes.get(persona) or {}
        out.append(_dir_value(v.get("signal", "NEUTRAL")))
    return out


def input_cascade_flag(
    records: Sequence[Dict[str, Any]],
    personas: Optional[Sequence[str]] = None,
    corr_threshold: float = 0.9,
    min_personas: int = 3,
) -> CascadeReport:
    """Flag a correlated-input cascade across exp records' persona votes.

    When ≥ ``min_personas`` personas move together with mean pairwise direction
    correlation ≥ ``corr_threshold``, their agreement is NOT independent evidence
    — it's one signal wearing several hats (the V3.2.124 failure mode).
    """
    from smt.learning.validation._stats import pearson_corr  # local: pure-stats dep

    personas = list(personas or _PERSONAS)
    series = {p: _persona_series(records, p) for p in personas}
    # Keep only personas that actually vote (non-degenerate direction series).
    active = [p for p in personas if len(set(series[p])) > 1]
    pair_corr: Dict[str, float] = {}
    corrs: List[float] = []
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            a, b = active[i], active[j]
            c = pearson_corr(series[a], series[b])
            pair_corr[f"{a}~{b}"] = c
            corrs.append(c)
    mean_corr = sum(corrs) / len(corrs) if corrs else 0.0
    cluster = [f"{k}" for k, c in pair_corr.items() if c >= corr_threshold]
    flagged = len(active) >= min_personas and mean_corr >= corr_threshold
    if flagged:
        log.info("[FAITHFULNESS] input-cascade flagged: mean_corr=%.2f cluster=%s",
                 mean_corr, cluster)
    return CascadeReport(flagged=flagged, mean_corr=mean_corr,
                         pair_corr=pair_corr, cluster=active if flagged else [])
