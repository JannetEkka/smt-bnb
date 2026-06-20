"""Reward function the optimizer maximizes.

    R = NetPnL  +  α · FatTailBonus  −  β · OvertradingPenalty

- NetPnL: realized USD already NET of round-trip fees (post-exit-stack).
  Never use gross. Fees are real money on every flip.
- FatTailBonus: KDE-weighted top-5% tail of the win distribution (V4.2.5
  asymmetry strategy). A book whose wins are concentrated in a few fat
  winners scores higher than an equal-total book of many small wins.
  KDE via smt.learning.validation.kde.
- OvertradingPenalty (AUTOPSY gap 5): computed on POST-EXIT-STACK outcomes
  (net losers), NOT raw trade count. V4.2.5 had 84 scratch-trades at ~$0
  EV that were the *wide net* catching 17 fat winners — if the penalty
  fired on raw count it would kill the asymmetry. It fires on the genuine
  net-loser bleed (|loss| + fees) so scratches survive.

α, β are themselves learnable inside the outer Optuna loop.

── Direction quality (AUTOPSY gap 9, revised 2026-06-07) ──────────────────
Two distinct mechanisms — do not conflate:

1. DirectionQualityWeight (the DATA mechanism). The learner sees the FULL
   corpus (every cell, wins AND losses) and computes a Beta posterior of
   +2h direction accuracy per cell (pair × direction × regime × F&G band ×
   conviction bucket). Low-accuracy cells get LOW posterior weight from the
   data itself — they are NOT excluded. Bias-by-exclusion is the failure
   mode the operator named. `direction_quality_weights()` implements this;
   the fat-tail credit is scaled by it.

2. min_direction_acc floor (the CANDIDATE GUARD). A whole candidate
   parameterization whose *simulated book* is net directionally wrong
   (< min_direction_acc at +2h, e.g. the V3.2.124 7%-accuracy model) gets
   reward −inf so the optimizer never chases it. This rejects a candidate,
   it does NOT exclude any data from training. Keep both.
"""

from __future__ import annotations
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

from smt.core.risk import ROUND_TRIP_FEE_PCT
from smt.personas.base import bare_pair, fng_band, regime_bucket
from smt.learning.validation.kde import kde_quantile

log = logging.getLogger("smt.learning.reward")

# Minimum +2h direction accuracy a candidate's book must clear (gap 9 guard).
# Learnable, but this is the default operating point the operator validated.
DEFAULT_MIN_DIRECTION_ACC = 0.55
# EV-per-trade must clear this multiple of round-trip fees (CLAUDE.md rule 6).
DEFAULT_EV_FEE_MULTIPLE = 1.5

Cell = Tuple[str, str, str, str, str]


@dataclass
class TradeOutcome:
    """A realized, POST-EXIT-STACK trade outcome — what the reward consumes.

    `net_pnl_usd` is realized PnL already net of round-trip fees. A `scratch`
    is a near-$0 outcome (the wide net); the overtrading penalty must NOT
    punish scratches (gap 5), only genuine net losers.
    """
    pair: str
    direction: str                       # LONG / SHORT
    regime: str = "NORMAL"
    lane: str = "fast"
    net_pnl_usd: float = 0.0             # realized, net of fees, post-exit-stack
    fees_usd: float = 0.0
    direction_correct: Optional[bool] = None   # +2h ground-truth direction match
    conviction: float = 0.0             # JUDGE confidence at entry
    fng: Optional[int] = None           # F&G (CMC) at entry


Outcomes = Union[TradeOutcome, Sequence[TradeOutcome]]


def _as_list(outcomes: Outcomes) -> List[TradeOutcome]:
    if isinstance(outcomes, TradeOutcome):
        return [outcomes]
    return list(outcomes)


def _conviction_bucket(conf: float) -> str:
    c = float(conf or 0.0)
    if c >= 0.80:
        return "high"
    if c >= 0.60:
        return "med"
    return "low"


def cell_of(o: TradeOutcome) -> Cell:
    """The (pair × dir × regime × F&G band × conviction) cell of an outcome."""
    return (
        bare_pair(o.pair),
        (o.direction or "").upper(),
        regime_bucket(o.regime),
        fng_band(o.fng),
        _conviction_bucket(o.conviction),
    )


def direction_accuracy(outcomes: Outcomes) -> Optional[float]:
    """Fraction of outcomes (with a known +2h direction) that were correct.

    Returns None when no outcome carries direction-correctness info — in that
    case the candidate guard cannot fire and reward is computed on PnL alone.
    """
    outs = [o for o in _as_list(outcomes) if o.direction_correct is not None]
    if not outs:
        return None
    return sum(1 for o in outs if o.direction_correct) / len(outs)


def direction_quality_weights(
    outcomes: Outcomes, prior_a: float = 1.0, prior_b: float = 1.0
) -> Dict[Cell, float]:
    """Beta-posterior +2h direction accuracy per cell (gap-9 DATA mechanism).

    weight(cell) = (prior_a + wins) / (prior_a + prior_b + n). Low-accuracy
    cells get low weight from the data — never excluded. With the default
    Beta(1,1) prior an all-correct cell tends to 1.0, an all-wrong cell to 0.0,
    and a thin/unknown cell shrinks toward 0.5.
    """
    wins: Dict[Cell, int] = defaultdict(int)
    tot: Dict[Cell, int] = defaultdict(int)
    for o in _as_list(outcomes):
        if o.direction_correct is None:
            continue
        c = cell_of(o)
        tot[c] += 1
        if o.direction_correct:
            wins[c] += 1
    return {
        c: (prior_a + wins[c]) / (prior_a + prior_b + tot[c])
        for c in tot
    }


def fat_tail_bonus(
    net_pnls: Sequence[float],
    weights: Optional[Sequence[float]] = None,
    tail_frac: float = 0.05,
) -> float:
    """KDE-thresholded top-`tail_frac` tail mass of the win distribution.

    A few fat winners → large bonus; many small equal winners (same total) →
    small bonus (only the top slice counts). The KDE supplies a smoothed
    (1−tail_frac) quantile as the tail threshold; we floor at the top-k order
    statistics so a peaked/degenerate distribution doesn't sweep in every win.
    `weights` (per-trade DirectionQualityWeight) scale each tail winner's
    contribution so fat tails from low-accuracy cells count less.
    """
    pnls = list(net_pnls)
    if weights is None:
        weights = [1.0] * len(pnls)
    wins = [(p, w) for p, w in zip(pnls, weights) if p > 0]
    if not wins:
        return 0.0
    k = max(1, math.ceil(tail_frac * len(wins)))
    if len(wins) < 3:
        return float(sum(p * w for p, w in wins))
    q = kde_quantile([p for p, _ in wins], 1.0 - tail_frac)
    tail = [(p, w) for p, w in wins if p >= q]
    if len(tail) < k:   # peaked/degenerate → take the top-k by value
        tail = sorted(wins, key=lambda pw: pw[0])[-k:]
    return float(sum(p * w for p, w in tail))


def overtrading_penalty(
    outcomes: Outcomes,
    scratch_band_usd: Optional[float] = None,
    ev_fee_multiple: float = DEFAULT_EV_FEE_MULTIPLE,
) -> float:
    """Bleed from GENUINE net losers — not raw trade count (AUTOPSY gap 5).

    A scratch is |net_pnl| ≤ scratch_band (default 1.5× the median round-trip
    fee, per CLAUDE.md rule 6). Scratches are the wide net that catches fat
    winners — they are NOT penalized. Only trades that lost MORE than a scratch
    contribute their (|loss| + fees) to the penalty.
    """
    outs = _as_list(outcomes)
    if not outs:
        return 0.0
    fees = [abs(o.fees_usd) for o in outs if o.fees_usd]
    if scratch_band_usd is None:
        med_fee = sorted(fees)[len(fees) // 2] if fees else 0.0
        scratch_band_usd = max(ev_fee_multiple * med_fee, 1e-9)
    losers = [o for o in outs if o.net_pnl_usd < -scratch_band_usd]
    return float(sum(abs(o.net_pnl_usd) + abs(o.fees_usd) for o in losers))


def compute_reward(
    outcomes: Outcomes,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    min_direction_acc: float = DEFAULT_MIN_DIRECTION_ACC,
    scratch_band_usd: Optional[float] = None,
    ev_fee_multiple: float = DEFAULT_EV_FEE_MULTIPLE,
    apply_direction_quality: bool = True,
) -> float:
    """R = NetPnL + α·FatTailBonus − β·OvertradingPenalty.

    Returns −inf if the candidate's +2h direction accuracy is below
    `min_direction_acc` (gap-9 candidate guard — rejects the parameterization,
    excludes no data).
    """
    outs = _as_list(outcomes)
    if not outs:
        return 0.0

    net_pnl = sum(o.net_pnl_usd for o in outs)

    # ── gap-9 candidate guard: reject net-directionally-wrong parameterizations
    acc = direction_accuracy(outs)
    if acc is not None and acc < min_direction_acc:
        log.info("[REWARD] candidate REJECTED dir_acc=%.3f < floor=%.2f → -inf",
                 acc, min_direction_acc)
        return float("-inf")

    # ── gap-9 DATA mechanism: scale fat-tail credit by per-cell posterior acc
    pnls = [o.net_pnl_usd for o in outs]
    weights: Optional[List[float]] = None
    if apply_direction_quality:
        dqw = direction_quality_weights(outs)
        if dqw:
            weights = [
                dqw.get(cell_of(o), 1.0) if o.direction_correct is not None else 1.0
                for o in outs
            ]
    ftb = fat_tail_bonus(pnls, weights=weights)
    pen = overtrading_penalty(outs, scratch_band_usd=scratch_band_usd,
                              ev_fee_multiple=ev_fee_multiple)

    reward = net_pnl + alpha * ftb - beta * pen
    log.debug("[REWARD] net=%.2f ftb=%.2f pen=%.2f acc=%s → R=%.2f",
              net_pnl, ftb, pen, f"{acc:.3f}" if acc is not None else "n/a", reward)
    return reward


def round_trip_fee_usd(notional_usd: float, fee_pct: float = ROUND_TRIP_FEE_PCT) -> float:
    """Round-trip taker fee in USD for a given notional."""
    return abs(notional_usd) * fee_pct
