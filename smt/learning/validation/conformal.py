"""Split-conformal prediction interval — distribution-free (MAPIE-style).

No normality assumption. Given calibration nonconformity scores (|residual|s
from held-out data), the (1−α) interval half-width is the finite-sample-corrected
empirical quantile of those scores. By exchangeability the interval covers the
truth with probability ≥ 1−α — so the dashboard shows an HONEST band
(expected_daily_pnl_usd ± half-width) instead of a false-precision point.

Consumes the CPCV Sharpe bag (smt.learning.validation.cpcv) or any residual
sample. The bandit / JUDGE confidence maps onto the calibrated coverage level.

Pure-Python; never raises.
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from smt.learning.validation._stats import mean, quantile

log = logging.getLogger("smt.learning.validation.conformal")


@dataclass
class ConformalInterval:
    point: float
    lower: float
    upper: float
    half_width: float
    confidence: float


def split_conformal_halfwidth(calib_scores: Sequence[float], alpha: float = 0.10) -> float:
    """(1−α) half-width = the ⌈(n+1)(1−α)⌉/n empirical quantile of |residual| scores.

    The (n+1) finite-sample correction is what gives conformal its guarantee.
    """
    scores = sorted(abs(float(s)) for s in calib_scores)
    n = len(scores)
    if n == 0:
        return 0.0
    level = math.ceil((n + 1) * (1.0 - alpha)) / n
    return quantile(scores, min(level, 1.0))


def conformal_interval(
    point: float,
    calib_residuals: Sequence[float],
    confidence: float = 0.90,
) -> ConformalInterval:
    """Symmetric split-conformal interval around ``point`` at ``confidence``."""
    alpha = 1.0 - confidence
    hw = split_conformal_halfwidth([abs(r) for r in calib_residuals], alpha)
    return ConformalInterval(point=point, lower=point - hw, upper=point + hw,
                             half_width=hw, confidence=confidence)


def conformal_pnl_interval(
    pnl_samples: Sequence[float],
    confidence: float = 0.90,
) -> ConformalInterval:
    """Honest expected-PnL band: point = mean(samples), residuals = sample − mean."""
    pts = [float(x) for x in pnl_samples]
    if not pts:
        return ConformalInterval(0.0, 0.0, 0.0, 0.0, confidence)
    mu = mean(pts)
    return conformal_interval(mu, [x - mu for x in pts], confidence)


def empirical_coverage(
    intervals: Sequence[Tuple[float, float]],
    truths: Sequence[float],
) -> float:
    """Fraction of truths that fall inside their (lower, upper) interval."""
    n = min(len(intervals), len(truths))
    if n == 0:
        return 0.0
    covered = sum(1 for i in range(n) if intervals[i][0] <= truths[i] <= intervals[i][1])
    return covered / n
