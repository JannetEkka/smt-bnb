"""Deflated Sharpe Ratio — Bailey & López de Prado, JFDS 2014.

Discounts the observed Sharpe for (a) the number of trials searched and
(b) the skew / kurtosis of the returns. The intuition: if you try N random
strategies, the BEST in-sample Sharpe is large even with zero true edge, so
the bar a real strategy must clear rises with N.

We report the **signed deflated statistic** ``dsr`` (a z-score): the observed
Sharpe minus the expected-maximum Sharpe under the null of N zero-skill trials,
standardized by the (non-normal) variance of the Sharpe estimator. The operator
rule (PLAN.md Session F): **reject the candidate if ``dsr < 0``** — i.e. the
strategy's Sharpe does not even beat the best you'd expect from N coin-flips.
``psr`` is the matching probability Φ(dsr) (the Probabilistic Sharpe Ratio
against the deflated benchmark) for the dashboard's honest-interval surface.

Pure-Python (see _stats); never raises on degenerate input.
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass
from typing import Optional, Sequence

from smt.learning.validation._stats import (
    EULER_GAMMA, kurtosis, norm_cdf, norm_ppf, sharpe_ratio, skewness,
)

log = logging.getLogger("smt.learning.validation.dsr")


@dataclass
class DSRResult:
    dsr: float            # signed deflated z-stat — reject candidate if < 0
    psr: float            # Φ(dsr): probabilistic Sharpe vs the deflated benchmark
    sharpe: float         # observed (non-annualized) Sharpe
    benchmark_sr: float   # SR*₀ — expected max Sharpe under N null trials
    n_obs: int
    n_trials: int
    skew: float
    kurtosis: float

    @property
    def passed(self) -> bool:
        return self.dsr > 0.0


def sharpe_estimator_variance(sr: float, n_obs: int, skew: float, kurt: float) -> float:
    """Var[SR̂] under non-normal returns (Lo 2002 / Mertens). raw kurtosis (normal=3)."""
    if n_obs < 2:
        return 0.0
    adj = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    return max(adj, 1e-12) / (n_obs - 1)


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """E[max Sharpe] over ``n_trials`` independent zero-skill trials (Bailey-LdP).

    SR*₀ = √Var[SR] · [ (1−γ)·Z(1−1/N) + γ·Z(1−1/(N·e)) ],  γ = Euler-Mascheroni.
    """
    if n_trials <= 1 or sharpe_variance <= 0:
        return 0.0
    sd = math.sqrt(sharpe_variance)
    z1 = norm_ppf(1.0 - 1.0 / n_trials)
    z2 = norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    return sd * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def prob_sharpe_ratio(sr: float, sr_benchmark: float, n_obs: int,
                      skew: float, kurt: float) -> float:
    """PSR(sr*) = Φ( (sr − sr*)·√(T−1) / √(1 − γ3·sr + ((γ4−1)/4)·sr²) )."""
    if n_obs < 2:
        return 0.5
    denom = math.sqrt(max(1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr, 1e-12))
    z = (sr - sr_benchmark) * math.sqrt(n_obs - 1) / denom
    return norm_cdf(z)


def deflated_sharpe(
    returns: Sequence[float],
    n_trials: int = 1,
    *,
    sharpe_variance: Optional[float] = None,
    benchmark_sr: Optional[float] = None,
) -> DSRResult:
    """Compute the Deflated Sharpe Ratio for a per-period return series.

    ``n_trials``         — independent strategy configs tried (deflation strength).
    ``sharpe_variance``  — cross-trial Var[SR] if you have all trial Sharpes; else
                           estimated from this strategy's SR-estimator variance.
    ``benchmark_sr``     — override SR*₀ (else the expected-max under the null).
    """
    rets = [float(r) for r in returns]
    n = len(rets)
    sr = sharpe_ratio(rets)
    sk = skewness(rets)
    ku = kurtosis(rets)
    var_sr = (sharpe_variance if sharpe_variance is not None
              else sharpe_estimator_variance(sr, n, sk, ku))
    sr_star = (benchmark_sr if benchmark_sr is not None
               else expected_max_sharpe(n_trials, var_sr))
    psr = prob_sharpe_ratio(sr, sr_star, n, sk, ku)
    # Signed deflated z-stat: same standardization as PSR, sign carries the verdict.
    if n < 2:
        z = 0.0
    else:
        denom = math.sqrt(max(1.0 - sk * sr + ((ku - 1.0) / 4.0) * sr * sr, 1e-12))
        z = (sr - sr_star) * math.sqrt(n - 1) / denom
    res = DSRResult(dsr=z, psr=psr, sharpe=sr, benchmark_sr=sr_star,
                    n_obs=n, n_trials=int(n_trials), skew=sk, kurtosis=ku)
    log.debug("[DSR] sr=%.4f sr*=%.4f n=%d trials=%d → dsr=%.4f psr=%.3f",
              sr, sr_star, n, n_trials, z, psr)
    return res
