"""Kernel Density Estimate over per-pair return distributions.

Feeds the conformal prior + the fat-tail bonus term in the reward
function (smt.learning.reward).

Session D pulls forward a MINIMAL, dependency-free Gaussian KDE (Silverman
bandwidth) so reward.FatTailBonus can be KDE-weighted without numpy/scipy.
Session F extends this with a leave-one-out cross-validated bandwidth
(`cv_bandwidth`) for the fat-tail / conformal density; the conformal interval
itself lives in smt.learning.validation.conformal. `kde_fit` always returns a
usable density estimator.
"""

from __future__ import annotations
import math
from typing import Iterable, List, Optional

SQRT_2PI = math.sqrt(2.0 * math.pi)
SQRT_2 = math.sqrt(2.0)


def _empirical_quantile(sorted_data: List[float], p: float) -> float:
    """Linear-interpolated empirical quantile. `sorted_data` must be sorted."""
    n = len(sorted_data)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_data[0]
    idx = p * (n - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_data[lo]
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (idx - lo)


def silverman_bandwidth(data: List[float]) -> float:
    """Silverman's rule-of-thumb bandwidth: 0.9 · min(σ, IQR/1.349) · n^(-1/5)."""
    n = len(data)
    if n < 2:
        return 1.0
    mean = sum(data) / n
    var = sum((x - mean) ** 2 for x in data) / (n - 1)
    sd = math.sqrt(var) if var > 0 else 0.0
    s = sorted(data)
    iqr = _empirical_quantile(s, 0.75) - _empirical_quantile(s, 0.25)
    spread = min(sd, iqr / 1.349) if iqr > 0 else sd
    if spread <= 0:
        spread = sd if sd > 0 else 1.0
    bw = 0.9 * spread * (n ** (-0.2))
    return bw if bw > 0 else 1.0


class GaussianKDE:
    """1-D Gaussian kernel density estimator (pure Python)."""

    def __init__(self, data: Iterable[float], bandwidth: Optional[float] = None):
        self.data: List[float] = [float(x) for x in data]
        self.n = len(self.data)
        self.bandwidth = float(bandwidth) if bandwidth else silverman_bandwidth(self.data)
        if self.bandwidth <= 0:
            self.bandwidth = 1.0

    def pdf(self, x: float) -> float:
        if self.n == 0:
            return 0.0
        norm = 1.0 / (self.n * self.bandwidth * SQRT_2PI)
        acc = 0.0
        for xi in self.data:
            u = (x - xi) / self.bandwidth
            acc += math.exp(-0.5 * u * u)
        return norm * acc

    def cdf(self, x: float) -> float:
        """Smoothed CDF = mean of per-kernel normal CDFs."""
        if self.n == 0:
            return 0.0
        acc = 0.0
        denom = self.bandwidth * SQRT_2
        for xi in self.data:
            acc += 0.5 * (1.0 + math.erf((x - xi) / denom))
        return acc / self.n

    def quantile(self, p: float) -> float:
        """Inverse smoothed-CDF via bisection. Falls back to empirical."""
        if self.n == 0:
            return 0.0
        if self.n == 1:
            return self.data[0]
        p = min(max(p, 0.0), 1.0)
        lo = min(self.data) - 3.0 * self.bandwidth
        hi = max(self.data) + 3.0 * self.bandwidth
        for _ in range(64):
            mid = (lo + hi) / 2.0
            if self.cdf(mid) < p:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    def loo_log_likelihood(self) -> float:
        """Leave-one-out log-likelihood — the CV objective for bandwidth choice.

        Each point's density is estimated from the OTHER n−1 points, so an
        over-narrow bandwidth (which would spike on each held-out point) is
        penalized rather than rewarded.
        """
        if self.n < 2:
            return float("-inf")
        norm = 1.0 / ((self.n - 1) * self.bandwidth * SQRT_2PI)
        total = 0.0
        for i, xi in enumerate(self.data):
            acc = 0.0
            for j, xj in enumerate(self.data):
                if i == j:
                    continue
                u = (xi - xj) / self.bandwidth
                acc += math.exp(-0.5 * u * u)
            dens = norm * acc
            total += math.log(dens) if dens > 0.0 else -50.0
        return total


def cv_bandwidth(data: Iterable[float], candidates: Optional[List[float]] = None) -> float:
    """Leave-one-out CV bandwidth: the grid point maximizing LOO log-likelihood.

    Defaults to a multiplicative grid around Silverman's rule. Falls back to
    Silverman for n < 3. Pure-Python, O(n²·|grid|) — fine for the small return
    samples the reward/conformal layers feed it.
    """
    pts = [float(x) for x in data]
    if len(pts) < 3:
        return silverman_bandwidth(pts)
    base = silverman_bandwidth(pts)
    if candidates is None:
        candidates = [base * m for m in (0.25, 0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0)]
    best_bw, best_ll = base, float("-inf")
    for bw in candidates:
        if bw <= 0:
            continue
        ll = GaussianKDE(pts, bw).loo_log_likelihood()
        if ll > best_ll:
            best_ll, best_bw = ll, bw
    return best_bw


def kde_fit(data: Iterable[float], bandwidth: Optional[float] = None) -> GaussianKDE:
    """Fit a Gaussian KDE to `data`. Returns a GaussianKDE with .pdf/.cdf/.quantile."""
    return GaussianKDE(data, bandwidth)


def kde_quantile(data: Iterable[float], p: float, bandwidth: Optional[float] = None) -> float:
    """Convenience: KDE-smoothed quantile of `data` at probability `p`."""
    pts = [float(x) for x in data]
    if not pts:
        return 0.0
    if len(pts) < 3:
        return _empirical_quantile(sorted(pts), p)
    return GaussianKDE(pts, bandwidth).quantile(p)
