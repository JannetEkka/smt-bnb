"""Pure-Python statistics shared by the validation gates.

No numpy / scipy: the learning core stays dependency-free so a bare container
runs (same contract as smt.learning.validation.kde and the optimizer builtin
TPE). Everything here is small, exact, and deterministic.
"""

from __future__ import annotations
import math
from typing import List, Sequence

SQRT2 = math.sqrt(2.0)
# Euler–Mascheroni constant — used by the DSR expected-maximum-Sharpe term.
EULER_GAMMA = 0.5772156649015329


def mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def variance(xs: Sequence[float], ddof: int = 1) -> float:
    xs = list(xs)
    n = len(xs)
    if n - ddof <= 0:
        return 0.0
    mu = mean(xs)
    return sum((x - mu) ** 2 for x in xs) / (n - ddof)


def stdev(xs: Sequence[float], ddof: int = 1) -> float:
    v = variance(xs, ddof)
    return math.sqrt(v) if v > 0 else 0.0


def sharpe_ratio(returns: Sequence[float], rf: float = 0.0) -> float:
    """Non-annualized Sharpe of a per-period return series (0.0 if no dispersion)."""
    xs = [float(r) - rf for r in returns]
    sd = stdev(xs, ddof=1)
    if sd <= 0.0:
        return 0.0
    return mean(xs) / sd


def skewness(xs: Sequence[float]) -> float:
    """Sample skewness γ3 = m3 / m2**1.5 (population moments). 0 if degenerate."""
    xs = list(xs)
    n = len(xs)
    if n < 3:
        return 0.0
    mu = mean(xs)
    m2 = sum((x - mu) ** 2 for x in xs) / n
    if m2 <= 0:
        return 0.0
    m3 = sum((x - mu) ** 3 for x in xs) / n
    return m3 / (m2 ** 1.5)


def kurtosis(xs: Sequence[float]) -> float:
    """Raw (NON-excess) kurtosis γ4 = m4 / m2**2 — a normal distribution gives 3.0.

    The DSR / PSR variance term uses (γ4 − 1)/4, so it expects raw kurtosis.
    """
    xs = list(xs)
    n = len(xs)
    if n < 4:
        return 3.0
    mu = mean(xs)
    m2 = sum((x - mu) ** 2 for x in xs) / n
    if m2 <= 0:
        return 3.0
    m4 = sum((x - mu) ** 4 for x in xs) / n
    return m4 / (m2 ** 2)


def norm_cdf(x: float) -> float:
    """Standard-normal CDF Φ(x) via the error function."""
    return 0.5 * (1.0 + math.erf(x / SQRT2))


def norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF Φ⁻¹(p) — Acklam's rational approximation.

    Accurate to ~1e-9 on (0, 1). Saturates rather than raising at the edges.
    """
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def quantile(sorted_xs: List[float], p: float) -> float:
    """Linear-interpolated empirical quantile of an already-sorted list."""
    n = len(sorted_xs)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_xs[0]
    p = min(max(p, 0.0), 1.0)
    idx = p * (n - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_xs[lo]
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (idx - lo)


def pearson_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation; 0.0 when either series has no variance."""
    xs, ys = list(xs), list(ys)
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    mx, my = mean(xs[:n]), mean(ys[:n])
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    sxx = sum((xs[i] - mx) ** 2 for i in range(n))
    syy = sum((ys[i] - my) ** 2 for i in range(n))
    if sxx <= 0 or syy <= 0:
        return 0.0
    return sxy / math.sqrt(sxx * syy)
