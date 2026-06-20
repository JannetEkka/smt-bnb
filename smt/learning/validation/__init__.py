"""smt.learning.validation — CPCV / DSR / PBO / FDR / conformal / KDE + gate.

Every TPE-optimized candidate must pass DSR + PBO + FDR (``gate.validate_candidate``)
before going to CPCV (Bagged Combinatorial Purged CV) for held-out evaluation;
the CPCV Sharpe gets wrapped in a conformal prediction interval.
"""

from smt.learning.validation.dsr import deflated_sharpe, DSRResult
from smt.learning.validation.pbo import (
    probability_of_backtest_overfitting, per_lane_pbo, lanes_passing,
    PBOResult, PBO_THRESHOLD,
)
from smt.learning.validation.fdr import bh_fdr, candidate_fdr, BHResult, FDR_THRESHOLD
from smt.learning.validation.cpcv import (
    bagged_cpcv_sharpe, combinatorial_purged_splits, CPCVResult,
)
from smt.learning.validation.conformal import (
    conformal_interval, conformal_pnl_interval, split_conformal_halfwidth,
    empirical_coverage, ConformalInterval,
)
from smt.learning.validation.kde import kde_fit, kde_quantile, cv_bandwidth, GaussianKDE
from smt.learning.validation.gate import validate_candidate, ValidationReport

__all__ = [
    "deflated_sharpe", "DSRResult",
    "probability_of_backtest_overfitting", "per_lane_pbo", "lanes_passing",
    "PBOResult", "PBO_THRESHOLD",
    "bh_fdr", "candidate_fdr", "BHResult", "FDR_THRESHOLD",
    "bagged_cpcv_sharpe", "combinatorial_purged_splits", "CPCVResult",
    "conformal_interval", "conformal_pnl_interval", "split_conformal_halfwidth",
    "empirical_coverage", "ConformalInterval",
    "kde_fit", "kde_quantile", "cv_bandwidth", "GaussianKDE",
    "validate_candidate", "ValidationReport",
]
