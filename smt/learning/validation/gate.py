"""Candidate validation gate — the one call the optimizer makes before shipping.

A TPE-optimized candidate must clear ALL THREE statistical gates before its
weights are written to v4/learned_params.json:

    DSR  > 0       (deflated for the trial count + non-normal returns)
    PBO  ≤ 0.20    (the IS winner is not an OOS coin-flip)
    FDR  ≤ 0.10    (a real discovery survives the per-cell multiple-comparison tax)

It is an OR-of-failures: any one breach rejects the candidate. CPCV + conformal
then wrap the survivor's Sharpe in an honest interval for the dashboard, and the
faithfulness flip (smt.learning.faithfulness) confirms FLOW / demotes SENTIMENT
before any weight actually ships.

Emits the Session-F activation log:
    [VALIDATION] candidate dsr=<f> pbo=<f> fdr=<f> verdict=<PASS|REJECT>
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from smt.learning.validation.dsr import deflated_sharpe
from smt.learning.validation.fdr import FDR_THRESHOLD, candidate_fdr
from smt.learning.validation.pbo import (
    PBO_THRESHOLD, Matrix, probability_of_backtest_overfitting,
)

log = logging.getLogger("smt.learning.validation.gate")


@dataclass
class ValidationReport:
    dsr: float
    pbo: float
    fdr: float
    verdict: str                       # "PASS" | "REJECT"
    psr: float = 0.0
    sharpe: float = 0.0
    reasons: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"


def validate_candidate(
    *,
    returns: Sequence[float],
    n_trials: int,
    returns_matrix: Matrix,
    per_cell_pvalues: Sequence[float],
    dsr_min: float = 0.0,
    pbo_max: float = PBO_THRESHOLD,
    fdr_max: float = FDR_THRESHOLD,
    n_splits: int = 8,
) -> ValidationReport:
    """Run DSR + PBO + FDR on a candidate; REJECT on the first breach.

    ``returns``           — the candidate's per-period net-PnL/return series (DSR).
    ``n_trials``          — configs searched this study (DSR deflation strength).
    ``returns_matrix``    — T×N (time × configs) performance matrix (PBO/CSCV).
    ``per_cell_pvalues``  — per-pair/per-cell edge p-values (BH-FDR family).
    """
    dsr_res = deflated_sharpe(returns, n_trials=n_trials)
    pbo_res = probability_of_backtest_overfitting(returns_matrix, n_splits=n_splits)
    fdr_q = candidate_fdr(per_cell_pvalues, q_level=fdr_max)

    reasons: List[str] = []
    if dsr_res.dsr < dsr_min:
        reasons.append(f"DSR {dsr_res.dsr:.3f} < {dsr_min:.2f}")
    if pbo_res.pbo > pbo_max:
        reasons.append(f"PBO {pbo_res.pbo:.3f} > {pbo_max:.2f}")
    if fdr_q > fdr_max:
        reasons.append(f"FDR {fdr_q:.3f} > {fdr_max:.2f}")

    verdict = "PASS" if not reasons else "REJECT"
    log.info("[VALIDATION] candidate dsr=%.3f pbo=%.3f fdr=%.3f verdict=%s",
             dsr_res.dsr, pbo_res.pbo, fdr_q, verdict)
    if reasons:
        log.info("[VALIDATION] reject reasons: %s", "; ".join(reasons))
    return ValidationReport(
        dsr=dsr_res.dsr, pbo=pbo_res.pbo, fdr=fdr_q, verdict=verdict,
        psr=dsr_res.psr, sharpe=dsr_res.sharpe, reasons=reasons,
    )
