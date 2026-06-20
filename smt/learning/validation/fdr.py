"""False Discovery Rate — Benjamini-Hochberg, JRSSB 1995.

Across 8 pairs × many param cells we run many hypothesis tests ("does this
cell have a real edge?"). Naïve per-test significance inflates false positives;
BH caps the expected proportion of false discoveries among the rejections.

Two surfaces:
- ``bh_fdr(pvals, q_level)`` — the step-up procedure: BH-adjusted q-values, the
  rejection mask, and the discovery count for a family of p-values.
- ``candidate_fdr(per_cell_pvalues, q_level)`` — a candidate's verdict figure:
  the SMALLEST BH-adjusted q-value across its per-cell studies. If even the best
  cell's q-value exceeds ``q_level`` the candidate has NO real discovery anywhere
  → **reject if fdr > 0.10**.

Pure-Python; never raises (empty input ⇒ q-value 1.0).
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import List, Sequence

log = logging.getLogger("smt.learning.validation.fdr")

FDR_THRESHOLD = 0.10


@dataclass
class BHResult:
    adjusted: List[float]        # BH q-values, in the INPUT order
    rejected: List[bool]         # q ≤ q_level, in the INPUT order
    n_discoveries: int
    q_level: float
    min_q: float                 # smallest q-value = best cell (candidate figure)
    threshold_p: float = 0.0     # largest raw p that is still a discovery
    adjusted_sorted: List[float] = field(default_factory=list)


def bh_adjusted_pvalues(pvals: Sequence[float]) -> List[float]:
    """Benjamini-Hochberg step-up adjusted p-values (q-values), in input order.

    q_(i) = min_{k ≥ i} ( m/k · p_(k) ), then cumulative-min from the largest
    rank down and clamped to ≤ 1.0. Monotone in the ranks, as BH requires.
    """
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])      # ascending p
    adj_sorted = [0.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):                            # m .. 1
        i = order[rank - 1]
        q = (m / rank) * float(pvals[i])
        prev = min(prev, q)
        adj_sorted[rank - 1] = min(prev, 1.0)
    out = [0.0] * m
    for rank, i in enumerate(order):
        out[i] = adj_sorted[rank]
    return out


def bh_fdr(pvals: Sequence[float], q_level: float = FDR_THRESHOLD) -> BHResult:
    """Run BH at ``q_level`` over a family of p-values."""
    pvals = [float(p) for p in pvals]
    if not pvals:
        return BHResult(adjusted=[], rejected=[], n_discoveries=0,
                        q_level=q_level, min_q=1.0, threshold_p=0.0, adjusted_sorted=[])
    adjusted = bh_adjusted_pvalues(pvals)
    rejected = [q <= q_level for q in adjusted]
    threshold_p = max((pvals[i] for i, r in enumerate(rejected) if r), default=0.0)
    return BHResult(
        adjusted=adjusted,
        rejected=rejected,
        n_discoveries=sum(rejected),
        q_level=q_level,
        min_q=min(adjusted),
        threshold_p=threshold_p,
        adjusted_sorted=sorted(adjusted),
    )


def candidate_fdr(per_cell_pvalues: Sequence[float], q_level: float = FDR_THRESHOLD) -> float:
    """Smallest BH-adjusted q-value across a candidate's per-cell studies.

    Reject the candidate when this exceeds ``q_level`` — no cell is a real
    discovery once the family-wide multiple-comparison inflation is paid for.
    """
    res = bh_fdr(per_cell_pvalues, q_level=q_level)
    return res.min_q
