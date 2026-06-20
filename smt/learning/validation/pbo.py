"""Probability of Backtest Overfitting — Bailey/Borwein/López de Prado/Zhu, JCF 2017.

CSCV (Combinatorially Symmetric Cross-Validation): the probability that the
configuration which looks best IN-SAMPLE is a below-median performer
OUT-OF-SAMPLE. High PBO ⇒ the "winner" is an artefact of the search.

Algorithm:
  1. Take a T×N performance matrix M (rows = time periods, cols = N candidate
     configs), split the rows into S contiguous blocks.
  2. For every way to choose S/2 blocks as in-sample (the rest out-of-sample):
       n* = argmax of the IS Sharpe;  ω = OOS rank of n* in (0,1);
       logit λ = ln(ω / (1−ω)).  λ < 0 ⇔ the IS winner is below the OOS median.
  3. PBO = fraction of splits with λ ≤ 0.
Filter: candidate passes only if **PBO ≤ 0.20**.

Per-lane attribution (Session F backlog): run CSCV per lane so a slow-lane
drawdown never rejects a healthy fast-lane scalp book — the lanes overfit (or
don't) independently.

Also the production live-PBO stopping rule (rolling 30d window; halt new entries
when live PBO > 0.30) — that scheduler lands in Session G.

Pure-Python; never raises (degenerate input ⇒ PBO 0.0).
"""

from __future__ import annotations
import itertools
import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

from smt.learning.validation._stats import sharpe_ratio

log = logging.getLogger("smt.learning.validation.pbo")

Matrix = Sequence[Sequence[float]]
PBO_THRESHOLD = 0.20


@dataclass
class PBOResult:
    pbo: float
    n_splits: int
    n_configs: int
    lambdas: List[float] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.pbo <= PBO_THRESHOLD


def _block_indices(n_rows: int, n_blocks: int) -> List[List[int]]:
    """Partition row indices 0..n_rows-1 into n_blocks near-equal contiguous blocks."""
    size = n_rows / n_blocks
    blocks: List[List[int]] = []
    for b in range(n_blocks):
        lo = int(round(b * size))
        hi = int(round((b + 1) * size))
        blocks.append(list(range(lo, hi)))
    return blocks


def _col_perf(matrix: Matrix, rows: List[int], col: int,
              metric: Callable[[Sequence[float]], float]) -> float:
    return metric([matrix[r][col] for r in rows])


def probability_of_backtest_overfitting(
    returns_matrix: Matrix,
    n_splits: int = 8,
    metric: Callable[[Sequence[float]], float] = sharpe_ratio,
) -> PBOResult:
    """CSCV PBO over a T×N (time × configs) performance matrix.

    ``n_splits`` (S) must be even; C(S, S/2) symmetric IS/OOS partitions are
    evaluated. Needs ≥ 2 configs and ≥ S rows or it degrades to PBO 0.0.
    """
    rows = list(returns_matrix)
    T = len(rows)
    N = len(rows[0]) if T else 0
    if N < 2 or T < n_splits or n_splits < 2:
        return PBOResult(pbo=0.0, n_splits=n_splits, n_configs=N, lambdas=[])
    if n_splits % 2 == 1:
        n_splits -= 1

    blocks = _block_indices(T, n_splits)
    all_blocks = set(range(n_splits))
    lambdas: List[float] = []
    eps = 1.0 / (N + 1)

    for is_combo in itertools.combinations(range(n_splits), n_splits // 2):
        is_rows: List[int] = []
        for b in is_combo:
            is_rows.extend(blocks[b])
        oos_rows: List[int] = []
        for b in (all_blocks - set(is_combo)):
            oos_rows.extend(blocks[b])
        if not is_rows or not oos_rows:
            continue

        is_perf = [_col_perf(rows, is_rows, c, metric) for c in range(N)]
        oos_perf = [_col_perf(rows, oos_rows, c, metric) for c in range(N)]
        n_star = max(range(N), key=lambda c: is_perf[c])

        # Relative OOS rank of the IS winner in (0,1): 1 ⇒ best OOS, ~0 ⇒ worst.
        rank = sum(1 for c in range(N) if oos_perf[c] <= oos_perf[n_star])
        omega = rank / (N + 1)
        omega = min(max(omega, eps), 1.0 - eps)
        lambdas.append(math.log(omega / (1.0 - omega)))

    if not lambdas:
        return PBOResult(pbo=0.0, n_splits=n_splits, n_configs=N, lambdas=[])
    pbo = sum(1 for lam in lambdas if lam <= 0.0) / len(lambdas)
    log.debug("[PBO] splits=%d configs=%d combos=%d → pbo=%.3f",
              n_splits, N, len(lambdas), pbo)
    return PBOResult(pbo=pbo, n_splits=n_splits, n_configs=N, lambdas=lambdas)


def per_lane_pbo(
    matrices_by_lane: Dict[str, Matrix],
    n_splits: int = 8,
    metric: Callable[[Sequence[float]], float] = sharpe_ratio,
) -> Dict[str, PBOResult]:
    """Independent CSCV PBO per lane — a slow-lane drawdown cannot reject the fast lane."""
    return {
        lane: probability_of_backtest_overfitting(mat, n_splits=n_splits, metric=metric)
        for lane, mat in matrices_by_lane.items()
    }


def lanes_passing(per_lane: Dict[str, PBOResult], threshold: float = PBO_THRESHOLD) -> Dict[str, bool]:
    """{lane: PBO ≤ threshold}. The fast lane stays True even when slow fails."""
    return {lane: res.pbo <= threshold for lane, res in per_lane.items()}
