"""Bagged Combinatorial Purged Cross-Validation — López de Prado, AFML ch. 7.

Standard k-fold leaks future information into the training set via overlapping
labels (a trade's +Nh outcome straddles the train/test boundary). CPCV:
  1. split observations into N contiguous groups, choose k as the test set;
  2. PURGE training observations whose label window overlaps any test obs;
  3. EMBARGO a fraction of observations immediately AFTER each test block;
  4. bag the OOS Sharpe across all C(N, k) paths → a distribution, not a point.

The Sharpe distribution is what smt.learning.validation.conformal wraps in a
calibrated prediction interval. Here we evaluate OOS Sharpe directly on each
purged test split (the "model" is the candidate parameterization being graded).

Pure-Python; never raises.
"""

from __future__ import annotations
import itertools
import logging
import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from smt.learning.validation._stats import mean, sharpe_ratio, stdev

log = logging.getLogger("smt.learning.validation.cpcv")


@dataclass
class CPCVResult:
    sharpes: List[float]         # one OOS Sharpe per combinatorial path
    mean_sharpe: float
    std_sharpe: float
    n_paths: int
    test_sizes: List[int] = field(default_factory=list)


def _groups(n_obs: int, n_groups: int) -> List[List[int]]:
    size = n_obs / n_groups
    out: List[List[int]] = []
    for g in range(n_groups):
        lo = int(round(g * size))
        hi = int(round((g + 1) * size))
        out.append(list(range(lo, hi)))
    return out


def combinatorial_purged_splits(
    n_obs: int,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo_frac: float = 0.01,
    label_horizon: int = 1,
) -> List[Tuple[List[int], List[int]]]:
    """Yield (train_idx, test_idx) for every C(n_groups, n_test_groups) split.

    ``label_horizon`` — bars a label spans; training obs within this many bars of
    any test obs are PURGED. ``embargo_frac`` — fraction of n_obs embargoed right
    after each test block (kills serial-correlation leakage).
    """
    groups = _groups(n_obs, n_groups)
    embargo = int(math.ceil(embargo_frac * n_obs))
    splits: List[Tuple[List[int], List[int]]] = []

    for test_combo in itertools.combinations(range(n_groups), n_test_groups):
        test_idx: List[int] = []
        for g in test_combo:
            test_idx.extend(groups[g])
        test_set = set(test_idx)

        # Forbidden = test ∪ purge-neighborhood ∪ embargo-after-each-test-obs.
        forbidden = set(test_idx)
        for t in test_idx:
            for d in range(1, label_horizon + 1):
                forbidden.add(t - d)
                forbidden.add(t + d)
            for e in range(1, embargo + 1):
                forbidden.add(t + e)          # embargo only AFTER the test block

        train_idx = [i for i in range(n_obs) if i not in forbidden]
        splits.append((train_idx, sorted(test_set)))
    return splits


def bagged_cpcv_sharpe(
    returns: Sequence[float],
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo_frac: float = 0.01,
    label_horizon: int = 1,
) -> CPCVResult:
    """OOS Sharpe distribution over all combinatorial purged paths of ``returns``."""
    rets = [float(r) for r in returns]
    n = len(rets)
    if n < n_groups:
        return CPCVResult(sharpes=[], mean_sharpe=0.0, std_sharpe=0.0, n_paths=0)
    splits = combinatorial_purged_splits(
        n, n_groups, n_test_groups, embargo_frac, label_horizon)
    sharpes: List[float] = []
    sizes: List[int] = []
    for _train, test in splits:
        sharpes.append(sharpe_ratio([rets[i] for i in test]))
        sizes.append(len(test))
    return CPCVResult(
        sharpes=sharpes,
        mean_sharpe=mean(sharpes),
        std_sharpe=stdev(sharpes) if len(sharpes) > 1 else 0.0,
        n_paths=len(sharpes),
        test_sizes=sizes,
    )
