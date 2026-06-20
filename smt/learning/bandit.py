"""Regime-conditional contextual bandit (Thompson Sampling, Beta-Binomial).

After JUDGE clears its confidence floor, the bandit answers "has THIS
(pair × direction × regime) cell actually paid before?" It samples a
probability-of-profit per arm and the daemon scales JUDGE-conf by it
(decision-time gate). Explores cold cells early, exploits proven ones.

Arms are keyed (pair × direction × regime_bucket): 8 pairs × {LONG, SHORT}
× {BULLISH, BEARISH, NEUTRAL} = up to 48 arms (the Session-D prompt's "24"
undercounts the full cross-product). Arms are created lazily and each is a
Beta(α, β) posterior over win-probability.

select_playbook() generalizes the same machinery to pick WHICH per-pair
playbook to fire (B.2 200d-MA vs B.7 institutional-flush vs B.11 single-pair
flush) via Thompson sampling over candidate labels.

Cold-start (AUTOPSY gap 8): the daemon RL state restarts from zero at the
V6.0.11 stop. An arm is not trusted to GATE (scale down JUDGE conf) until it
has `warmup_pulls` observations — before that the bandit is pass-through, so a
cold restart can't silently veto every entry. Seed arms from the 80d+ corpus
via seed_from_corpus() to shorten the warm-up.
"""

from __future__ import annotations
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from smt.personas.base import bare_pair, regime_bucket

log = logging.getLogger("smt.learning.bandit")

PRIOR_ALPHA = 1.0
PRIOR_BETA = 1.0
WARMUP_PULLS = 20          # arm not trusted to gate until this many observations

ArmKey = Tuple[str, str, str]   # (pair, label, regime_bucket)


@dataclass
class BetaArm:
    alpha: float = PRIOR_ALPHA
    beta: float = PRIOR_BETA
    pulls: int = 0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def sample(self, rng: random.Random) -> float:
        return rng.betavariate(self.alpha, self.beta)

    def update(self, won: bool, weight: float = 1.0) -> None:
        if won:
            self.alpha += weight
        else:
            self.beta += weight
        self.pulls += 1


class ContextualBandit:
    """Thompson-Sampling Beta-Binomial bandit over (pair × label × regime)."""

    def __init__(
        self,
        seed: Optional[int] = None,
        prior_alpha: float = PRIOR_ALPHA,
        prior_beta: float = PRIOR_BETA,
        warmup_pulls: int = WARMUP_PULLS,
    ):
        self.rng = random.Random(seed)
        self.prior = (prior_alpha, prior_beta)
        self.warmup_pulls = warmup_pulls
        self.arms: Dict[ArmKey, BetaArm] = {}

    # ── arm access ────────────────────────────────────────────────────────────

    def key(self, pair: str, label: str, regime: str) -> ArmKey:
        return (bare_pair(pair), str(label).upper(), regime_bucket(regime))

    def arm(self, pair: str, label: str, regime: str) -> BetaArm:
        k = self.key(pair, label, regime)
        if k not in self.arms:
            self.arms[k] = BetaArm(*self.prior)
        return self.arms[k]

    # ── update + query ────────────────────────────────────────────────────────

    def update(self, pair: str, label: str, regime: str, won: bool, weight: float = 1.0) -> None:
        """Posterior update on close: realized PnL > 0 → won=True."""
        self.arm(pair, label, regime).update(bool(won), weight)

    def prob_of_profit(self, pair: str, label: str, regime: str) -> float:
        """Posterior MEAN win-probability for the arm (deterministic estimate)."""
        return self.arm(pair, label, regime).mean

    def posterior_mean(self, pair: str, label: str, regime: str) -> float:
        return self.prob_of_profit(pair, label, regime)

    def thompson_sample(self, pair: str, label: str, regime: str) -> float:
        """Draw a win-probability sample (exploration ∝ posterior uncertainty)."""
        return self.arm(pair, label, regime).sample(self.rng)

    def is_warm(self, pair: str, label: str, regime: str) -> bool:
        return self.arm(pair, label, regime).pulls >= self.warmup_pulls

    # ── decision-time gate (JUDGE-conf scaling) ───────────────────────────────

    def scaled_confidence(self, judge_conf: float, pair: str, direction: str, regime: str) -> float:
        """Scale JUDGE confidence by the arm's probability-of-profit.

        Cold-start safe: a cell with < warmup_pulls observations is pass-through
        (returns judge_conf unchanged) so a fresh restart can't veto everything.
        """
        if not self.is_warm(pair, direction, regime):
            return float(judge_conf)
        return float(judge_conf) * self.prob_of_profit(pair, direction, regime)

    # ── playbook selection (Thompson over candidate labels) ───────────────────

    def select_playbook(
        self, pair: str, regime: str, candidates: Sequence[str] = ("LONG", "SHORT")
    ) -> str:
        """Pick the candidate label with the highest Thompson sample."""
        best_label, best_sample = candidates[0], -1.0
        for label in candidates:
            s = self.thompson_sample(pair, label, regime)
            if s > best_sample:
                best_sample, best_label = s, label
        log.debug("[BANDIT] select_playbook %s/%s → %s", bare_pair(pair),
                  regime_bucket(regime), best_label)
        return best_label

    # ── cold-start corpus seeding (gap 8) ─────────────────────────────────────

    def seed_from_corpus(self, records: Iterable[Dict[str, Any]], weight: float = 1.0) -> int:
        """Seed Beta posteriors from historical trades (trades.json schema).

        Each executed trade with pair / side / regime / win updates its arm.
        Returns the number of records consumed. Tolerant of missing fields.
        """
        n = 0
        for r in records:
            if not isinstance(r, dict):
                continue
            if r.get("skipped"):
                continue
            pair = r.get("pair") or r.get("symbol")
            side = r.get("side") or r.get("direction")
            won = r.get("win")
            regime = r.get("regime", "NEUTRAL")
            if pair is None or side is None or won is None:
                continue
            self.update(pair, side, regime, bool(won), weight=weight)
            n += 1
        log.info("[BANDIT] seeded %d arms-updates from corpus (%d arms live)", n, len(self.arms))
        return n

    # ── optional persistence (daemon, Session E) ──────────────────────────────

    def save(self, path: str = "v4/bandit_state.json") -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        payload = {f"{k[0]}|{k[1]}|{k[2]}": [a.alpha, a.beta, a.pulls] for k, a in self.arms.items()}
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def load(self, path: str = "v4/bandit_state.json") -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                payload = json.load(f)
            for ks, (a, b, p) in payload.items():
                pair, label, reg = ks.split("|")
                arm = self.arms.setdefault((pair, label, reg), BetaArm(*self.prior))
                arm.alpha, arm.beta, arm.pulls = float(a), float(b), int(p)
            return True
        except Exception as exc:
            log.warning("[BANDIT] could not load state from %s: %s", path, exc)
            return False
