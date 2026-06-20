"""Synthetic ground-truth simulator (known-edge regime-switching).

Before backtesting on real data, synthesize a market with KNOWN edge
magnitude, KNOWN regime durations, KNOWN cascade behavior. If the
optimizer doesn't recover the known edge on synthetic data, it's
broken — don't trust it on real data.

This is Session G but pulled forward into Session D as scaffolding to
validate the optimizer + reward + bandit before any live data touch.

Planted edges (AUTOPSY Session-D addendum):
- FLOW is the primary signal: trend-aligned and ~`flow_acc_trend` accurate
  in TRENDING_UP / TRENDING_DOWN, but it FOLLOWS the downtrend into crash
  bounces (so it is wrong exactly where SENTIMENT-contra is right).
- SENTIMENT contra-LONG in CRASH cells (matches the V3.2.224 dataset):
  ~`sent_acc_crash` accurate, and uninformative elsewhere.
- TECHNICAL is a mild trend confirmer; WHALE/ONCHAIN/REGIME are noise.
Because FLOW is good in trends but bad in crash bounces, the reward-optimal
JUDGE weighting is FLOW-heavy with a SENTIMENT minority — an INTERIOR
optimum. `planted_flow_weight` is that designed optimum; the optimizer must
recover it (±15%) on a 50-trial study.

Common-random-numbers: the market (regimes, truths, votes, win-multipliers)
is regenerated deterministically from (seed, n) on every call, so two weight
vectors are scored on the SAME market — the objective is smooth, recovery
is reliable.
"""

from __future__ import annotations
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from smt.learning.reward import TradeOutcome
from smt.personas.base import JUDGE_SEED_PRIORS

log = logging.getLogger("smt.learning.synthetic")

REGIMES = ["TRENDING_UP", "TRENDING_DOWN", "RANGING"]
PERSONAS = list(JUDGE_SEED_PRIORS.keys())  # flow, technical, whale, onchain, sentiment, regime

# Regime HMM (rows: from, cols: to) — order = REGIMES. Symmetric → uniform
# stationary (1/3 each) with moderate persistence: long enough episodes to be
# a real regime signal, short enough that 1000 steps mix to ±10% of stationary.
TRANSITION = [
    [0.76, 0.12, 0.12],   # from TRENDING_UP
    [0.12, 0.76, 0.12],   # from TRENDING_DOWN
    [0.12, 0.12, 0.76],   # from RANGING
]


@dataclass
class SimConfig:
    # Continuous persona signals in [-1, 1]: signal = clamp(truth·edge + noise).
    # A positive edge points (noisily) at truth; a NEGATIVE edge points away.
    flow_edge_trend: float = 0.95    # FLOW strongly trend-aligned (the primary edge)
    flow_edge_crash: float = -0.50   # FLOW keeps shorting the bounce → wrong in crash
    sent_edge_crash: float = 0.90    # SENTIMENT contra-LONG is right in crash bounces
    tech_edge_trend: float = 0.35    # TECHNICAL mild trend confirmer
    noise_sd: float = 0.55           # Gaussian noise sd on every signal
    p_truth_trend: float = 0.78      # P(truth == trend direction) in a trend
    crash_frac: float = 0.30         # fraction of TRENDING_DOWN steps that are crash bounces
    tp_usd: float = 60.0
    sl_usd: float = 55.0
    fee_usd: float = 4.0
    fat_tail_prob: float = 0.06      # P(a win is a fat-tail winner)
    fat_tail_mult: float = 5.0       # magnitude multiplier for fat-tail winners


@dataclass
class _Step:
    regime: str
    crash: bool
    truth: int                       # +1 LONG, -1 SHORT
    votes: Dict[str, float]          # persona -> signal in [-1, 1]
    win_mult: float
    fng: int


# Designed interior optimum of the JUDGE flow-vs-sentiment weighting under the
# default config — the two PLANTED edges (FLOW trend + SENTIMENT crash-bounce).
# Calibrated by maximizing reward over the flow/sentiment ridge, averaged across
# markets: argmax flow ≈ 0.475. FLOW is the dominant weight; it cedes a minority
# share to SENTIMENT for the crash cells, so the optimum sits below 1.0. The
# optimizer recovers this within ±15% on a 50-trial study (test_learning_smoke).
PLANTED_FLOW_WEIGHT = 0.475


class RegimeSwitchingSimulator:
    """Regime-switching market with planted, recoverable persona edges."""

    def __init__(self, seed: Optional[int] = 7, config: Optional[SimConfig] = None):
        self.seed = seed
        self.config = config or SimConfig()
        self.planted_flow_weight = PLANTED_FLOW_WEIGHT

    # ── HMM regime path ──────────────────────────────────────────────────────

    def stationary_distribution(self) -> Dict[str, float]:
        """Power-iterate the transition matrix to its stationary distribution."""
        pi = [1.0 / len(REGIMES)] * len(REGIMES)
        for _ in range(2000):
            nxt = [0.0] * len(REGIMES)
            for i in range(len(REGIMES)):
                for j in range(len(REGIMES)):
                    nxt[j] += pi[i] * TRANSITION[i][j]
            if max(abs(nxt[k] - pi[k]) for k in range(len(REGIMES))) < 1e-12:
                pi = nxt
                break
            pi = nxt
        return {REGIMES[k]: pi[k] for k in range(len(REGIMES))}

    def _gen_regimes(self, n: int, rng: random.Random) -> List[str]:
        # Start from the stationary distribution so the path is unbiased.
        pi = self.stationary_distribution()
        r = rng.random()
        acc = 0.0
        state = 0
        for k, reg in enumerate(REGIMES):
            acc += pi[reg]
            if r <= acc:
                state = k
                break
        path = []
        for _ in range(n):
            path.append(REGIMES[state])
            r = rng.random()
            acc = 0.0
            for j in range(len(REGIMES)):
                acc += TRANSITION[state][j]
                if r <= acc:
                    state = j
                    break
        return path

    # ── Market generation (deterministic given seed, n) ──────────────────────

    def simulate_market(self, n: int) -> List[_Step]:
        cfg = self.config
        rng = random.Random(f"{self.seed}:{n}")
        regimes = self._gen_regimes(n, rng)
        steps: List[_Step] = []
        for reg in regimes:
            crash = (reg == "TRENDING_DOWN") and (rng.random() < cfg.crash_frac)

            # Ground-truth +Nh direction.
            if reg == "TRENDING_UP":
                truth = 1 if rng.random() < cfg.p_truth_trend else -1
            elif reg == "TRENDING_DOWN":
                if crash:
                    truth = 1 if rng.random() < cfg.p_truth_trend else -1   # bounce up
                else:
                    truth = -1 if rng.random() < cfg.p_truth_trend else 1
            else:  # RANGING
                truth = 1 if rng.random() < 0.5 else -1

            votes = self._gen_votes(reg, crash, truth, rng)

            # Fat right tail on wins (V4.2.5 asymmetry); clean fixed losses.
            win_mult = 1.0
            if rng.random() < cfg.fat_tail_prob:
                win_mult = cfg.fat_tail_mult * (0.6 + rng.random())

            fng = 15 if crash else (28 if reg == "TRENDING_DOWN" else
                                    72 if reg == "TRENDING_UP" else 50)
            steps.append(_Step(reg, crash, truth, votes, win_mult, fng))
        return steps

    def _gen_votes(self, reg: str, crash: bool, truth: int, rng: random.Random) -> Dict[str, float]:
        cfg = self.config

        def sig(edge: float) -> float:
            return max(-1.0, min(1.0, truth * edge + rng.gauss(0.0, cfg.noise_sd)))

        def noise() -> float:
            return rng.gauss(0.0, cfg.noise_sd) * 0.5   # zero-mean: no edge

        # FLOW — trend-aligned; flips against the bounce in crash (negative edge).
        if reg == "RANGING":
            flow = noise()
        elif crash:
            flow = sig(cfg.flow_edge_crash)
        else:
            flow = sig(cfg.flow_edge_trend)

        # SENTIMENT — contra-LONG, useful ONLY in crash bounces.
        sent = sig(cfg.sent_edge_crash) if crash else noise()

        # TECHNICAL — mild trend confirmer.
        tech = sig(cfg.tech_edge_trend) if (reg in ("TRENDING_UP", "TRENDING_DOWN") and not crash) else noise()

        return {
            "flow": flow,
            "technical": tech,
            "sentiment": sent,
            "whale": noise(),
            "onchain": noise(),
            "regime": noise(),
        }

    # ── Book generation given a JUDGE weighting ──────────────────────────────

    def simulate_book(
        self,
        judge_weights: Optional[Dict[str, float]] = None,
        n: int = 600,
        pair: str = "BTC",
    ) -> List[TradeOutcome]:
        """Replay the market under `judge_weights`; return realized outcomes.

        JUDGE picks sign(Σ w_p · vote_p). Win → +tp·win_mult, loss → −sl.
        Outcomes carry direction_correct (+Nh), conviction, regime, fng so the
        reward's gap-9 mechanisms have something to chew on.
        """
        weights = dict(judge_weights or JUDGE_SEED_PRIORS)
        wsum = sum(abs(w) for w in weights.values()) or 1.0
        cfg = self.config
        outs: List[TradeOutcome] = []
        for s in self.simulate_market(n):
            score = sum(weights.get(p, 0.0) * v for p, v in s.votes.items())
            if score > 0:
                d = 1
            elif score < 0:
                d = -1
            else:
                # No conviction → a scratch (the wide net), no directional bet.
                outs.append(TradeOutcome(
                    pair=pair, direction="LONG", regime=s.regime, lane="fast",
                    net_pnl_usd=0.0, fees_usd=cfg.fee_usd,
                    direction_correct=None, conviction=0.0, fng=s.fng,
                ))
                continue
            correct = (d == s.truth)
            pnl = cfg.tp_usd * s.win_mult if correct else -cfg.sl_usd
            outs.append(TradeOutcome(
                pair=pair,
                direction="LONG" if d == 1 else "SHORT",
                regime=s.regime,
                lane="fast",
                net_pnl_usd=pnl,
                fees_usd=cfg.fee_usd,
                direction_correct=correct,
                conviction=min(1.0, abs(score) / wsum),
                fng=s.fng,
            ))
        return outs

    def regime_counts(self, n: int) -> Dict[str, int]:
        counts = {r: 0 for r in REGIMES}
        for s in self.simulate_market(n):
            counts[s.regime] += 1
        return counts


def simulate_known_edge(
    judge_weights: Optional[Dict[str, float]] = None,
    n: int = 600,
    seed: int = 7,
    pair: str = "BTC",
) -> List[TradeOutcome]:
    """Convenience: outcomes from a fresh simulator under `judge_weights`."""
    return RegimeSwitchingSimulator(seed=seed).simulate_book(judge_weights, n=n, pair=pair)
