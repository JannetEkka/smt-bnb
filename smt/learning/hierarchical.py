"""Bayesian hierarchical pooling across the 8 pairs (PyMC) — Session G.

ADA and DOGE have thin trade histories; pure per-pair posterior fits are
noisy. Partial-pooling lets them borrow strength from BTC/ETH (which trade
often) while still letting their idiosyncratic regime patterns (DOGE
Musk-political proxy, ADA whale-first frontrunning per docs/research/{ada,
doge}.md) dominate where the data supports it.

Hyperprior: weakly-informative on the cross-pair mean. Per-pair posterior
mean is the deployed value; posterior CI feeds the conformal interval
(smt.learning.validation.conformal).

── DESIGN (documented in Session D; FIT implemented in Session G) ──────────
Model (per tunable θ, e.g. a persona weight or a TP cap):
    μ ~ Normal(prior_mean, prior_sd)              # cross-pair hyper-mean
    τ ~ HalfNormal(τ0)                            # cross-pair spread
    θ_pair ~ Normal(μ, τ)   for pair in 8 pairs   # partial pooling
    y_pair ~ Likelihood(θ_pair, data_pair)
Thin pairs (ADA/DOGE) shrink toward μ; rich pairs (BTC/ETH) stay near their
own data. Deployed value = E[θ_pair | data]; CI = posterior interval.

── GAP 6 (alts-vs-majors) — gate IMPLEMENTED here in Session D ─────────────
Pooling borrows the majors' (BTC/ETH) posterior into the alts. That is the
WRONG direction in extreme fear: in F&G < 22 capitulation, alts bounce on
their own idiosyncratic flush dynamics, NOT in lockstep with BTC — so
borrowing BTC strength mis-points the alt. Therefore pooling is GATED OFF in
the capitulation band and ON in normal/greed. `should_pool()` is the gate the
Session-G fit must honor; the fit itself stays stubbed below.
"""

from __future__ import annotations
import logging
from typing import Optional

from smt.personas.base import bare_pair, fng_band

log = logging.getLogger("smt.learning.hierarchical")

MAJORS = frozenset({"BTC", "ETH"})
ALTS = frozenset({"BNB", "LTC", "SOL", "XRP", "ADA", "DOGE"})


def should_pool(fear_greed: Optional[int]) -> bool:
    """Gap-6 gate: pool EXCEPT in extreme fear (F&G < 22, CMC scale).

    In capitulation, alts decouple from BTC, so borrowing the majors' posterior
    into alts points the wrong way — disable pooling and let each pair stand on
    its own (thin) data.
    """
    return fng_band(fear_greed) != "capitulation"


def pools_into_majors(pair: str, fear_greed: Optional[int]) -> bool:
    """Whether `pair` (an alt) should borrow strength from the majors right now."""
    return bare_pair(pair) in ALTS and should_pool(fear_greed)


def fit_hierarchical(*args, **kwargs):
    raise NotImplementedError(
        "Session G: PyMC partial-pooling across pairs. Design + the gap-6 "
        "should_pool() gate are implemented; the MCMC fit lands in Session G."
    )
