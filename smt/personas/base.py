"""Persona ABC + PersonaVote + JudgeDecision + F&G context helpers.

Contract: every Persona implements `analyze(pair, context) -> PersonaVote`.
NEUTRAL votes get diluted (0.25× per V3.2.x NEUTRAL_DILUTION) in JUDGE and
do NOT count toward the 2-personas-min rule.

A persona that hits rate-limit / 401 / timeout MUST return NEUTRAL —
never default to a direction. Silent failure of a persona is OK (noise
can't materially hurt PnL); silent failure of execution is NOT.

The F&G context-multiplier curve (Session C addendum, AUTOPSY Finding 7)
lives here so JUDGE and any future weight-aware persona can share it:

    F&G < 22 (CMC) → capitulation: hedge-disable SHORT; SENTIMENT zeroed;
                     FLOW raised (0.40 → 0.55).
    F&G 22-75      → normal weights.
    F&G > 75       → greed: SENTIMENT still veto-only; partial-close sooner
                     (latter consumed by exit_cascade, not by JUDGE here).

F&G calibration: alt.me ≈ CMC − 7pts. All thresholds in this codebase are
already on the CMC scale — do NOT re-add −7 offsets.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional


VoteDirection = Literal["LONG", "SHORT", "NEUTRAL"]
JudgeAction = Literal["LONG", "SHORT", "WAIT", "BLOCK"]


@dataclass
class PersonaVote:
    direction: VoteDirection
    confidence: float                # 0.0-1.0
    reasoning: str = ""              # one short line for the dashboard


@dataclass
class JudgeDecision:
    """Output of JudgePersona.decide().

    action:
      LONG / SHORT — JUDGE conf cleared the raw_judge floor, direction set
      WAIT         — conf below floor or no clear direction; no entry this cycle
      BLOCK        — HARD-BLOCK action mask fired (catastrophic regime cell)
    """
    action: JudgeAction
    confidence: float
    reasoning: str = ""
    persona_breakdown: Dict[str, float] = field(default_factory=dict)
    lane_hint: Literal["fast", "bigwick", "slow"] = "fast"


# ── Seed JUDGE priors (Session C). Sum to 1.0 ──────────────────────────────
# FLOW-heavy, SENTIMENT veto-only (low base weight, can never carry alone).
# SENTIMENT × FLOW alone in capitulation: SENT zeroed, FLOW lifted to 0.55.
JUDGE_SEED_PRIORS: Dict[str, float] = {
    "flow":      0.40,
    "technical": 0.30,
    "whale":     0.10,
    "onchain":   0.10,
    "sentiment": 0.05,   # veto-only — cannot lift JUDGE on its own
    "regime":    0.05,
}

# raw_judge_min_confidence floor (V5.0.9 contract — see CLAUDE.md rule 12).
# V3.2.x ran 0.90 on a wider weighted sum; with normalized priors (sum=1.0)
# 0.55 is the equivalent operating point. Tune in Session D via Optuna TPE.
JUDGE_CONF_FLOOR: float = 0.55

# Per-pair raw_judge floor override (V6.0.7b — populate post-calibration).
PAIR_RAW_JUDGE_FLOOR: Dict[str, float] = {}

# V3.2.277 HARD-BLOCK action mask. (pair, direction, regime_bucket).
# Aggregated from 80d RL data — combos with n>=20 AND WR<15% AND avg<-$15.
# These ALWAYS lose; bypass JUDGE entirely with action=BLOCK.
HARD_BLOCK_CELLS = frozenset({
    ("BTC",  "LONG", "BEARISH"),
    ("ADA",  "LONG", "BEARISH"),
    ("DOGE", "LONG", "BEARISH"),
})


def fng_band(fear_greed: Optional[int]) -> str:
    """V6.1 CMC-scale F&G band classifier."""
    if fear_greed is None:
        return "normal"
    try:
        fg = int(fear_greed)
    except (TypeError, ValueError):
        return "normal"
    if fg < 22:
        return "capitulation"
    if fg > 75:
        return "greed"
    return "normal"


def fng_persona_weight_mults(fear_greed: Optional[int]) -> Dict[str, float]:
    """Per-band multipliers applied on top of JUDGE_SEED_PRIORS.

    Capitulation: FLOW × 1.375 (0.40 → 0.55), SENTIMENT × 0 (zeroed).
    Greed: SENTIMENT still veto-only (× 1.0; veto is applied separately).
    Normal: identity.
    """
    band = fng_band(fear_greed)
    if band == "capitulation":
        return {
            "flow":      1.375,
            "technical": 1.0,
            "whale":     1.0,
            "onchain":   1.0,
            "sentiment": 0.0,
            "regime":    1.0,
        }
    return {p: 1.0 for p in JUDGE_SEED_PRIORS}


def regime_bucket(regime: Optional[str]) -> str:
    """Map a fine-grained regime label to the coarse BULLISH / BEARISH / NEUTRAL
    bucket the JUDGE / bandit use.

    Regime is really a 2-D stack: DIRECTION (up / down / sideways) × VOLATILITY
    (quiet / volatile / squeeze). The standard ADX heuristic: ADX>25 → trending,
    ADX<20 → CHOPPY/ranging, 20-25 → transition. The non-directional volatility
    states (RANGING, CHOPPY, VOLATILE, QUIET, SQUEEZE) all bucket NEUTRAL here —
    they say "no directional edge", which is exactly NEUTRAL for vote weighting.
    Session E's compute_pair_regime should emit CHOPPY (low-ADX whipsaw) distinct
    from RANGING (clean band) so the exit cascade can treat them differently.
    """
    r = (regime or "").upper()
    if r in ("TRENDING_UP", "RECOVERY", "BULLISH"):
        return "BULLISH"
    if r in ("TRENDING_DOWN", "CRASH", "BEARISH"):
        return "BEARISH"
    # RANGING / CHOPPY / VOLATILE / QUIET / SQUEEZE / NORMAL → no directional edge
    return "NEUTRAL"


def neutral_vote(reason: str = "no data") -> PersonaVote:
    """Standard NEUTRAL output for blank-context / degraded paths."""
    return PersonaVote(direction="NEUTRAL", confidence=0.0, reasoning=reason)


def bare_pair(pair: str) -> str:
    """'BTCUSDT' → 'BTC'; 'BTC' passes through."""
    return (pair or "").replace("USDT", "").replace("usdt", "").upper()


class Persona(ABC):
    name: str = ""                   # set by subclass

    @abstractmethod
    def analyze(self, pair: str, context: Dict[str, Any]) -> PersonaVote:
        ...
