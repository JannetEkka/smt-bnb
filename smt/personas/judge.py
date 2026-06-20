"""JudgePersona — V5.0.9 raw_judge_bypass aggregator.

CLAUDE.md rule 12 contract: when JUDGE conf ≥ floor, JUDGE direction IS
the trade. Bypasses ALL secondary gates (SESSION, FLOW-active-against,
FLOW-STABILITY, 1H-AGREE, 1H-ABS-FLOOR, DISPLACEMENT, FLOW-CONFIRM-queue,
FLOW-BG-revalidation). JUDGE has already weighed every persona; downstream
gates are V3.x band-aids that would re-litigate the decision.

Decision order (Session C):
  1. HARD-BLOCK action mask (V3.2.277). Catastrophic regime cells —
     ALWAYS return BLOCK regardless of vote strength. Three cells in
     base.HARD_BLOCK_CELLS: BTC/ADA/DOGE LONG in BEARISH.
  2. Tally LONG / SHORT weighted-confidence sums using:
        seed prior × F&G band multiplier × per-pair JUDGE weight × vote.confidence
     SENTIMENT contribution is capped: it can only LOWER the leading
     direction's confidence (veto-only — CLAUDE.md rule 12 + AUTOPSY F3).
  3. Capitulation hedge-disable (F&G < 22 CMC): never SHORT — return WAIT
     if leading direction is SHORT.
  4. If leading direction's confidence ≥ raw_judge_min_confidence floor
     → action LONG/SHORT, lane_hint=fast.
  5. Else → action WAIT.

NEVER raises.

Original V6.0 ref: archive/v6.0/v4/smt_nightly_trade_v3_1.py:9254-10575.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List

from smt.personas.base import (
    JudgeDecision,
    JUDGE_SEED_PRIORS,
    JUDGE_CONF_FLOOR,
    PAIR_RAW_JUDGE_FLOOR,
    HARD_BLOCK_CELLS,
    PersonaVote,
    bare_pair,
    fng_band,
    fng_persona_weight_mults,
    regime_bucket,
)

log = logging.getLogger("smt.personas.judge")

# V6.0.7 PAIR_JUDGE_PERSONA_WEIGHTS — per-pair multipliers on top of seed
# priors. Captures "who to trust per pair" from PERSONA_CONDITION_MATRIX.
# DOGE soft across the board (V6.0.7 "all-personas-broken" caution).
PAIR_JUDGE_PERSONA_WEIGHTS: Dict[str, Dict[str, float]] = {
    "BTC":  {"flow": 1.0, "technical": 1.3, "whale": 1.2, "onchain": 1.0, "sentiment": 0.4, "regime": 1.0},
    "ETH":  {"flow": 1.3, "technical": 1.5, "whale": 1.0, "onchain": 1.0, "sentiment": 0.4, "regime": 1.0},
    "BNB":  {"flow": 1.3, "technical": 1.0, "whale": 1.5, "onchain": 1.0, "sentiment": 0.0, "regime": 1.0},  # SENT 38% — silence
    "SOL":  {"flow": 1.0, "technical": 1.3, "whale": 1.3, "onchain": 1.0, "sentiment": 0.25, "regime": 1.0},
    "LTC":  {"flow": 1.0, "technical": 1.5, "whale": 1.4, "onchain": 1.0, "sentiment": 0.25, "regime": 1.0},
    "XRP":  {"flow": 1.6, "technical": 1.2, "whale": 1.0, "onchain": 1.0, "sentiment": 0.0, "regime": 1.0},  # SENT 43% — silence (regulatory-driven)
    "ADA":  {"flow": 1.4, "technical": 1.0, "whale": 1.0, "onchain": 1.0, "sentiment": 0.7, "regime": 1.0},
    "DOGE": {"flow": 1.0, "technical": 1.0, "whale": 1.1, "onchain": 0.85, "sentiment": 0.6, "regime": 0.85},
}


class JudgePersona:
    """JUDGE aggregates persona votes — does NOT subclass Persona (no analyze())."""

    name = "judge"

    def decide(
        self,
        pair: str,
        votes: Dict[str, PersonaVote],
        context: Dict[str, Any],
    ) -> JudgeDecision:
        """Aggregate per-persona votes into the final JUDGE decision.

        `votes` is a dict keyed by persona.name ("flow", "technical", "whale",
        "onchain", "sentiment", "regime"). Missing keys default to NEUTRAL.
        """
        bare = bare_pair(pair)
        vote_map = self._normalize(votes)

        # Putative direction PRE-weighting — needed for HARD-BLOCK lookup.
        long_raw = sum(_directed_conf(vote_map.get(n), "LONG") for n in JUDGE_SEED_PRIORS)
        short_raw = sum(_directed_conf(vote_map.get(n), "SHORT") for n in JUDGE_SEED_PRIORS)
        putative = (
            "LONG" if long_raw > short_raw
            else "SHORT" if short_raw > long_raw
            else "NEUTRAL"
        )

        regime_map = context.get("regime") or {}
        regime_label = regime_map.get(pair) or regime_map.get(bare) or ""
        if isinstance(regime_label, dict):
            regime_label = regime_label.get("regime", "")
        bucket = regime_bucket(str(regime_label))

        # ── 1. HARD-BLOCK action mask ────────────────────────────────────────
        if putative in ("LONG", "SHORT") and (bare, putative, bucket) in HARD_BLOCK_CELLS:
            log.info("[JUDGE V5.0.9] %s %s %s HARD-BLOCK", bare, putative, bucket)
            return JudgeDecision(
                action="BLOCK",
                confidence=0.0,
                reasoning=f"HARD-BLOCK ({bare} {putative} {bucket})",
                persona_breakdown={},
            )

        # ── 2. Weighted-confidence tally (SENTIMENT deferred to veto step) ───
        fg = context.get("fear_greed")
        band = fng_band(fg)
        band_mult = fng_persona_weight_mults(fg)
        pair_mult = PAIR_JUDGE_PERSONA_WEIGHTS.get(
            bare, {p: 1.0 for p in JUDGE_SEED_PRIORS}
        )

        breakdown: Dict[str, float] = {}
        long_score = 0.0
        short_score = 0.0
        for name, prior in JUDGE_SEED_PRIORS.items():
            if name == "sentiment":
                continue  # SENTIMENT is veto-only (handled below)
            v = vote_map.get(name) or PersonaVote("NEUTRAL", 0.0, "")
            w = prior * band_mult.get(name, 1.0) * pair_mult.get(name, 1.0)
            if v.direction == "LONG":
                long_score += w * v.confidence
                breakdown[name] = w * v.confidence
            elif v.direction == "SHORT":
                short_score += w * v.confidence
                breakdown[name] = -w * v.confidence
            else:
                breakdown[name] = 0.0

        if long_score > short_score:
            direction = "LONG"
            conf = long_score
        elif short_score > long_score:
            direction = "SHORT"
            conf = short_score
        else:
            direction = "NEUTRAL"
            conf = 0.0

        # ── 3. SENTIMENT veto-only ───────────────────────────────────────────
        sent_v = vote_map.get("sentiment") or PersonaVote("NEUTRAL", 0.0, "")
        sent_w = (JUDGE_SEED_PRIORS["sentiment"]
                  * band_mult.get("sentiment", 1.0)
                  * pair_mult.get("sentiment", 1.0))
        if (sent_v.direction in ("LONG", "SHORT")
                and direction in ("LONG", "SHORT")
                and sent_v.direction != direction
                and sent_w > 0):
            veto = 0.5 * sent_w * sent_v.confidence
            conf = max(0.0, conf - veto)
            breakdown["sentiment_veto"] = -veto

        # ── 4. Capitulation hedge-disable (F&G < 22 → no SHORT) ──────────────
        if band == "capitulation" and direction == "SHORT":
            log.info("[JUDGE V5.0.9] %s SHORT blocked by capitulation hedge-disable (F&G=%s)",
                     bare, fg)
            return JudgeDecision(
                action="WAIT",
                confidence=conf,
                reasoning=f"capitulation hedge-disable (F&G={fg})",
                persona_breakdown=breakdown,
            )

        # ── 5. Confidence floor (V5.0.9 raw_judge_bypass) ────────────────────
        floor = PAIR_RAW_JUDGE_FLOOR.get(bare, JUDGE_CONF_FLOOR)
        if direction in ("LONG", "SHORT") and conf >= floor:
            log.info("[JUDGE V5.0.9] %s %s conf=%.2f lane=fast (band=%s floor=%.2f)",
                     bare, direction, conf, band, floor)
            return JudgeDecision(
                action=direction,
                confidence=conf,
                reasoning=f"raw_judge {direction} conf={conf:.2f} band={band}",
                persona_breakdown=breakdown,
                lane_hint="fast",
            )

        return JudgeDecision(
            action="WAIT",
            confidence=conf,
            reasoning=f"WAIT (conf {conf:.2f} < floor {floor:.2f}, dir={direction})",
            persona_breakdown=breakdown,
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(votes: Dict[str, PersonaVote]) -> Dict[str, PersonaVote]:
        """Accept lowercased persona names; tolerate dict missing keys."""
        if votes is None:
            return {}
        return {str(k).lower(): v for k, v in votes.items() if v is not None}

    @staticmethod
    def votes_from_personas(personas: List[Any], pair: str, context: Dict[str, Any]
                            ) -> Dict[str, PersonaVote]:
        """Run each persona.analyze() and key results by persona.name.

        Used by the daemon (Session E). Tests inject the vote map directly.
        """
        out: Dict[str, PersonaVote] = {}
        for p in personas:
            try:
                v = p.analyze(pair, context)
            except Exception as e:
                log.warning("[JUDGE] persona %s raised on %s: %s — NEUTRAL",
                            getattr(p, "name", "?"), pair, e)
                v = PersonaVote("NEUTRAL", 0.0, "persona error")
            out[getattr(p, "name", "?").lower()] = v
        return out


def _directed_conf(v: PersonaVote | None, direction: str) -> float:
    if v is None or v.direction != direction:
        return 0.0
    return float(v.confidence)
