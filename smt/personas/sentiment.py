"""SentimentPersona — funding rate + Gemini-grounded macro bias.

AUTOPSY Finding 3 + PERSONA_INPUT_AUDIT: SENTIMENT is the most
miscalibrated persona. 52% +4h overall, 50% on high-conviction —
expresses confidence it hasn't earned. The Sentiment-neutralization
mechanism is the single highest-leverage win signature across Tier-A
versions (V3.2.224, V3.1.72, V3.2.218, V4.2.5).

Per CLAUDE.md rule 9 (INPUT BEFORE WEIGHT), Session C fixes the prompt
INPUT before touching weights. The fix has two parts:

  1. PAIR_SENTIMENT_CATALYST_SURFACE (V6.0.7b base) — per-pair catalyst
     surface routes Gemini to read the right news vertical for the pair.
  2. CONTRA-pair note (NEW in Session C, AUTOPSY Finding 7 + audit
     finding "DOGE/ADA sentiment peaks mark tops") — for the CONTRA
     pairs (ADA, DOGE, sometimes LTC), the prompt explicitly says to
     INVERT raw bullish/bearish to LONG/SHORT vote.

JUDGE seed prior keeps SENTIMENT at 0.05 weight (veto-only — cannot
lift JUDGE alone). In F&G < 22 (capitulation), the weight is zeroed
entirely (see smt/personas/base.fng_persona_weight_mults).

Inputs (priority order):
  1. context["sentiment_signal"][pair] — pre-computed dict from the
     daemon's batched Gemini call.
  2. context["funding_rates"][pair] — Layer-1 contrarian funding read.
  3. Gemini live call — Session E. For now: degrade to NEUTRAL.

Failure modes (Gemini 429 / 401 / timeout / disabled) → NEUTRAL.
Never raises.

Original V6.0 ref: archive/v6.0/v4/smt_nightly_trade_v3_1.py:6691-7975.
CoinGecko news + CoinDesk veto are NOT ported (retired V5.1.3 / V3.2.277).
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Tuple

from smt.personas.base import Persona, PersonaVote, neutral_vote, bare_pair

log = logging.getLogger("smt.personas.sentiment")

# ── V6.0.7b PAIR_SENTIMENT_CATALYST_SURFACE + V6.1 CONTRA flag ─────────────
# Each pair's Gemini system prompt embeds these fields so the model reads
# the right news vertical AND knows whether to flip raw bull/bear → vote.
# CONTRA pairs invert sentiment (peaks mark tops; boredom marks bottoms).
PAIR_SENTIMENT_CATALYST_SURFACE: Dict[str, Dict[str, Any]] = {
    "BTC": {
        "vertical": "institutional/macro",
        "keywords": ["ETF inflow", "BlackRock", "macro liquidity", "Fed",
                     "FOMC", "wallet 1k+", "halving"],
        "contra": False,
        "lane_bias": "slow",
    },
    "ETH": {
        "vertical": "institutional + structural (staking/L2)",
        "keywords": ["ETH ETF", "staking inflow", "Lido", "EigenLayer",
                     "L2 TVL", "Pectra", "DEX volume"],
        "contra": False,
        "lane_bias": "slow",
    },
    "BNB": {
        "vertical": "exchange / Launchpool mechanics",
        "keywords": ["Launchpool", "BNB Chain TVL", "Binance listings",
                     "burn", "CZ", "exchange risk"],
        "contra": False,
        "lane_bias": "bigwick",
    },
    "SOL": {
        "vertical": "retail + ecosystem (memes + Firedancer)",
        "keywords": ["Solana DEX vol", "Firedancer", "memecoin",
                     "SOL ETF", "validator"],
        "contra": False,
        "lane_bias": "fast",
    },
    "LTC": {
        "vertical": "BTC-dominance proxy + commodity classification",
        "keywords": ["BTC dominance", "CFTC commodity", "halving",
                     "Bitwise LTC ETF"],
        "contra": True,   # PERSONA_CONDITION_MATRIX: SENTIMENT 53% — boredom inverts
        "lane_bias": "slow",
    },
    "XRP": {
        "vertical": "regulatory/legal",
        "keywords": ["SEC", "CLARITY Act", "Ripple", "spot XRP ETF",
                     "Senate Banking", "Hinman"],
        "contra": False,
        "lane_bias": "slow",
    },
    "ADA": {
        "vertical": "founder/governance (CONTRA)",
        "keywords": ["Hoskinson", "DRep", "Cardano governance",
                     "stealth accumulation", "Voltaire"],
        "contra": True,   # V6.1 fix: ADA sentiment peaks mark tops
        "lane_bias": "slow",
    },
    "DOGE": {
        "vertical": "meme / political-proxy (CONTRA)",
        "keywords": ["Musk", "DOGE Day", "200d EMA", "X platform",
                     "Tesla payments"],
        "contra": True,   # V6.1 fix: DOGE retail euphoria = whale exit signal
        "lane_bias": "fast",
    },
}


def build_pair_prompt(pair: str) -> str:
    """Per-pair Gemini system prompt builder.

    V6.1 fix (CLAUDE.md rule 9 — input before weight): explicitly state the
    CONTRA convention for ADA / DOGE / LTC so the model emits LONG when raw
    sentiment is euphoric on a CONTRA pair (peaks mark tops).

    Session E wires this into the live Gemini batch call. Until then this
    function is reachable via `build_pair_prompt(pair)` for tests / smoke.
    """
    bare = bare_pair(pair)
    surface = PAIR_SENTIMENT_CATALYST_SURFACE.get(bare, {
        "vertical": "general crypto", "keywords": [], "contra": False,
        "lane_bias": "fast",
    })
    contra_note = (
        "CONTRA-pair convention — historically euphoric sentiment on this pair "
        "PRECEDES tops (whales distribute into retail FOMO). Output LONG when "
        "raw sentiment is FEARFUL/BORED, SHORT when raw is EUPHORIC. "
    ) if surface["contra"] else (
        "Sentiment-aligned convention — output LONG when raw is BULLISH, "
        "SHORT when raw is BEARISH. "
    )
    return (
        f"You score directional sentiment for {bare}USDT on a 0.5-2.5h trade horizon. "
        f"Read {surface['vertical']} news only. Catalyst keywords: "
        f"{', '.join(surface['keywords'])}. "
        f"{contra_note}"
        f"Output JSON {{\"direction\":\"LONG|SHORT|NEUTRAL\", \"confidence\":0.0-1.0, "
        f"\"reasoning\":\"<one sentence>\"}}. "
        f"Be CONSERVATIVE — NEUTRAL is acceptable; do not over-claim. "
        f"Do not output direction unless news in the last 24h supports it."
    )


class SentimentPersona(Persona):
    name = "sentiment"

    def analyze(self, pair: str, context: Dict[str, Any]) -> PersonaVote:
        try:
            pre = (context.get("sentiment_signal") or {}).get(pair) \
                or (context.get("sentiment_signal") or {}).get(bare_pair(pair))
            if isinstance(pre, dict):
                direction, conf, reason = _normalize_pre(pre)
                if direction in ("LONG", "SHORT"):
                    return PersonaVote(direction=direction, confidence=conf,
                                       reasoning=reason)

            # Layer-1 funding fallback (V3.2.208 contrarian).
            # Positive funding → overleveraged longs → bearish signal.
            fr = (context.get("funding_rates") or {}).get(pair) \
                or (context.get("funding_rates") or {}).get(bare_pair(pair))
            if fr is not None:
                try:
                    rate = float(fr)
                except (TypeError, ValueError):
                    rate = 0.0
                # Funding extremes ±0.025% (V3.2.214)
                if rate >= 0.00025:
                    return PersonaVote(
                        "SHORT",
                        min(0.55, 0.30 + min(abs(rate) * 1000, 0.25)),
                        f"funding {rate*100:+.3f}% (overleveraged longs)",
                    )
                if rate <= -0.00025:
                    return PersonaVote(
                        "LONG",
                        min(0.55, 0.30 + min(abs(rate) * 1000, 0.25)),
                        f"funding {rate*100:+.3f}% (overleveraged shorts)",
                    )
            # No live Gemini call here — Session E. Daemon will batch-fetch and
            # populate context["sentiment_signal"] before personas run.
        except Exception as e:
            log.warning("[SENTIMENT] %s analyze error: %s", pair, e)
        return neutral_vote("no sentiment input")


def _normalize_pre(pre: Dict[str, Any]) -> Tuple[str, float, str]:
    direction = str(pre.get("direction", "NEUTRAL")).upper()
    try:
        conf = float(pre.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    reason = str(pre.get("reasoning") or f"sent {direction}")
    return direction, conf, reason
