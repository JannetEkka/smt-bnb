"""WhalePersona — aggTrades net-flow + ICT BSL/SSL liquidity sweeps.

PERSONA_INPUT_AUDIT reclassified WHALE from "noise" to "best when
confident": 62% +4h overall, 72% when conviction ≥ 70%. Conviction is
the only persona-wide MONOTONIC calibration in the audit. JUDGE should
up-weight high-conviction WHALE votes (handled in JUDGE, not here).

Inputs (priority order):
  1. context["whale_data"][pair] — pre-computed {direction, confidence}.
  2. context["aggtrades"][pair] — raw 15-min net USD bias.
  3. Nothing → NEUTRAL (honest abstain; do NOT default to a direction).

Per-pair whale-trade USD thresholds (V3.2.261 baseline; see archive
monolith line 5109-5118). Scaled to pair's natural volume.

Original V6.0 ref: archive/v6.0/v4/smt_nightly_trade_v3_1.py:6235-6691.
"""

from __future__ import annotations
import logging
from typing import Any, Dict

from smt.personas.base import Persona, PersonaVote, neutral_vote, bare_pair

log = logging.getLogger("smt.personas.whale")

# V3.2.261: per-pair cumulative 15m whale-flow USD threshold.
PAIR_WHALE_CUMULATIVE_15M_USD: Dict[str, float] = {
    "BTC":  1_500_000,
    "ETH":    750_000,
    "BNB":    400_000,
    "LTC":    250_000,
    "SOL":    400_000,
    "XRP":    300_000,
    "ADA":    180_000,
    "DOGE":   120_000,
}


class WhalePersona(Persona):
    name = "whale"

    def analyze(self, pair: str, context: Dict[str, Any]) -> PersonaVote:
        try:
            pre = (context.get("whale_data") or {}).get(pair) \
                or (context.get("whale_data") or {}).get(bare_pair(pair))
            if isinstance(pre, dict):
                direction = str(pre.get("direction", "NEUTRAL")).upper()
                if direction in ("LONG", "SHORT"):
                    return PersonaVote(
                        direction=direction,
                        confidence=_clamp01(pre.get("confidence", 0.5)),
                        reasoning=str(pre.get("reasoning") or f"whale {direction}"),
                    )

            agg = (context.get("aggtrades") or {}).get(pair) \
                or (context.get("aggtrades") or {}).get(bare_pair(pair))
            if isinstance(agg, dict):
                net_usd = float(agg.get("net_15m_usd") or 0.0)
                thresh = PAIR_WHALE_CUMULATIVE_15M_USD.get(bare_pair(pair), 250_000)
                ratio = abs(net_usd) / max(thresh, 1.0)
                if ratio >= 1.0:
                    direction = "LONG" if net_usd > 0 else "SHORT"
                    conf = min(0.80, 0.40 + ratio * 0.20)
                    return PersonaVote(
                        direction=direction,
                        confidence=conf,
                        reasoning=f"agg15m {net_usd:+.0f} USD ratio={ratio:.2f}",
                    )
        except Exception as e:
            log.warning("[WHALE] %s analyze error: %s", pair, e)
        return neutral_vote("no whale flow")


def _clamp01(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, x))
