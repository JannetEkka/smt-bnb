"""OnChainPersona — 7th persona (V6.0.8). Per-pair on-chain reads.

Per-pair signal sources (live wiring in Session E):
  BTC : Binance spot-perp 24h basis + 7d aggTrades stealth accumulation
  ETH : DefiLlama Lido + RocketPool + EigenLayer 24h TVL Δ
  BNB : DefiLlama BSC chain 24h TVL Δ (no key)
  SOL : DefiLlama Solana DEX 24h vol vs 30d p80 + 7d aggTrades
  LTC : CMC global-metrics BTC-dominance 24h shift
  XRP : 7d aggTrades only
  ADA : Koios /drep_list 24h DRep-count Δ
  DOGE: 7d aggTrades softened ×0.85 (V6.0.7 all-personas-broken audit)

NEUTRAL on rate-limit / 401 / timeout — never defaults to a direction.

PERSONA_INPUT_AUDIT: cannot audit (V6.0.8 added persona but trade-log
schema never extended). Logging gap fix → Session B (done in tracker.py).
JUDGE seed prior keeps ONCHAIN at 0.10 until real-data calibration.

Inputs:
  1. context["onchain_signal"][pair] — pre-computed dict.
  2. Nothing → NEUTRAL.

Original V6.0 ref: archive/v6.0/v4/smt_nightly_trade_v3_1.py:10928-11448.
"""

from __future__ import annotations
import logging
from typing import Any, Dict

from smt.personas.base import Persona, PersonaVote, neutral_vote, bare_pair

log = logging.getLogger("smt.personas.onchain")

# V6.0.7 DOGE softening — "all personas broken" audit caution.
ONCHAIN_CONF_SOFTEN: Dict[str, float] = {
    "DOGE": 0.85,
}


class OnChainPersona(Persona):
    name = "onchain"

    def analyze(self, pair: str, context: Dict[str, Any]) -> PersonaVote:
        try:
            pre = (context.get("onchain_signal") or {}).get(pair) \
                or (context.get("onchain_signal") or {}).get(bare_pair(pair))
            if isinstance(pre, dict):
                direction = str(pre.get("direction", "NEUTRAL")).upper()
                if direction in ("LONG", "SHORT"):
                    conf = _clamp01(pre.get("confidence", 0.5))
                    soften = ONCHAIN_CONF_SOFTEN.get(bare_pair(pair), 1.0)
                    return PersonaVote(
                        direction=direction,
                        confidence=conf * soften,
                        reasoning=(str(pre.get("reasoning") or f"onchain {direction}")
                                   + (f" (soften ×{soften:.2f})" if soften != 1.0 else "")),
                    )
        except Exception as e:
            log.warning("[ONCHAIN] %s analyze error: %s", pair, e)
        return neutral_vote("no onchain input")


def _clamp01(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, x))
