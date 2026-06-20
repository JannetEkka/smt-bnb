"""RegimePersona — per-pair candle regime (authoritative since V5.0.7).

PERSONA_INPUT_AUDIT — broken classifier finding (NEW Session C fix
priority): BULLISH label predicts UP 81% (great), BEARISH label
predicts UP 66% (anti-predictive — lagging classifier). NEUTRAL has
mild DN bias. Operator's V3.x band-aids (capitulation FLOOR_GUARD +
regime hysteresis) were patching around this lag.

V6.1 interim fix here: treat BEARISH as "expect bounce" not "stay
short" by emitting LONG when the upstream pair regime says BEARISH.
The proper rewrite (forward-looking features) is Session F.

Inputs:
  1. context["regime"][pair] — pre-computed per-pair regime label
     from compute_pair_regime(symbol) (V5.0.7). Strings:
     TRENDING_UP / TRENDING_DOWN / RECOVERY / CRASH / RANGING / NORMAL.
  2. Nothing → NEUTRAL.

BTC-anchored / global regime input is BANNED (V4.2.3).

Original V6.0 ref: archive/v6.0/v4/smt_nightly_trade_v3_1.py:10575-10923.
"""

from __future__ import annotations
import logging
from typing import Any, Dict

from smt.personas.base import Persona, PersonaVote, neutral_vote, bare_pair

log = logging.getLogger("smt.personas.regime")


class RegimePersona(Persona):
    name = "regime"

    def analyze(self, pair: str, context: Dict[str, Any]) -> PersonaVote:
        try:
            regime_map = context.get("regime") or {}
            label = (regime_map.get(pair)
                     or regime_map.get(bare_pair(pair))
                     or "")
            if isinstance(label, dict):
                label = label.get("regime", "")
            label = str(label).upper()

            if label in ("TRENDING_UP", "RECOVERY", "BULLISH"):
                return PersonaVote("LONG", 0.55, f"regime {label} (audit 81% UP)")
            if label in ("TRENDING_DOWN", "CRASH", "BEARISH"):
                # PERSONA_INPUT_AUDIT: BEARISH label → market goes UP 66%.
                # Interim fix: invert. Real fix in Session F.
                return PersonaVote(
                    "LONG",
                    0.40,
                    f"regime {label} (audit-INVERT — BEARISH is anti-predictive)",
                )
            if label in ("RANGING",):
                # Mild DN bias in NEUTRAL bucket per audit (38% UP).
                return PersonaVote("SHORT", 0.30, f"regime {label} (mild DN bias)")
        except Exception as e:
            log.warning("[REGIME] %s analyze error: %s", pair, e)
        return neutral_vote("no regime label")
