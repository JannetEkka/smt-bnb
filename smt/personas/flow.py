"""FlowPersona — orderbook + recent-volume bias.

Primary directional signal per AUTOPSY Finding 3 (66% +4h, 74% +2h —
most stable across all 11 winning versions). Conviction is mildly
inverted (PERSONA_INPUT_AUDIT Table) so treat as BINARY: direction with
a base confidence, not a smooth confidence scale.

Inputs (consumed in priority order; all optional — degrades to NEUTRAL):
  1. context["flow_signal"][pair] — pre-computed {direction, confidence}
     dict injected by the daemon (Session E will populate this from the
     WEEX 60% + Binance 40% orderbook composite per V3.2.183/V6.0.9).
  2. context["klines"][pair#1h] — fall-back 4-bar slope when no live flow.
  3. Nothing → NEUTRAL.

NEVER raises. Per CLAUDE.md rule 9 (INPUT BEFORE WEIGHT), if you tune
FLOW's JUDGE weight, audit the input feed FIRST.

Original V6.0 ref: archive/v6.0/v4/smt_nightly_trade_v3_1.py:7975-8576.
We do NOT port the 600-line monolith — Session C carries the contract,
Session E wires the live composite into context["flow_signal"].
"""

from __future__ import annotations
import logging
from typing import Any, Dict

from smt.personas.base import Persona, PersonaVote, neutral_vote

log = logging.getLogger("smt.personas.flow")


class FlowPersona(Persona):
    name = "flow"

    def analyze(self, pair: str, context: Dict[str, Any]) -> PersonaVote:
        try:
            flow_signal = (context.get("flow_signal") or {}).get(pair) \
                or (context.get("flow_signal") or {}).get(pair.replace("USDT", ""))
            if isinstance(flow_signal, dict):
                direction = str(flow_signal.get("direction", "NEUTRAL")).upper()
                if direction in ("LONG", "SHORT"):
                    conf = _clamp01(flow_signal.get("confidence", 0.5))
                    return PersonaVote(
                        direction=direction,
                        confidence=conf,
                        reasoning=f"flow_signal {direction} {conf:.0%}",
                    )

            klines = context.get("klines") or {}
            for key in (f"{pair}#1h", f"{pair}_1h", f"{pair}:1h"):
                candles = klines.get(key) or []
                if len(candles) >= 4:
                    try:
                        closes = [float(c[4]) for c in candles[-4:]]
                    except (IndexError, ValueError):
                        break
                    if closes[0] <= 0:
                        break
                    chg_pct = (closes[-1] - closes[0]) / closes[0] * 100.0
                    if chg_pct > 0.5:
                        return PersonaVote(
                            "LONG",
                            min(0.60, 0.30 + abs(chg_pct) * 0.05),
                            f"4h slope +{chg_pct:.2f}%",
                        )
                    if chg_pct < -0.5:
                        return PersonaVote(
                            "SHORT",
                            min(0.60, 0.30 + abs(chg_pct) * 0.05),
                            f"4h slope {chg_pct:.2f}%",
                        )
                    break
        except Exception as e:
            log.warning("[FLOW] %s analyze error: %s", pair, e)
        return neutral_vote("no flow data")


def _clamp01(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x
