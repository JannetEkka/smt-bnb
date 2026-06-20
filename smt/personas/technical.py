"""TechnicalPersona — kline-derived structure (RSI + slope + ADX floor).

Per PERSONA_CONDITION_MATRIX: 60% +4h overall, regime-dependent (100%
trending V3.2.224, 22% choppy fear V3.2.277). FIX verdict: regime-
conditional weight (applied in JUDGE, not here). This persona just
emits a binary direction + confidence from structure; JUDGE down-weights
when context["regime"][pair] is RANGING/NEUTRAL.

Inputs (priority order):
  1. context["technical_signal"][pair] — pre-computed dict.
  2. context["klines"][pair#1h] — RSI-14 + 4-bar momentum fallback.
  3. Nothing → NEUTRAL.

ADX floor per pair lives in each `smt/pairs/<pair>.py` CONFIG.adx_floor
(BTC 14, ETH 16, BNB 12, SOL 10, LTC/XRP/ADA/DOGE 14). We do not enforce
the floor here — JUDGE consults it when down-weighting low-ADX votes.

Original V6.0 ref: archive/v6.0/v4/smt_nightly_trade_v3_1.py:8576-9254.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List

from smt.personas.base import Persona, PersonaVote, neutral_vote

log = logging.getLogger("smt.personas.technical")


class TechnicalPersona(Persona):
    name = "technical"

    def analyze(self, pair: str, context: Dict[str, Any]) -> PersonaVote:
        try:
            pre = (context.get("technical_signal") or {}).get(pair)
            if isinstance(pre, dict):
                direction = str(pre.get("direction", "NEUTRAL")).upper()
                if direction in ("LONG", "SHORT"):
                    return PersonaVote(
                        direction=direction,
                        confidence=_clamp01(pre.get("confidence", 0.5)),
                        reasoning=str(pre.get("reasoning") or f"tech {direction}"),
                    )

            closes = self._get_closes(pair, context, n=20)
            if len(closes) >= 15:
                rsi = _rsi_14(closes)
                if rsi is not None:
                    if rsi <= 25:
                        return PersonaVote("LONG", 0.65, f"RSI {rsi:.1f} oversold")
                    if rsi >= 75:
                        return PersonaVote("SHORT", 0.65, f"RSI {rsi:.1f} overbought")
                    # Mid-range: structure-slope fallback (4-bar momentum)
                    if closes[0] > 0:
                        chg = (closes[-1] - closes[-4]) / closes[-4] * 100.0
                        if chg > 0.40:
                            return PersonaVote(
                                "LONG",
                                min(0.55, 0.30 + abs(chg) * 0.04),
                                f"4-bar +{chg:.2f}%, RSI {rsi:.1f}",
                            )
                        if chg < -0.40:
                            return PersonaVote(
                                "SHORT",
                                min(0.55, 0.30 + abs(chg) * 0.04),
                                f"4-bar {chg:.2f}%, RSI {rsi:.1f}",
                            )
        except Exception as e:
            log.warning("[TECHNICAL] %s analyze error: %s", pair, e)
        return neutral_vote("no kline structure")

    def _get_closes(self, pair: str, context: Dict[str, Any], n: int = 20) -> List[float]:
        klines = context.get("klines") or {}
        for key in (f"{pair}#1h", f"{pair}_1h", f"{pair}:1h"):
            candles = klines.get(key) or []
            if len(candles) >= n:
                try:
                    return [float(c[4]) for c in candles[-n:]]
                except (IndexError, ValueError):
                    return []
        return []


def _clamp01(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, x))


def _rsi_14(closes: List[float]) -> float | None:
    if len(closes) < 15:
        return None
    gains = []
    losses = []
    for i in range(1, 15):
        delta = closes[-15 + i] - closes[-16 + i]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
    avg_g = sum(gains) / 14.0
    avg_l = sum(losses) / 14.0
    if avg_l == 0:
        return 100.0 if avg_g > 0 else 50.0
    rs = avg_g / avg_l
    return 100.0 - (100.0 / (1.0 + rs))
