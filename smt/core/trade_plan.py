"""TradePlan: structured output of Strategy.entry_signal().

Every entry candidate the bot considers becomes a TradePlan. The
JUDGE and risk gate consume it; smt.core.execution.place() turns it
into a WEEX algoOrder; smt.core.tracker stores it; the logger writes
it to TrainData JSONL for the learning loop.

A `None` return from entry_signal() means "no trade this cycle".

ExitDecision / HoldDecision mirror this on the open-position side.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional


Lane = Literal["fast", "bigwick", "slow"]
Direction = Literal["LONG", "SHORT"]


@dataclass
class TradePlan:
    pair: str
    lane: Lane
    direction: Direction
    entry_price: float
    exit_target: float
    exit_stop: float
    hold_max: float                  # hours
    qty: float
    leverage: int
    est_fees: float                  # USD, round-trip taker
    est_profit_net: float            # USD; must exceed est_fees (risk gate)
    est_time_hours: float            # expected hold, not the cap
    decision_confidence: float       # JUDGE composite, 0.0-1.0
    persona_votes: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""              # piped to dashboard + Discord


@dataclass
class ExitDecision:
    should_exit: bool
    reason: str                       # e.g. FLOW_AGAINST / CHANDELIER_TRAIL / MAX_HOLD
    target_price: Optional[float] = None   # None = market close


@dataclass
class HoldDecision:
    should_hold: bool
    adjust_target: Optional[float] = None  # raise TP?
    adjust_stop: Optional[float] = None    # nudge SL?
    reasoning: str = ""
