"""Track 2 — CoinMarketCap Strategy Skill (no execution).

The deliverable is a *backtestable strategy spec*, not a live trader: feed CMC market
state in, get a structured, auditable spec out. It is exactly SMT's regime + multi-
persona scoring authored as an LLM Skill — the six personas vote, the learned Judge
aggregates (raw-judge bypass above a confidence floor, SENTIMENT veto-only, HARD-BLOCK
mask), and we emit entry/exit/risk rules plus a ≤500-char "why".

Pure + offline: pass a context built by ``cmc_adapter.build_context`` (or any dict in
the SMT context schema). No network, no keys. Brain imported, never copied.
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger("smt.hackathons.bnb.strategy_skill")

REASON_MAX_CHARS = 500
SKILL_NAME = "smt-multi-persona-regime"
# The six SMT personas, in display order.
_PERSONA_ORDER = ("flow", "technical", "whale", "onchain", "sentiment", "regime")


def _load_personas() -> List[Any]:
    """Instantiate the six SMT personas (imported from the shared brain)."""
    from smt.personas.flow import FlowPersona
    from smt.personas.technical import TechnicalPersona
    from smt.personas.whale import WhalePersona
    from smt.personas.onchain import OnChainPersona
    from smt.personas.sentiment import SentimentPersona
    from smt.personas.regime import RegimePersona
    return [FlowPersona(), TechnicalPersona(), WhalePersona(),
            OnChainPersona(), SentimentPersona(), RegimePersona()]


def build_reason(symbol: str, decision: Any, votes: Dict[str, Any], max_chars: int = REASON_MAX_CHARS) -> str:
    """≤max_chars plain-English 'why', naming the personas that drove the call (XAI payload)."""
    action = getattr(decision, "action", "WAIT")
    conf = int(round(float(getattr(decision, "confidence", 0.0) or 0.0) * 100))
    drivers = sorted(
        [(n, v) for n, v in (votes or {}).items()
         if v is not None and getattr(v, "direction", "NEUTRAL") in ("LONG", "SHORT")],
        key=lambda nv: float(getattr(nv[1], "confidence", 0.0) or 0.0), reverse=True,
    )[:3]
    why = ", ".join(f"{n.capitalize()} {v.direction.lower()} {int(round(v.confidence * 100))}%"
                    for n, v in drivers) or "no persona conviction"
    tail = getattr(decision, "reasoning", "") or ""
    out = f"SMT {action} {symbol} · conf {conf}% — driven by {why}. {tail}".strip()
    return out if len(out) <= max_chars else out[: max_chars - 1].rstrip() + "…"


def generate_strategy_spec(
    symbol: str,
    context: Dict[str, Any],
    *,
    judge: Any = None,
    personas: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Run personas → Judge on `context`, emit a backtestable strategy spec (dict).

    The spec is deterministic given the context, references the un-disableable fee
    floor, and carries the per-persona votes so a judge (or a backtester) can audit
    exactly why the call was made.
    """
    from smt.personas.judge import JudgePersona

    judge = judge or JudgePersona()
    personas = personas if personas is not None else _load_personas()
    votes = JudgePersona.votes_from_personas(personas, symbol, context)
    decision = judge.decide(symbol, votes, context)

    regime = (context.get("regime") or {}).get(symbol, "NORMAL")
    fng = context.get("fear_greed", 50)
    funding = (context.get("funding_rates") or {}).get(symbol)
    action = getattr(decision, "action", "WAIT")
    conf = round(float(getattr(decision, "confidence", 0.0) or 0.0), 4)
    lane = getattr(decision, "lane_hint", None) or "fast"

    spec: Dict[str, Any] = {
        "skill": SKILL_NAME,
        "version": "6.1.0",
        "symbol": symbol,
        "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": context.get("source", "context"),
        "market": {"fear_greed": fng, "regime": regime, "funding_rate": funding},
        "signal": {"action": action, "confidence": conf, "lane": lane},
        "entry_rules": _entry_rules(action, regime),
        "exit_rules": [
            "HARD fee floor: only hold while est_profit_net > round-trip fees (0.12%)",
            "take-profit: lane TP cap (fast tighter than slow); partial-close +0.25% then SL→entry",
            "stop-loss: per-pair ATR-scaled; flow-drop exit if FLOW flips against for N checks",
            "time stop: dynamic max-hold (extend winners, cut dead) — fast ≈ 0.5h, slow ≈ days",
        ],
        "risk": {
            "fee_floor": "net > fees (un-disableable)",
            "drawdown_cap_pct": 30,
            "position_pct": 0.02,
            "sentiment_role": "veto-only (can lower conviction, never raise it)",
            "hard_block": "BTC/ADA/DOGE LONG in BEARISH regime are masked to BLOCK",
        },
        "persona_votes": {
            n: {"direction": getattr(votes.get(n), "direction", "NEUTRAL"),
                "confidence": round(float(getattr(votes.get(n), "confidence", 0.0) or 0.0), 4),
                "why": getattr(votes.get(n), "reasoning", "")}
            for n in _PERSONA_ORDER
        },
        "reasoning": build_reason(symbol, decision, votes),
        "backtest": {
            "horizons_h": [2, 4],
            "metric": "net_pnl_after_fees",
            "validation": "Deflated Sharpe / PBO / CPCV / conformal (smt/learning/validation/)",
            "ground_truth": "+2h/+4h forward-kline direction accuracy (smt/learning/groundtruth.py)",
        },
    }
    log.info("[SKILL] %s %s conf=%.2f regime=%s lane=%s", symbol, action, conf, regime, lane)
    return spec


def _entry_rules(action: str, regime: str) -> List[str]:
    """Human-readable entry rules that mirror the Judge contract (for the backtester)."""
    if action not in ("LONG", "SHORT"):
        return [f"WAIT — Judge conviction below the per-pair raw-judge floor in {regime}; no entry."]
    side = action.lower()
    return [
        f"enter {action} when Judge conviction ≥ per-pair floor (raw-judge bypass) in {regime}",
        f"require FLOW (primary edge) not opposing {side}; TECHNICAL agreement raises size",
        "SENTIMENT may only veto (lower conviction), never trigger an entry alone",
        "skip if the (pair, direction, regime) cell is HARD-BLOCKed",
    ]
