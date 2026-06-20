"""TrainData (exp) schema + JSONL writer — the learning loop's data tap.

Session E defines the v6.1 experience record. The daemon writes ONE *eval*
record per evaluated cycle-cell (pair × cycle) plus ONE *close* record per
position that closes. Both share a STABLE schema: every record carries every
key; eval records leave the on-close fields ``None`` until the join.

Why one schema, two events:
- The learner (`smt.learning.corpus`) needs WAIT/BLOCK cells too — gap-9
  direction-quality is computed over the FULL corpus (every decision, wins AND
  losses), not just executed winners. So we log a record even when the daemon
  does nothing this cycle.
- `seed_from_corpus` / `to_outcomes` key on `win`/`pnl_usd`; eval records carry
  ``win=None`` so they are skipped as outcomes and only the close records feed
  the bandit/reward. `split_cmt_uat` keys on `mode`.

REQUIRED fields (Session E acceptance) — every record carries all of them:
  persona_votes (all personas incl onchain as {name:{signal,confidence}}),
  lane, conviction (raw JUDGE conf), fear_greed (CMC), btc_dominance, regime
  (per-pair), direction, action (LONG/SHORT/WAIT/BLOCK), entry_price, ts, mode
  (UAT/CMT) — plus the on-close join fields pnl_usd, pnl_pct, exit_reason,
  hours_open, win.

The +2h / +4h ground-truth direction accuracy is NOT computed here: we log
``entry_price`` + ``ts`` so a later pass (Session F) joins forward klines.
"""

from __future__ import annotations
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

log = logging.getLogger("smt.experience")

SCHEMA_VERSION = "6.1.0"
EXP_DIR = "v4/rl_training_data"

# Persona vote slots — the schema always carries all of them so the learner can
# count a persona's silence (NEUTRAL) as data, not a missing key. `onchain` (Oc)
# is the slot the V6.0 schema never had — closing that gap is a Session E goal.
PERSONA_NAMES = ("flow", "technical", "whale", "onchain", "sentiment", "regime")

# Single source of truth for the acceptance assertion (test imports this).
REQUIRED_FIELDS = (
    "persona_votes", "lane", "conviction", "fear_greed", "btc_dominance",
    "regime", "direction", "action", "entry_price", "ts", "mode",
    "reasoning",                       # XAI: ≤500-char human-readable "why"
    # on-close join fields
    "pnl_usd", "pnl_pct", "exit_reason", "hours_open", "win",
)

# XAI: the "why" string is capped so it stays glanceable in logs + dashboards.
REASONING_MAX_CHARS = 500


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _f(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def serialize_votes(votes: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """{persona_name: PersonaVote} → {name: {signal, confidence}} for ALL personas.

    Accepts PersonaVote objects or already-serialized {signal,confidence} dicts.
    Missing personas default to NEUTRAL/0.0 so the onchain (Oc) slot is always
    present (V6.0 logging gap).
    """
    votes = votes or {}
    out: Dict[str, Dict[str, Any]] = {}
    for name in PERSONA_NAMES:
        v = votes.get(name)
        if v is None:
            out[name] = {"signal": "NEUTRAL", "confidence": 0.0}
            continue
        if isinstance(v, dict):
            sig = str(v.get("signal") or v.get("direction") or "NEUTRAL").upper()
            conf = _f(v.get("confidence"), 0.0)
        else:  # PersonaVote
            sig = str(getattr(v, "direction", "NEUTRAL") or "NEUTRAL").upper()
            conf = _f(getattr(v, "confidence", 0.0), 0.0)
        out[name] = {"signal": sig, "confidence": conf}
    return out


def votes_from_breakdown(breakdown: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Reconstruct {name:{signal,confidence}} from a JUDGE persona_breakdown.

    The breakdown stored on a position is ``{name: signed_float}`` (JUDGE's
    signed contribution). Sign → direction, |value| → confidence. Non-persona
    keys (e.g. ``sentiment_veto``) are ignored. Used as the close-record
    fallback when a position predates the per-position `exp_entry` stash.
    """
    breakdown = breakdown or {}
    norm: Dict[str, Any] = {}
    for name in PERSONA_NAMES:
        val = breakdown.get(name)
        if val is None:
            norm[name] = {"signal": "NEUTRAL", "confidence": 0.0}
            continue
        v = _f(val, 0.0)
        sig = "LONG" if v > 0 else "SHORT" if v < 0 else "NEUTRAL"
        norm[name] = {"signal": sig, "confidence": abs(v)}
    return norm


def build_eval_record(
    *,
    pair: str,
    mode: str,
    lane: str,
    direction: str,
    action: str,
    conviction: float,
    conviction_scaled: float,
    fear_greed: Optional[int],
    btc_dominance: Optional[float],
    regime: str,
    entry_price: float,
    persona_votes: Dict[str, Dict[str, Any]],
    executed: bool,
    tracker_key: Optional[str] = None,
    judge_action: Optional[str] = None,
    bandit_veto: bool = False,
    reasoning: str = "",
    ts: Optional[str] = None,
) -> Dict[str, Any]:
    """One evaluated cycle-cell record. On-close fields are ``None`` until join."""
    return {
        "schema": SCHEMA_VERSION,
        "event": "eval",
        "ts": ts or _now_iso(),
        "pair": pair,
        "mode": mode,
        "lane": lane,
        "direction": direction,
        "action": action,
        "judge_action": judge_action or action,
        "reasoning": str(reasoning or "")[:REASONING_MAX_CHARS],
        "conviction": _f(conviction),
        "conviction_scaled": _f(conviction_scaled),
        "bandit_veto": bool(bandit_veto),
        "fear_greed": int(fear_greed) if fear_greed is not None else None,
        "btc_dominance": _f(btc_dominance) if btc_dominance is not None else None,
        "regime": str(regime or "NORMAL"),
        "entry_price": _f(entry_price),
        "persona_votes": persona_votes,
        "executed": bool(executed),
        "tracker_key": tracker_key,
        # ── on-close join fields (filled by the close record) ──
        "pnl_usd": None,
        "pnl_pct": None,
        "exit_reason": None,
        "hours_open": None,
        "win": None,
    }


def build_close_record(
    *,
    entry_ctx: Dict[str, Any],
    pnl_usd: float,
    pnl_pct: float,
    exit_reason: str,
    hours_open: float,
    win: bool,
    ts: Optional[str] = None,
) -> Dict[str, Any]:
    """A close record = the entry-context record + populated on-close fields.

    Carries every REQUIRED field with real (non-None) close values so a single
    record is self-contained for the learner.
    """
    rec = dict(entry_ctx or {})
    rec.update(
        event="close",
        ts_close=ts or _now_iso(),
        executed=True,
        pnl_usd=_f(pnl_usd),
        pnl_pct=_f(pnl_pct),
        exit_reason=str(exit_reason or ""),
        hours_open=_f(hours_open),
        win=bool(win),
    )
    # `side` mirrors `direction` so corpus.seed_from_corpus / to_outcomes consume it.
    rec.setdefault("side", rec.get("direction") or "")
    return rec


def exp_path(directory: str = EXP_DIR, when: Optional[datetime] = None) -> str:
    d = when or datetime.now(timezone.utc)
    return os.path.join(directory, f"exp_{d:%Y%m%d}.jsonl")


def write_record(record: Dict[str, Any], directory: str = EXP_DIR) -> str:
    """Append one JSONL record to today's exp file. Returns the path written."""
    path = exp_path(directory)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return path
