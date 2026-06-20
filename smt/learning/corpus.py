"""Cold-start corpus loader (operator-mandated CORPUS SCOPE, 2026-06-07).

Sources:
- docs/data/trades.json — 5,382 trades (3,862 executed), Feb-May 2026. Rich
  per-trade schema (pair/side/lane/regime/win/pnl_usd/personas/...).
- v4/rl_training_data/exp_*.jsonl — daemon-cycle experience (the `cmt_` prefix
  marks competition-mode records).

Rules (operator):
- NO "80-day window" framing — use the FULL daemon-cycle history.
- Split CMT (competition-mode: different equity/leverage) from UAT and analyze
  them SEPARATELY — never blend.
- Forward-validation: Jan-Mar trains, Apr-May validates (no look-ahead leak).

Everything degrades gracefully: a missing file yields an empty list, so the
learner runs cold rather than crashing (the cold-start path itself).
"""

from __future__ import annotations
import glob
import json
import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from smt.learning.reward import TradeOutcome, direction_quality_weights, Cell

log = logging.getLogger("smt.learning.corpus")

TRADES_JSON = "docs/data/trades.json"
EXPERIENCE_DIR = "v4/rl_training_data"
DERIVED_OUTCOMES_JSON = "v4/derived_outcomes.json"
TRAIN_MONTHS = {"01", "02", "03"}      # Jan-Mar → train
VALIDATE_MONTHS = {"04", "05"}         # Apr-May → validate

_PERSONA_CODE = {"Wh": "whale", "Sn": "sentiment", "Fl": "flow",
                 "Tc": "technical", "Oc": "onchain", "Rg": "regime"}
_VOTE_RE = re.compile(r"\s*([LSN])\s*\((\d+)")


# ── loading ───────────────────────────────────────────────────────────────────

def load_trades(path: str = TRADES_JSON) -> List[Dict[str, Any]]:
    """Load the trades array from trades.json. [] if absent/unreadable."""
    if not os.path.exists(path):
        log.info("[CORPUS] %s not found — cold start", path)
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("trades", []) if isinstance(data, dict) else list(data)
    except Exception as exc:
        log.warning("[CORPUS] failed to read %s: %s", path, exc)
        return []


def load_derived_outcomes(path: str = DERIVED_OUTCOMES_JSON) -> List[Dict[str, Any]]:
    """Load per-trade outcomes derived from the raw daemon logs.

    Built by `scripts/derive_outcomes_from_logs.py` — RICHER than trades.json
    (carries regime + F&G + exit reason + Gross/Net PnL per close). This is the
    preferred bandit-seeding source when present; trades.json is the fallback.
    Records already use the bandit/outcome field names (pair/side/regime/win/
    net_pnl_usd), so `seed_from_corpus` and `to_outcomes` consume them directly.
    """
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("outcomes", []) if isinstance(data, dict) else list(data)
    except Exception as exc:
        log.warning("[CORPUS] failed to read %s: %s", path, exc)
        return []


def load_best_corpus() -> List[Dict[str, Any]]:
    """Prefer the log-derived outcomes (regime-tagged); fall back to trades.json."""
    derived = load_derived_outcomes()
    if derived:
        log.info("[CORPUS] using %d log-derived outcomes (regime-tagged)", len(derived))
        return derived
    return load_trades()


def load_experience(directory: str = EXPERIENCE_DIR, limit_files: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load exp_*.jsonl daemon-cycle records. [] if dir absent."""
    if not os.path.isdir(directory):
        return []
    out: List[Dict[str, Any]] = []
    files = sorted(glob.glob(os.path.join(directory, "exp_*.jsonl")))
    if limit_files:
        files = files[:limit_files]
    for fp in files:
        try:
            with open(fp) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.append(json.loads(line))
        except Exception as exc:
            log.warning("[CORPUS] skipping %s: %s", fp, exc)
    return out


# ── CMT / UAT split ───────────────────────────────────────────────────────────

def is_cmt(record: Dict[str, Any]) -> bool:
    """True if a record is competition-mode (cmt_ marker in id/symbol/mode)."""
    blob = " ".join(str(record.get(k, "")) for k in ("id", "symbol", "mode", "account", "version"))
    return "cmt" in blob.lower()


def split_cmt_uat(records: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Partition records into (cmt, uat). Analyze separately — never blend."""
    cmt, uat = [], []
    for r in records:
        (cmt if is_cmt(r) else uat).append(r)
    return cmt, uat


# ── forward validation (no look-ahead) ────────────────────────────────────────

def _month_of(record: Dict[str, Any]) -> Optional[str]:
    ts = record.get("timestamp") or record.get("ts") or record.get("closed_at")
    if not ts:
        return None
    m = re.search(r"\d{4}-(\d{2})-\d{2}", str(ts))
    return m.group(1) if m else None


def forward_validation_split(
    records: Iterable[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(train Jan-Mar, validate Apr-May). Undated records go to train."""
    train, validate = [], []
    for r in records:
        mo = _month_of(r)
        if mo in VALIDATE_MONTHS:
            validate.append(r)
        else:
            train.append(r)
    return train, validate


# ── trades.json → TradeOutcome (so reward.* mechanisms apply) ──────────────────

def to_outcomes(records: Iterable[Dict[str, Any]]) -> List[TradeOutcome]:
    """Map executed trade records to TradeOutcome.

    `win` (realized PnL > 0) is used as the +Nh direction-correct proxy — the
    independent +2h ground truth lives in docs/data/groundtruth.json (Session F).
    """
    outs: List[TradeOutcome] = []
    for r in records:
        if not isinstance(r, dict) or r.get("skipped"):
            continue
        pair = r.get("pair") or r.get("symbol")
        side = r.get("side") or r.get("action")
        if pair is None or side is None:
            continue
        win = r.get("win")
        pnl = r.get("pnl_usd")
        outs.append(TradeOutcome(
            pair=str(pair),
            direction=str(side).upper(),
            regime=str(r.get("regime", "NORMAL")),
            lane=str(r.get("lane") or "fast"),
            net_pnl_usd=float(pnl) if pnl is not None else 0.0,
            fees_usd=0.0,
            direction_correct=(bool(win) if win is not None else None),
            conviction=float(r.get("confidence_pct", 0.0)) / 100.0,
            fng=r.get("fear_greed"),
        ))
    return outs


def direction_quality_posteriors(records: Iterable[Dict[str, Any]]) -> Dict[Cell, float]:
    """Per-cell Beta-posterior direction accuracy over the corpus (gap-9 DATA).

    Low-accuracy cells get low weight from the data itself — nothing is excluded.
    """
    return direction_quality_weights(to_outcomes(records))


DEFAULT_DAEMON_LOG_DIR = "v4/logs"
_LOG_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*\|\s*(\w+)\s*\|\s*(.*)$")
_SIGNAL_LINE = re.compile(r"^\s*([A-Z]{2,6}):\s*(LONG|SHORT)\s*\((\d+)%\)")
_CLOSE_LINE = re.compile(r"([A-Za-z_]+(?:usdt|USDT))\s+closed")


def load_daemon_logs(directory: str = DEFAULT_DAEMON_LOG_DIR, limit_files: Optional[int] = None
                     ) -> List[Dict[str, Any]]:
    """Parse `daemon_*.log` lines into structured events.

    The daemon logs (Jan 8 → May, 109 files) are the RAW flight recorder. They
    include the January competition window that trades.json (Feb 1+) misses, plus
    per-cycle persona signal %. BUT they do NOT log regime/F&G, and per-trade
    win/loss is only semi-structured (`"trades_closed"` counters + `<pair> closed`
    cleanup lines) — so they cannot cleanly feed the bandit's win/loss update on
    their own (trades.json remains the clean outcome source). Their real payoff is
    feature-rich cycle data, best mined by loading to BigQuery (the BQML route in
    docs/ML_ARCHITECTURE.md), not ad-hoc Python on 394 MB.

    Returns event dicts: {"kind": "signal", ts, pair, side, confidence} for entry
    signals, {"kind": "close", ts, pair} for close-cleanup lines.
    """
    if not os.path.isdir(directory):
        return []
    events: List[Dict[str, Any]] = []
    files = sorted(glob.glob(os.path.join(directory, "daemon_*.log")))
    if limit_files:
        files = files[:limit_files]
    for fp in files:
        try:
            with open(fp, errors="ignore") as f:
                for line in f:
                    m = _LOG_LINE.match(line)
                    if not m:
                        continue
                    ts, _level, msg = m.group(1), m.group(2), m.group(3)
                    s = _SIGNAL_LINE.match(msg)
                    if s:
                        events.append({"kind": "signal", "ts": ts, "pair": s.group(1),
                                       "side": s.group(2), "confidence": int(s.group(3)) / 100.0})
                        continue
                    c = _CLOSE_LINE.search(msg)
                    if c and "closed" in msg.lower():
                        events.append({"kind": "close", "ts": ts,
                                       "pair": c.group(1).upper().replace("CMT_", "")})
        except Exception as exc:
            log.warning("[CORPUS] skipping %s: %s", fp, exc)
    return events


def daemon_signal_events(directory: str = DEFAULT_DAEMON_LOG_DIR) -> List[Dict[str, Any]]:
    """Just the entry-signal events from the daemon logs (pair/side/confidence)."""
    return [e for e in load_daemon_logs(directory) if e["kind"] == "signal"]


# ── per-persona conviction reliability (backlog: JUDGE-weight seed) ────────────

def persona_reliability(records: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    """P(persona's vote direction == realized profitable direction) per persona.

    Reads the abbreviated `personas` dict (e.g. {"Fl":"L(55%)"}) against each
    trade's realized profitable side.

    ⚠ COARSE PROXY: this uses trade `win` (did the position net positive after
    the EXIT STACK), which conflates direction quality with exit timing — it is
    NOT the +2h/+4h independent-kline direction accuracy. The authoritative
    per-persona reliability lives in docs/data/groundtruth.json + persona_audit
    .json and is wired in Session F (faithfulness). Use this only as a cheap
    cold-start hint, never to demote a persona on its own (rule #9: audit input).
    """
    hit: Dict[str, int] = {}
    tot: Dict[str, int] = {}
    for r in records:
        if not isinstance(r, dict) or r.get("skipped"):
            continue
        side = str(r.get("side") or "").upper()
        win = r.get("win")
        personas = r.get("personas")
        if side not in ("LONG", "SHORT") or win is None or not isinstance(personas, dict):
            continue
        profitable_dir = side if win else ("SHORT" if side == "LONG" else "LONG")
        prof_code = profitable_dir[0]  # 'L' or 'S'
        for code, val in personas.items():
            name = _PERSONA_CODE.get(code)
            if not name:
                continue
            m = _VOTE_RE.match(str(val))
            if not m or m.group(1) == "N":
                continue
            tot[name] = tot.get(name, 0) + 1
            if m.group(1) == prof_code:
                hit[name] = hit.get(name, 0) + 1
    return {n: hit.get(n, 0) / tot[n] for n in tot if tot[n] > 0}
