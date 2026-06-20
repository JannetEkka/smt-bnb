"""Ground-truth +2h/+4h join + per-persona reliability + forward regime classifier.

This is the reason Session E logged ``entry_price`` + ``ts`` on every exp record
but did NOT grade direction inline: grading the bot with the bot's own logs is
circular. Here we join INDEPENDENT forward klines (Binance public, the same
source as scripts/groundtruth_verify.py) to each record and recover the REAL
+2h/+4h ``direction_correct`` — then feed that to reward.direction_quality_weights
so the gap-9 posterior runs on real accuracy instead of the win-proxy.

Three concerns:
- ``join_forward_returns`` / ``outcomes_with_ground_truth`` — the join itself
  (price fetcher is injected, so tests never touch the network).
- ``persona_reliability_curves`` / ``recommended_weight_adjustments`` — the
  authoritative per-persona conviction-reliability from docs/data/persona_audit
  .json (deferred from Session D). Confirms FLOW, demotes SENTIMENT (anti-
  predictive at high conviction), trusts WHALE only when confident.
- ``ForwardRegimeClassifier`` — the deferred-from-Session-C regime rewrite. The
  old BEARISH label was anti-predictive (66% UP); this learns P(up at +Nh) from
  forward-labelled features instead of asserting a (wrong) directional prior.

Pure-Python; the network fetcher degrades to None and never blocks.
"""

from __future__ import annotations
import json
import logging
import math
import os
import random
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from smt.learning.reward import TradeOutcome
from smt.personas.base import bare_pair

log = logging.getLogger("smt.learning.groundtruth")

PERSONA_AUDIT_JSON = "docs/data/persona_audit.json"
# persona_audit.json abbreviations → full persona names.
_AUDIT_CODE = {"Wh": "whale", "Sn": "sentiment", "Fl": "flow", "Tc": "technical"}
BINANCE_KLINES = "https://data-api.binance.vision/api/v3/klines"


# ── forward-kline join ──────────────────────────────────────────────────────

def direction_correct(side: str, entry_price: float, future_price: float) -> Optional[bool]:
    """Did the position's DIRECTION match the realized move? None if not a bet."""
    s = str(side).upper()
    if entry_price is None or future_price is None or entry_price <= 0:
        return None
    if s == "LONG":
        return future_price > entry_price
    if s == "SHORT":
        return future_price < entry_price
    return None   # WAIT / BLOCK / NEUTRAL — no directional bet to grade


def join_forward_returns(
    records: Sequence[Dict[str, Any]],
    price_fetcher: Callable[[str, str], Optional[Dict[str, float]]],
    horizons: Sequence[str] = ("h2", "h4"),
    primary: str = "h4",
) -> List[Dict[str, Any]]:
    """Attach ``direction_correct_<h>`` + ``ret_<h>`` + canonical ``direction_correct``.

    ``price_fetcher(pair, ts) -> {"h2": px, "h4": px}`` is injected — production
    passes ``binance_forward_price_fetcher``; tests pass a stub. Records missing
    ``entry_price``/``ts`` (or with no fetch) keep ``direction_correct = None``.
    """
    out: List[Dict[str, Any]] = []
    for rec in records:
        r = dict(rec)
        side = r.get("direction") or r.get("side") or ""
        entry = r.get("entry_price")
        ts = r.get("ts") or r.get("timestamp")
        pair = r.get("pair") or r.get("symbol") or ""
        future = price_fetcher(pair, ts) if (entry and ts and pair) else None
        canonical: Optional[bool] = None
        for h in horizons:
            fpx = future.get(h) if future else None
            corr = direction_correct(side, entry, fpx)
            r[f"direction_correct_{h}"] = corr
            if fpx is not None and entry:
                r[f"ret_{h}"] = (float(fpx) - float(entry)) / float(entry)
        # canonical = primary horizon, else first available
        canonical = r.get(f"direction_correct_{primary}")
        if canonical is None:
            for h in horizons:
                if r.get(f"direction_correct_{h}") is not None:
                    canonical = r.get(f"direction_correct_{h}")
                    break
        r["direction_correct"] = canonical
        out.append(r)
    return out


def outcomes_with_ground_truth(joined_records: Sequence[Dict[str, Any]]) -> List[TradeOutcome]:
    """Joined records → TradeOutcome carrying the REAL +Nh ``direction_correct``.

    This is what makes reward.direction_quality_weights a +2h/+4h posterior rather
    than a win-proxy posterior. Only directional records (LONG/SHORT) are emitted.
    """
    outs: List[TradeOutcome] = []
    for r in joined_records:
        side = str(r.get("direction") or r.get("side") or "").upper()
        if side not in ("LONG", "SHORT"):
            continue
        pnl = r.get("pnl_usd")
        outs.append(TradeOutcome(
            pair=str(r.get("pair") or r.get("symbol") or ""),
            direction=side,
            regime=str(r.get("regime", "NORMAL")),
            lane=str(r.get("lane") or "fast"),
            net_pnl_usd=float(pnl) if pnl is not None else 0.0,
            fees_usd=0.0,
            direction_correct=r.get("direction_correct"),
            conviction=float(r.get("conviction", 0.0) or 0.0),
            fng=r.get("fear_greed"),
        ))
    return outs


def _ts_to_ms(ts: str) -> Optional[int]:
    import datetime as dt
    try:
        return int(dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp() * 1000)
    except Exception:
        return None


def binance_forward_price_fetcher(
    horizons_hours: Dict[str, int] = None,
    timeout: int = 15,
) -> Callable[[str, str], Optional[Dict[str, float]]]:
    """Build a fetcher that pulls forward 1h klines from Binance public data.

    Independent of the bot's own logs (the whole point). Returns a closure
    ``(pair, ts) -> {"h2": close, "h4": close}`` that degrades to None on any
    network/parse failure — NEVER blocks the join.
    """
    horizons_hours = horizons_hours or {"h2": 2, "h4": 4}

    def _fetch(pair: str, ts: str) -> Optional[Dict[str, float]]:
        sym = bare_pair(pair) + "USDT"
        ms = _ts_to_ms(ts)
        if ms is None:
            return None
        end = ms + (max(horizons_hours.values()) + 1) * 3600_000
        url = (f"{BINANCE_KLINES}?symbol={sym}&interval=1h"
               f"&startTime={ms}&endTime={end}&limit=12")
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                kl = json.loads(resp.read())
        except Exception as exc:  # noqa: BLE001 — degrade, don't block
            log.warning("[GROUNDTRUTH] forward-kline fetch failed for %s: %s", sym, exc)
            return None
        if not kl:
            return None
        out: Dict[str, float] = {}
        for name, h in horizons_hours.items():
            if h < len(kl):
                out[name] = float(kl[h][4])     # close of the +h hour candle
        return out or None

    return _fetch


# ── per-persona conviction-reliability (authoritative, from persona_audit.json) ─

@dataclass
class ReliabilityCurve:
    persona: str
    n: int
    acc_h2: float
    acc_h4: float
    hc_n: int
    hc_acc_h4: float
    monotonic: bool             # high-conviction MORE accurate than overall


def load_persona_audit(path: str = PERSONA_AUDIT_JSON) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        log.warning("[GROUNDTRUTH] could not read %s: %s", path, exc)
        return {}


def persona_reliability_curves(audit: Optional[Dict[str, Any]] = None,
                               path: str = PERSONA_AUDIT_JSON) -> Dict[str, ReliabilityCurve]:
    """Per-persona +2h/+4h accuracy + high-conviction accuracy from the audit.

    Reads the ``overall`` block (n, c2, c4, hc_n, hc_c4). This is the
    AUTHORITATIVE per-persona reliability — NOT the corpus win-proxy.
    """
    audit = audit if audit is not None else load_persona_audit(path)
    overall = (audit or {}).get("overall", {})
    out: Dict[str, ReliabilityCurve] = {}
    for code, stats in overall.items():
        name = _AUDIT_CODE.get(code, code.lower())
        n = int(stats.get("n", 0) or 0)
        c2 = int(stats.get("c2", 0) or 0)
        c4 = int(stats.get("c4", 0) or 0)
        hc_n = int(stats.get("hc_n", 0) or 0)
        hc_c4 = int(stats.get("hc_c4", 0) or 0)
        acc_h2 = c2 / n if n else 0.0
        acc_h4 = c4 / n if n else 0.0
        hc_acc_h4 = hc_c4 / hc_n if hc_n else 0.0
        out[name] = ReliabilityCurve(
            persona=name, n=n, acc_h2=acc_h2, acc_h4=acc_h4,
            hc_n=hc_n, hc_acc_h4=hc_acc_h4,
            monotonic=(hc_n > 0 and hc_acc_h4 >= acc_h4),
        )
    return out


def recommended_weight_adjustments(
    curves: Optional[Dict[str, ReliabilityCurve]] = None,
    breakeven: float = 0.50,
    confirm_acc: float = 0.60,
    monotonic_gap: float = 0.05,
) -> Dict[str, str]:
    """Map reliability curves → a weight verdict (feeds the Session-H weight ship).

    DEMOTE                — anti-predictive at high conviction (hc_acc < breakeven).
    TRUST_HIGH_CONV_ONLY  — clearly better when confident (hc_acc − acc ≥ gap).
    CONFIRM               — reliable overall (acc_h4 ≥ confirm_acc).
    LEAVE                 — borderline; no change warranted.
    """
    curves = curves if curves is not None else persona_reliability_curves()
    verdict: Dict[str, str] = {}
    for name, c in curves.items():
        if c.hc_n > 0 and c.hc_acc_h4 < breakeven:
            verdict[name] = "DEMOTE"
        elif c.hc_n > 0 and (c.hc_acc_h4 - c.acc_h4) >= monotonic_gap:
            verdict[name] = "TRUST_HIGH_CONV_ONLY"
        elif c.acc_h4 >= confirm_acc:
            verdict[name] = "CONFIRM"
        else:
            verdict[name] = "LEAVE"
    return verdict


# ── forward-looking regime classifier (deferred-from-Session-C rewrite) ─────────

@dataclass
class ForwardRegimeClassifier:
    """Tiny pure-Python logistic regression: features → P(up at +Nh).

    The Session-C audit found the old BEARISH→SHORT label was ANTI-predictive
    (66% of BEARISH-labelled cells went UP at +Nh). Rather than encode a wrong
    directional prior, we LEARN the forward probability from forward-labelled
    features (each row's label is the realized +Nh up/down). Standardizes inputs
    for stable gradient descent.
    """
    lr: float = 0.3
    epochs: int = 500
    l2: float = 1e-3
    seed: int = 0
    weights: List[float] = field(default_factory=list)
    bias: float = 0.0
    _mu: List[float] = field(default_factory=list)
    _sd: List[float] = field(default_factory=list)

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def _standardize(self, x: Sequence[float]) -> List[float]:
        return [(x[i] - self._mu[i]) / self._sd[i] for i in range(len(x))]

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> "ForwardRegimeClassifier":
        X = [list(map(float, row)) for row in X]
        y = [int(v) for v in y]
        n, d = len(X), len(X[0]) if X else 0
        if n == 0 or d == 0:
            return self
        # feature standardization
        self._mu = [sum(row[j] for row in X) / n for j in range(d)]
        self._sd = []
        for j in range(d):
            var = sum((row[j] - self._mu[j]) ** 2 for row in X) / n
            self._sd.append(math.sqrt(var) if var > 1e-12 else 1.0)
        Xs = [self._standardize(row) for row in X]
        rng = random.Random(self.seed)
        self.weights = [rng.uniform(-0.01, 0.01) for _ in range(d)]
        self.bias = 0.0
        for _ in range(self.epochs):
            gw = [0.0] * d
            gb = 0.0
            for i in range(n):
                z = self.bias + sum(self.weights[j] * Xs[i][j] for j in range(d))
                err = self._sigmoid(z) - y[i]
                for j in range(d):
                    gw[j] += err * Xs[i][j]
                gb += err
            for j in range(d):
                self.weights[j] -= self.lr * (gw[j] / n + self.l2 * self.weights[j])
            self.bias -= self.lr * (gb / n)
        return self

    def predict_proba(self, x: Sequence[float]) -> float:
        if not self.weights:
            return 0.5
        xs = self._standardize(list(map(float, x)))
        z = self.bias + sum(self.weights[j] * xs[j] for j in range(len(xs)))
        return self._sigmoid(z)

    def predict_direction(self, x: Sequence[float]) -> str:
        return "LONG" if self.predict_proba(x) >= 0.5 else "SHORT"

    def accuracy(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> float:
        if not X:
            return 0.0
        hits = sum(1 for i in range(len(X)) if int(self.predict_proba(X[i]) >= 0.5) == int(y[i]))
        return hits / len(X)
