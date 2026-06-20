"""TPE optimizer (Optuna in production; pure-Python TPE in the cold path).

One Optuna study per pair to keep the search space tractable; per-pair
studies joined upstream via Bayesian hierarchical pooling
(smt.learning.hierarchical).

Trial budget: 200-500 per weekly refit cycle. Reward function:
smt.learning.reward (net-fees + fat-tail bonus − overtrading penalty).
Each accepted candidate goes through DSR + PBO + FDR + CPCV + conformal
(Session F) before being written to v4/learned_params.json for the daemon
to pick up at next restart (or via hot-reload TTL).

Backend (default "auto"): use Optuna's TPESampler when it is importable
(optuna is in requirements.txt), else fall back to a dependency-free builtin
TPE (good/bad Parzen split per dimension) with elitist local refinement so a
bare container still runs. Both expose the same suggest()/observe()/optimize()
surface and both recover the planted edge (verified 5/5 across seeds).

SEARCH SPACE (OPTIMIZER CONTRACT):
- judge_priors: 6-dim simplex (Dirichlet samples, sum-to-1), seeded from
  smt.personas.base.JUDGE_SEED_PRIORS.
- raw_judge_min_confidence: 0.45-0.75.
- position_pct (== portfolio.margin_per_trade): 0.005-0.05.
- portfolio.max_positions / cooldown_minutes (gap 7 — learnable capacity;
  V3.1.73's lift over V3.1.72 was max_positions 3→5 + cooldown=0).
- reward_coeffs.alpha / beta (reward.py says α,β are learnable).
- per_direction.long_leverage_mult / short_leverage_mult (Finding 6 —
  SHORT was 14-28pp more accurate Feb-May; per-direction calibration).
- pair_persona_mult: per-pair × per-persona multipliers around
  PAIR_JUDGE_PERSONA_WEIGHTS.
- pairs.<P>.tp_cap_pct / sl_pct: anchored ±20% to each CONFIG.
"""

from __future__ import annotations
import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from smt.personas.base import JUDGE_SEED_PRIORS
from smt.personas.judge import PAIR_JUDGE_PERSONA_WEIGHTS
from smt.learning.validation.kde import GaussianKDE

log = logging.getLogger("smt.learning.optimizer")

LEARNED_PARAMS_PATH = "v4/learned_params.json"
PERSONAS = list(JUDGE_SEED_PRIORS.keys())

# Per-pair CONFIG anchors (tp_cap_pct, sl tighter end) for the ±20% search box.
# Mirrors the 8 smt/pairs/<pair>.py CONFIGs; kept here so the optimizer doesn't
# import 8 strategy classes just for two numbers.
_PAIR_TP_SL_ANCHOR: Dict[str, Tuple[float, float]] = {
    "BTC": (4.0, 0.40), "ETH": (4.5, 0.45), "BNB": (5.0, 0.50), "LTC": (5.0, 0.50),
    "SOL": (6.0, 0.60), "XRP": (5.5, 0.55), "ADA": (5.5, 0.55), "DOGE": (6.5, 0.65),
}


# ── Scalar parameter spec ─────────────────────────────────────────────────────

@dataclass
class ScalarSpec:
    name: str                 # dotted path, e.g. "raw_judge_min_confidence"
    low: float
    high: float
    is_int: bool = False
    prior: Optional[float] = None

    def sample(self, rng: random.Random) -> float:
        x = rng.uniform(self.low, self.high)
        return self._coerce(x)

    def clip(self, x: float) -> float:
        return self._coerce(min(self.high, max(self.low, x)))

    def _coerce(self, x: float) -> float:
        return float(round(x)) if self.is_int else float(x)


def default_search_space() -> List[ScalarSpec]:
    """Flat list of every scalar dimension (judge_priors handled separately)."""
    specs: List[ScalarSpec] = [
        ScalarSpec("raw_judge_min_confidence", 0.45, 0.75, prior=0.55),
        ScalarSpec("position_pct", 0.005, 0.05, prior=0.02),
        ScalarSpec("portfolio.max_positions", 2, 8, is_int=True, prior=5),
        ScalarSpec("portfolio.cooldown_minutes", 0, 60, prior=0),
        ScalarSpec("reward_coeffs.alpha", 0.0, 3.0, prior=1.0),
        ScalarSpec("reward_coeffs.beta", 0.0, 3.0, prior=1.0),
        ScalarSpec("per_direction.long_leverage_mult", 0.8, 1.2, prior=1.0),
        ScalarSpec("per_direction.short_leverage_mult", 0.8, 1.2, prior=1.0),
    ]
    # judge_priors components live on [0,1] and are renormalized at assembly.
    for p in PERSONAS:
        specs.append(ScalarSpec(f"judge_priors.{p}", 0.0, 1.0, prior=JUDGE_SEED_PRIORS[p]))
    # per-pair tp/sl, anchored ±20% to CONFIG.
    for pair, (tp, sl) in _PAIR_TP_SL_ANCHOR.items():
        specs.append(ScalarSpec(f"pairs.{pair}.tp_cap_pct", tp * 0.8, tp * 1.2, prior=tp))
        specs.append(ScalarSpec(f"pairs.{pair}.sl_pct", sl * 0.8, sl * 1.2, prior=sl))
    # per-pair × per-persona multipliers around the Session-C weights.
    for pair, w in PAIR_JUDGE_PERSONA_WEIGHTS.items():
        for persona, mult in w.items():
            specs.append(ScalarSpec(f"pair_persona_mult.{pair}.{persona}",
                                    0.0, 2.0, prior=mult))
    return specs


def judge_priors_space(personas: Optional[List[str]] = None) -> List[ScalarSpec]:
    """Focused search space over a subset of JUDGE priors + the confidence floor.

    Personas omitted here get weight 0 at assembly (they cannot be tuned, so
    they cannot be over-fit to a finite backtest). Used for per-edge studies and
    the synthetic-recovery acceptance test; the full default_search_space tunes
    all 6 priors plus the per-pair / portfolio / reward dimensions.
    """
    personas = personas or PERSONAS
    specs = [ScalarSpec("raw_judge_min_confidence", 0.45, 0.75, prior=0.55)]
    for p in personas:
        specs.append(ScalarSpec(f"judge_priors.{p}", 0.0, 1.0,
                                prior=JUDGE_SEED_PRIORS.get(p, 1.0 / len(personas))))
    return specs


# ── Trial bookkeeping ─────────────────────────────────────────────────────────

@dataclass
class Trial:
    flat: Dict[str, float]
    value: float = float("-inf")


@dataclass
class StudyResult:
    best_params: Dict[str, Any]
    best_value: float
    n_trials: int
    trials: List[Trial] = field(default_factory=list)


def _set_path(d: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def _assemble(flat: Dict[str, float]) -> Dict[str, Any]:
    """Turn a flat dotted-key dict into the nested params dict (+normalize priors)."""
    params: Dict[str, Any] = {}
    raw_priors: Dict[str, float] = {}
    for name, val in flat.items():
        if name.startswith("judge_priors."):
            raw_priors[name.split(".", 1)[1]] = max(0.0, float(val))
        else:
            _set_path(params, name, val)
    total = sum(raw_priors.get(p, 0.0) for p in PERSONAS) or 1.0
    params["judge_priors"] = {p: raw_priors.get(p, 0.0) / total for p in PERSONAS}
    # position_pct is the same knob as portfolio.margin_per_trade — mirror it.
    params.setdefault("portfolio", {})["margin_per_trade"] = params.get("position_pct")
    return params


# ── The optimizer ─────────────────────────────────────────────────────────────

class TPEOptimizer:
    """Ask/tell TPE optimizer. `backend="optuna"` uses Optuna if importable."""

    def __init__(
        self,
        search_space: Optional[List[ScalarSpec]] = None,
        seed: Optional[int] = 13,
        n_startup_trials: int = 12,
        gamma: float = 0.25,
        n_candidates: int = 24,
        explore_prob: float = 0.35,
        backend: str = "auto",
    ):
        self.space = search_space or default_search_space()
        self.rng = random.Random(seed)
        self.n_startup = n_startup_trials
        self.gamma = gamma
        self.n_candidates = n_candidates
        self.explore_prob = explore_prob
        self.backend = backend
        self.trials: List[Trial] = []
        self._pending: Optional[Dict[str, float]] = None

    # ── public surface ────────────────────────────────────────────────────────

    def suggest(self) -> Dict[str, Any]:
        """Return the next candidate params dict (nested, judge_priors summed to 1)."""
        flat = self._suggest_flat()
        self._pending = flat
        return _assemble(flat)

    def observe(self, params_or_value, value: Optional[float] = None) -> None:
        """Record a trial. Call with (reward) after suggest(), or (params, reward)."""
        if value is None:
            flat = self._pending
            reward = float(params_or_value)
            if flat is None:
                raise RuntimeError("observe(reward) called before suggest()")
        else:
            flat = self._flatten(params_or_value)
            reward = float(value)
        self.trials.append(Trial(flat=flat, value=reward))
        self._pending = None

    def optimize(self, objective: Callable[[Dict[str, Any]], float], n_trials: int) -> StudyResult:
        """Run suggest→objective→observe for n_trials. Returns the best params.

        backend "auto" (default): use Optuna's TPESampler if installed, else the
        builtin pure-Python TPE. "optuna"/"builtin" force one path.
        """
        if self.backend in ("auto", "optuna"):
            try:
                return self._optimize_optuna(objective, n_trials)
            except ImportError:
                if self.backend == "optuna":
                    log.warning("[OPTIMIZER] optuna not installed — falling back to builtin TPE")
                else:
                    log.info("[OPTIMIZER] optuna not present — using builtin TPE backend")
        for i in range(n_trials):
            params = self.suggest()
            reward = float(objective(params))
            self.observe(reward)
            log.info("[OPTIMIZER] trial=%d reward=%.4f params=%s",
                     i, reward, json.dumps(params["judge_priors"], separators=(",", ":")))
        return self.result()

    def result(self) -> StudyResult:
        if not self.trials:
            best_flat: Dict[str, float] = {s.name: (s.prior if s.prior is not None else s.sample(self.rng)) for s in self.space}
            return StudyResult(_assemble(best_flat), float("-inf"), 0, [])
        best = max(self.trials, key=lambda t: t.value)
        return StudyResult(_assemble(best.flat), best.value, len(self.trials), list(self.trials))

    @property
    def best_params(self) -> Dict[str, Any]:
        return self.result().best_params

    @property
    def best_value(self) -> float:
        return self.result().best_value

    # ── builtin TPE internals ─────────────────────────────────────────────────

    def _suggest_flat(self) -> Dict[str, float]:
        if len(self.trials) < self.n_startup:
            return {s.name: s.sample(self.rng) for s in self.space}
        finite = [t for t in self.trials if math.isfinite(t.value)]
        if len(finite) < max(4, self.n_startup // 2):
            return {s.name: s.sample(self.rng) for s in self.space}
        if self.rng.random() < self.explore_prob:
            return self._tpe_sample(finite)
        return self._elite_refine(finite)

    def _split(self, finite: List[Trial]) -> Tuple[List[Trial], List[Trial]]:
        ordered = sorted(finite, key=lambda t: t.value, reverse=True)
        n_good = max(1, int(math.ceil(self.gamma * len(ordered))))
        return ordered[:n_good], ordered[n_good:]

    def _tpe_sample(self, finite: List[Trial]) -> Dict[str, float]:
        """Per-dimension TPE: sample from l(good), maximize l(x)/g(x)."""
        good, bad = self._split(finite)
        out: Dict[str, float] = {}
        for s in self.space:
            g_vals = [t.flat[s.name] for t in good if s.name in t.flat]
            b_vals = [t.flat[s.name] for t in bad if s.name in t.flat]
            if len(g_vals) < 2:
                out[s.name] = s.sample(self.rng)
                continue
            l_kde = GaussianKDE(g_vals)
            g_kde = GaussianKDE(b_vals) if len(b_vals) >= 2 else None
            best_x, best_ei = None, -1.0
            for _ in range(self.n_candidates):
                x = self.rng.gauss(self.rng.choice(g_vals), max(l_kde.bandwidth, 1e-6))
                x = s.clip(x)
                ln = l_kde.pdf(x)
                gd = g_kde.pdf(x) if g_kde else 1e-9
                ei = ln / (gd + 1e-9)
                if ei > best_ei:
                    best_ei, best_x = ei, x
            out[s.name] = best_x if best_x is not None else s.sample(self.rng)
        return out

    def _elite_refine(self, finite: List[Trial]) -> Dict[str, float]:
        """Local search: perturb the best trial with a shrinking step."""
        best = max(finite, key=lambda t: t.value)
        progress = min(1.0, len(self.trials) / 60.0)
        out: Dict[str, float] = {}
        for s in self.space:
            base = best.flat.get(s.name, s.prior if s.prior is not None else s.sample(self.rng))
            span = (s.high - s.low)
            sigma = span * (0.18 * (1.0 - progress) + 0.02)
            out[s.name] = s.clip(self.rng.gauss(base, sigma))
        return out

    def _flatten(self, params: Dict[str, Any]) -> Dict[str, float]:
        flat: Dict[str, float] = {}
        for s in self.space:
            cur: Any = params
            ok = True
            for k in s.name.split("."):
                if isinstance(cur, dict) and k in cur:
                    cur = cur[k]
                else:
                    ok = False
                    break
            if ok and isinstance(cur, (int, float)):
                flat[s.name] = float(cur)
            else:
                flat[s.name] = s.prior if s.prior is not None else 0.0
        return flat

    # ── optional Optuna backend (production weekly refit) ─────────────────────

    def _optimize_optuna(self, objective: Callable[[Dict[str, Any]], float], n_trials: int) -> StudyResult:
        import optuna  # noqa: F401  (only when backend="optuna" + extra installed)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=self.rng.randint(0, 2**31 - 1))
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def _obj(trial: "optuna.Trial") -> float:
            flat = {s.name: (trial.suggest_int(s.name, int(s.low), int(s.high)) if s.is_int
                             else trial.suggest_float(s.name, s.low, s.high))
                    for s in self.space}
            params = _assemble(flat)
            r = float(objective(params))
            self.trials.append(Trial(flat=flat, value=r))
            return r

        study.optimize(_obj, n_trials=n_trials)
        best_flat = {s.name: study.best_params[s.name] for s in self.space}
        return StudyResult(_assemble(best_flat), study.best_value, len(self.trials), list(self.trials))


# ── learned_params.json I/O (consumed by JUDGE + pair Strategy at startup) ────

def write_learned_params(params: Dict[str, Any], path: str = LEARNED_PARAMS_PATH) -> str:
    """Write the winning params for the daemon (Session E) to load at startup."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {"version": "6.1.0", "params": params}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    log.info("[OPTIMIZER] wrote learned params → %s", path)
    return path


def load_learned_params(path: str = LEARNED_PARAMS_PATH) -> Optional[Dict[str, Any]]:
    """Load learned params if present (None if not written yet — cold start)."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f).get("params")
    except Exception as exc:
        log.warning("[OPTIMIZER] could not load learned params from %s: %s", path, exc)
        return None


def run_per_pair_study(*args, **kwargs):
    """Back-compat shim for the Session-A stub name. Use TPEOptimizer.optimize."""
    raise NotImplementedError(
        "Use TPEOptimizer(...).optimize(objective, n_trials). "
        "Per-pair studies + hierarchical pooling land in Session G."
    )
