"""Flow-drop exit stack — the PnL converter (Synthesis Finding 4).

One module, not scattered inline checks (PERSONA_GATE_AUDIT bucket 2).
Evaluates all exit branches in priority order; returns first triggered
ExitDecision. Consecutive-check counters live in the position dict
so they persist across cycles via PositionTracker state.

NOTE (addendum #3 — F&G calibration): Any future F&G-gated threshold
ported from archive must add +7pts (alt.me was ~7pts low vs CMC scale).
Current module has no F&G gates — those live in persona layer (Session C).

Port reference: archive/v6.0/v4/smt_daemon_v3_1.py monitor_positions()
lines 11680-12010; config dicts lines 990-1130 (V3.2.261-V6.0.7).
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from smt.core.trade_plan import ExitDecision

log = logging.getLogger("smt.exit_cascade")

# ── Per-pair partial-close trigger (% PnL, base). Floor: 0.25 ≥ fees. ──────
# V3.2.177: ATR-adaptive; these are the pair bases before lane scaling.
PARTIAL_CLOSE_TRIGGER: Dict[str, float] = {
    "BTC": 0.30, "ETH": 0.35, "LTC": 0.30, "BNB": 0.35,
    "SOL": 0.45, "XRP": 0.35, "ADA": 0.35, "DOGE": 0.40,
}

# ── Per-lane × per-pair profit_protect: (min_peak_pct, fade_frac, fee_floor_pct) ──
# V3.2.262: lane dimension for base (non-persistent-trend) case.
PROFIT_PROTECT_BASE: Dict[str, Dict[str, tuple]] = {
    "slow": {
        "BTC":  (0.20, 0.50, 0.12), "ETH":  (0.22, 0.50, 0.12),
        "BNB":  (0.25, 0.55, 0.14), "LTC":  (0.25, 0.55, 0.14),
        "SOL":  (0.30, 0.55, 0.16), "XRP":  (0.25, 0.55, 0.14),
        "ADA":  (0.30, 0.60, 0.16), "DOGE": (0.35, 0.60, 0.18),
    },
    "fast": {
        "BTC":  (0.18, 0.55, 0.12), "ETH":  (0.20, 0.55, 0.12),
        "BNB":  (0.22, 0.60, 0.14), "LTC":  (0.22, 0.60, 0.14),
        "SOL":  (0.25, 0.60, 0.16), "XRP":  (0.22, 0.60, 0.14),
        "ADA":  (0.25, 0.65, 0.16), "DOGE": (0.30, 0.65, 0.18),
    },
    "bigwick": {  # mean-reversion: lock peak aggressively
        "BTC":  (0.15, 0.60, 0.12), "ETH":  (0.18, 0.60, 0.12),
        "BNB":  (0.20, 0.65, 0.14), "LTC":  (0.20, 0.65, 0.14),
        "SOL":  (0.22, 0.65, 0.16), "XRP":  (0.20, 0.65, 0.14),
        "ADA":  (0.22, 0.70, 0.16), "DOGE": (0.25, 0.70, 0.18),
    },
}

# ── Per-pair × per-regime profit_protect fade override (V6.0.7) ─────────────
PROFIT_PROTECT_BY_REGIME: Dict[str, Dict[str, float]] = {
    "BTC":  {"TRENDING_UP": 0.70, "TRENDING_DOWN": 0.70, "RANGING": 0.50, "CRASH": 0.40, "RECOVERY": 0.55, "NORMAL": 0.50},
    "ETH":  {"TRENDING_UP": 0.68, "TRENDING_DOWN": 0.68, "RANGING": 0.50, "CRASH": 0.42, "RECOVERY": 0.55, "NORMAL": 0.50},
    "BNB":  {"TRENDING_UP": 0.62, "TRENDING_DOWN": 0.62, "RANGING": 0.52, "CRASH": 0.45, "RECOVERY": 0.55, "NORMAL": 0.52},
    "LTC":  {"TRENDING_UP": 0.62, "TRENDING_DOWN": 0.62, "RANGING": 0.52, "CRASH": 0.45, "RECOVERY": 0.55, "NORMAL": 0.52},
    "SOL":  {"TRENDING_UP": 0.65, "TRENDING_DOWN": 0.65, "RANGING": 0.55, "CRASH": 0.48, "RECOVERY": 0.58, "NORMAL": 0.55},
    "XRP":  {"TRENDING_UP": 0.62, "TRENDING_DOWN": 0.62, "RANGING": 0.55, "CRASH": 0.45, "RECOVERY": 0.55, "NORMAL": 0.55},
    "ADA":  {"TRENDING_UP": 0.62, "TRENDING_DOWN": 0.62, "RANGING": 0.55, "CRASH": 0.45, "RECOVERY": 0.55, "NORMAL": 0.55},
    "DOGE": {"TRENDING_UP": 0.65, "TRENDING_DOWN": 0.65, "RANGING": 0.55, "CRASH": 0.48, "RECOVERY": 0.58, "NORMAL": 0.55},
}

# ── Chandelier ATR multiplier by regime (V6.0.7): trail = peak − mult × ATR ─
CHANDELIER_ATR_MULT: Dict[str, Dict[str, float]] = {
    "BTC":  {"TRENDING_UP": 1.0, "TRENDING_DOWN": 1.0, "RANGING": 1.5, "CRASH": 1.2, "RECOVERY": 1.2, "NORMAL": 1.2},
    "ETH":  {"TRENDING_UP": 1.2, "TRENDING_DOWN": 1.2, "RANGING": 1.5, "CRASH": 1.3, "RECOVERY": 1.3, "NORMAL": 1.3},
    "BNB":  {"TRENDING_UP": 1.5, "TRENDING_DOWN": 1.5, "RANGING": 1.7, "CRASH": 1.5, "RECOVERY": 1.5, "NORMAL": 1.5},
    "LTC":  {"TRENDING_UP": 1.5, "TRENDING_DOWN": 1.5, "RANGING": 1.5, "CRASH": 1.3, "RECOVERY": 1.3, "NORMAL": 1.4},
    "SOL":  {"TRENDING_UP": 2.0, "TRENDING_DOWN": 2.0, "RANGING": 1.8, "CRASH": 1.5, "RECOVERY": 1.5, "NORMAL": 1.7},
    "XRP":  {"TRENDING_UP": 1.5, "TRENDING_DOWN": 1.5, "RANGING": 1.5, "CRASH": 1.3, "RECOVERY": 1.3, "NORMAL": 1.4},
    "ADA":  {"TRENDING_UP": 1.5, "TRENDING_DOWN": 1.5, "RANGING": 1.5, "CRASH": 1.3, "RECOVERY": 1.3, "NORMAL": 1.4},
    "DOGE": {"TRENDING_UP": 2.0, "TRENDING_DOWN": 2.0, "RANGING": 1.8, "CRASH": 1.5, "RECOVERY": 1.5, "NORMAL": 1.7},
}

# ── Min hold before exit-cascade gates can fire (minutes by lane) ────────────
_MIN_HOLD_MIN = {"fast": 5, "bigwick": 15, "slow": 10}
# ── Flow-drop consecutive checks before exit fires (by lane) ─────────────────
_FLOW_DROP_CONSEC = {"fast": 3, "bigwick": 3, "slow": 5}
# ── Never-profitable: min age before gate can fire (minutes by lane) ─────────
_NEVER_PROFIT_MIN_AGE = {"fast": 30, "bigwick": 30, "slow": 60}
# ── Dynamic max-hold extension when winning (hours by lane) ──────────────────
_MAX_HOLD_EXTEND = {"fast": 0.5, "bigwick": 0.5, "slow": 1.0}
# ── Default max-hold fallback (hours) when not stored in position ─────────────
_MAX_HOLD_DEFAULT = {"fast": 0.5, "bigwick": 2.5, "slow": 4.0}


def _age_minutes(position: Dict[str, Any]) -> float:
    opened_at = position.get("opened_at")
    if not opened_at:
        return 0.0
    try:
        dt = datetime.fromisoformat(str(opened_at).replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt).total_seconds() / 60.0
    except Exception:
        return 0.0


def _atr_from_klines(klines: Dict[str, Any], symbol: str, period: int = 14) -> float:
    """14-period ATR as % of close. Returns 0.0 if klines unavailable."""
    for key in (f"{symbol}#1h", f"{symbol}_1h", "1h"):
        candles = klines.get(key) or []
        if len(candles) >= period + 1:
            break
    else:
        return 0.0
    trs = []
    for i in range(1, min(len(candles), period + 1)):
        try:
            h, l, pc = float(candles[i][2]), float(candles[i][3]), float(candles[i - 1][4])
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        except (IndexError, ValueError):
            pass
    if not trs:
        return 0.0
    close = float(candles[-1][4]) if candles else 0.0
    return (sum(trs) / len(trs) / close * 100.0) if close > 0 else 0.0


class ExitCascade:
    """Stateless evaluator — all state lives in the position dict.

    Returns the first triggered ExitDecision in priority order, or None.
    Special case: ExitDecision(should_exit=False, reason="PARTIAL_CLOSE …")
    signals the daemon (Session E) to close 50% and move SL to entry.
    """

    def evaluate(
        self,
        position: Dict[str, Any],
        context: Dict[str, Any],
        pair_name: str,
    ) -> Optional[ExitDecision]:
        side = position.get("side", "LONG")
        lane = position.get("entry_lane", "slow")
        entry_price = float(position.get("entry_price") or 0.0)

        prices = context.get("prices") or {}
        symbol = f"{pair_name}USDT"
        current_price = float(prices.get(symbol) or prices.get(pair_name) or 0.0)
        if current_price <= 0 or entry_price <= 0:
            return None  # HTTP 429 sentinel or no market data → skip cycle

        if side == "LONG":
            pnl_pct = (current_price - entry_price) / entry_price * 100.0
        else:
            pnl_pct = (entry_price - current_price) / entry_price * 100.0

        # Update peak PnL in-place (persisted by tracker)
        peak_pnl = float(position.get("peak_pnl_pct") or 0.0)
        if pnl_pct > peak_pnl:
            position["peak_pnl_pct"] = pnl_pct
            peak_pnl = pnl_pct

        age_min = _age_minutes(position)
        min_hold = _MIN_HOLD_MIN.get(lane, 10)

        regime_raw = context.get("regime") or {}
        pair_regime = (regime_raw.get(symbol) or regime_raw.get(pair_name)
                       if isinstance(regime_raw, dict) else str(regime_raw)) or "NORMAL"

        atr_pct = _atr_from_klines(context.get("klines") or {}, symbol) or 0.50
        partial_done = bool(position.get("partial_close_done"))
        max_hold_h = float(position.get("max_hold_h") or _MAX_HOLD_DEFAULT.get(lane, 4.0))

        flow = context.get("flow_signal") or {}
        flow_dir = flow.get("direction", "NEUTRAL") or "NEUTRAL"
        flow_conf = float(flow.get("confidence") or 0.0)
        flow_present = flow_dir not in ("NEUTRAL", "", None)

        # ── 1. FLOW_AGAINST_EXIT ─────────────────────────────────────────────
        against = flow_present and flow_dir != side and flow_conf >= 0.85
        position["_against_consec"] = (position.get("_against_consec", 0) + 1) if against else 0
        if position["_against_consec"] >= 2 and age_min >= min_hold:
            log.info("[EXIT] %s FLOW_AGAINST side=%s flow=%s %.0f%% consec=%d",
                     pair_name, side, flow_dir, flow_conf * 100, position["_against_consec"])
            return ExitDecision(
                should_exit=True,
                reason=f"FLOW_AGAINST (dir={flow_dir} conf={flow_conf:.0%} ×{position['_against_consec']})",
            )

        # ── 2. FLOW_DROP_EXIT ────────────────────────────────────────────────
        is_weak = flow_present and (flow_conf < 0.50 or flow_dir != side)
        if flow_present:
            position["_drop_consec"] = (position.get("_drop_consec", 0) + 1) if is_weak else 0
        drop_needed = _FLOW_DROP_CONSEC.get(lane, 5)
        if (position.get("_drop_consec", 0) >= drop_needed
                and age_min >= min_hold
                and pair_regime not in ("TRENDING_UP", "TRENDING_DOWN")):
            log.info("[EXIT] %s FLOW_DROP lane=%s drop_consec=%d flow=%.0f%%",
                     pair_name, lane, position["_drop_consec"], flow_conf * 100)
            return ExitDecision(
                should_exit=True,
                reason=f"FLOW_DROP (flow={flow_conf:.0%} consec={position['_drop_consec']})",
            )

        # ── 3. PARTIAL_CLOSE signal (should_exit=False; daemon handles partial) ─
        if not partial_done:
            pc_base = PARTIAL_CLOSE_TRIGGER.get(pair_name, 0.40)
            lane_pc_mult = {"fast": 0.75, "bigwick": 0.75, "slow": 1.0}.get(lane, 1.0)
            pc_trigger = max(0.25, pc_base * lane_pc_mult)
            if pnl_pct >= pc_trigger:
                log.info("[EXIT] %s PARTIAL_CLOSE trigger=%.2f%% pnl=%.2f%%",
                         pair_name, pc_trigger, pnl_pct)
                return ExitDecision(
                    should_exit=False,
                    reason=f"PARTIAL_CLOSE (pnl={pnl_pct:.2f}%≥{pc_trigger:.2f}%)",
                )

        # ── 4. PROFIT_PROTECT (peak-fade) ────────────────────────────────────
        pp_lane = lane if lane in PROFIT_PROTECT_BASE else "slow"
        min_peak, fade_frac, fee_floor = PROFIT_PROTECT_BASE[pp_lane].get(pair_name, (0.25, 0.55, 0.14))
        fade_frac = PROFIT_PROTECT_BY_REGIME.get(pair_name, {}).get(pair_regime, fade_frac)
        fade_at = peak_pnl * fade_frac
        if peak_pnl >= min_peak and pnl_pct < fade_at and pnl_pct >= fee_floor and age_min >= 5:
            log.info("[EXIT] %s PROFIT_PROTECT peak=%.2f%% fade=%.0f%% @ %.2f%% pnl=%.2f%%",
                     pair_name, peak_pnl, fade_frac * 100, fade_at, pnl_pct)
            return ExitDecision(
                should_exit=True,
                reason=f"PROFIT_PROTECT (peak={peak_pnl:.2f}% fade={fade_at:.2f}%)",
            )

        # ── 5. CHANDELIER TRAIL (only after partial close) ───────────────────
        if partial_done and peak_pnl >= 0.50:
            ch_mult = CHANDELIER_ATR_MULT.get(pair_name, {}).get(pair_regime, 1.5)
            trail_floor = max(0.10, peak_pnl - ch_mult * atr_pct)
            if pnl_pct < trail_floor:
                log.info("[EXIT] %s CHANDELIER trail=%.2f%% peak=%.2f%% atr=%.2f%%×%.1f",
                         pair_name, trail_floor, peak_pnl, atr_pct, ch_mult)
                return ExitDecision(
                    should_exit=True,
                    reason=f"CHANDELIER_TRAIL (trail={trail_floor:.2f}%)",
                )

        # ── 6. NEVER_PROFITABLE_EXIT ─────────────────────────────────────────
        np_min_age = _NEVER_PROFIT_MIN_AGE.get(lane, 60)
        if age_min >= np_min_age and peak_pnl <= 0.08:
            np_flow_ok = flow_present and flow_dir == side and flow_conf >= 0.70
            if flow_present:
                position["_np_consec"] = 0 if np_flow_ok else position.get("_np_consec", 0) + 1
            if position.get("_np_consec", 0) >= 2:
                log.info("[EXIT] %s NEVER_PROFITABLE age=%.0fm peak=%.2f%%",
                         pair_name, age_min, peak_pnl)
                return ExitDecision(
                    should_exit=True,
                    reason=f"NEVER_PROFITABLE (peak={peak_pnl:.2f}% age={age_min:.0f}m)",
                )

        # ── 7. CONSOLIDATION_EXIT ────────────────────────────────────────────
        if pair_regime == "RANGING" and abs(pnl_pct) <= 0.10:
            cons_min = position.get(
                "consolidation_exit_min",
                {"fast": 20, "bigwick": 45, "slow": 60}.get(lane, 60),
            )
            if age_min >= cons_min:
                position["_cons_consec"] = position.get("_cons_consec", 0) + 1
                if position["_cons_consec"] >= 2:
                    log.info("[EXIT] %s CONSOLIDATION age=%.0fm pnl=%.2f%%",
                             pair_name, age_min, pnl_pct)
                    return ExitDecision(
                        should_exit=True,
                        reason=f"CONSOLIDATION (RANGING age={age_min:.0f}m)",
                    )
            else:
                position["_cons_consec"] = 0
        else:
            position["_cons_consec"] = 0

        # ── 8. MAX_HOLD ──────────────────────────────────────────────────────
        age_h = age_min / 60.0
        dyn_max = max_hold_h + (_MAX_HOLD_EXTEND.get(lane, 0.5) if pnl_pct > 0.10 or partial_done else 0.0)
        if age_h >= dyn_max:
            log.info("[EXIT] %s MAX_HOLD %.1fh limit=%.1fh pnl=%.2f%%",
                     pair_name, age_h, dyn_max, pnl_pct)
            return ExitDecision(
                should_exit=True,
                reason=f"MAX_HOLD ({age_h:.1f}h≥{dyn_max:.1f}h)",
            )

        return None
