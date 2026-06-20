"""PositionTracker — owns the open-position dict + sync with WEEX state.

Key format: "<SYM>#<lane>" (V4.3.0-beta canonical). Legacy formats
"<SYM>:<SIDE>" and bare "<SYM>" are normalized on load. Mixed formats
in a single log = migrator missed a code path = P0 plumbing bug.

After any daemon crash → run sync_with_weex() before resuming.

Port reference:
  archive/v6.0/v4/smt_nightly_trade_v3_1.py:12483 — TradeTracker class
  archive/v6.0/v4/smt_daemon_v3_1.py — sync_tracker_with_weex,
    _split_key, _migrate_active_trades

Discord trade-alert hook in add() and close() — reuses v4/trade_alert_logger.py.
"""

from __future__ import annotations
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from smt.core.trade_plan import TradePlan

log = logging.getLogger("smt.tracker")

LANE_SEP = "#"

# Lazy import — silently no-ops if v4 module unavailable
_post_entry: Any = None
_post_exit: Any = None


def _load_alert_hooks() -> None:
    global _post_entry, _post_exit
    if _post_entry is not None:
        return
    try:
        try:
            from v4.trade_alert_logger import post_trade_entry, post_trade_exit
        except ImportError:
            from trade_alert_logger import post_trade_entry, post_trade_exit  # type: ignore[no-redef]
        _post_entry = post_trade_entry
        _post_exit = post_trade_exit
    except Exception:
        _post_entry = lambda **_: None  # noqa: E731
        _post_exit = lambda **_: None   # noqa: E731


def split_key(key: str) -> Tuple[str, str]:
    """Parse "<SYM>#<lane>" → (sym, lane). Accept legacy bare and ':SIDE' formats."""
    if LANE_SEP in key:
        sym, lane = key.split(LANE_SEP, 1)
        return sym, lane
    sym = key.split(":", 1)[0] if ":" in key else key
    return sym, "slow"


def _make_key(symbol: str, lane: str) -> str:
    return f"{symbol}{LANE_SEP}{lane}"


class PositionTracker:
    """In-memory positions + JSON persistence (trade_state_v6_1_0.json).

    Keys: "<SYM>#<lane>" — e.g. "BTCUSDT#slow", "BTCUSDT#bigwick".
    Same-symbol multiple lanes can coexist (V4.3.0-beta pyramiding).
    """

    def __init__(self, state_path: str = "v4/trade_state_v6_1_0.json") -> None:
        self.state_path = state_path
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.closed_trades: List[Dict[str, Any]] = []
        self.cooldowns: Dict[str, str] = {}
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path) as f:
                data = json.load(f)
            self.positions = data.get("active", {})
            self.closed_trades = data.get("closed", [])
            self.cooldowns = data.get("cooldowns", {})
            self._migrate()
        except Exception as exc:
            log.warning("[TRACKER] Could not load state from %s: %s", self.state_path, exc)

    def _save(self) -> None:
        try:
            parent = os.path.dirname(self.state_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.state_path, "w") as f:
                json.dump(
                    {"active": self.positions, "closed": self.closed_trades, "cooldowns": self.cooldowns},
                    f, indent=2, default=str,
                )
        except Exception as exc:
            log.warning("[TRACKER] Could not save state: %s", exc)

    def _migrate(self) -> None:
        """Re-key legacy bare / ':SIDE' keys to '<SYM>#<lane>'."""
        migrated: Dict[str, Any] = {}
        for key, trade in (self.positions or {}).items():
            if LANE_SEP in key:
                migrated[key] = trade
                continue
            sym, _ = split_key(key)
            lane = (trade or {}).get("entry_lane") or "slow"
            new_key = _make_key(sym, lane)
            migrated[new_key] = trade
            log.debug("[TRACKER] migrate %s → %s", key, new_key)
        self.positions = migrated

    # ── Public API ───────────────────────────────────────────────────────────

    def add(self, plan: TradePlan, weex_response: Dict[str, Any]) -> None:
        """Record newly opened position. Fires Discord trade-entry alert."""
        _load_alert_hooks()
        sym, _ = split_key(plan.pair)
        key = _make_key(sym, plan.lane)
        self.positions[key] = {
            "pair": plan.pair,
            "side": plan.direction,
            "entry_lane": plan.lane,
            "entry_price": plan.entry_price,
            "exit_target": plan.exit_target,
            "exit_stop": plan.exit_stop,
            "hold_max": plan.hold_max,
            "qty": plan.qty,
            "leverage": plan.leverage,
            "est_fees": plan.est_fees,
            "confidence": plan.decision_confidence,
            "persona_votes": plan.persona_votes,
            "tracker_key": key,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "weex_entry_resp": weex_response,
            "peak_pnl_pct": 0.0,
            "partial_close_done": False,
            "_against_consec": 0,
            "_drop_consec": 0,
            "_np_consec": 0,
            "_cons_consec": 0,
        }
        self._save()
        try:
            _post_entry(
                pair=sym, side=plan.direction, entry_price=plan.entry_price,
                confidence=plan.decision_confidence, lane=plan.lane,
                persona_votes=plan.persona_votes,
            )
        except Exception:
            pass
        log.info("[TRACKER] ADD %s %s entry=%.4f lev=%dx",
                 key, plan.direction, plan.entry_price, plan.leverage)

    def close(self, key: str, fill_price: float, fill_qty: float, reason: str = "") -> Dict[str, Any]:
        """Close a position lane. Returns closed-trade dict for JSONL logger."""
        _load_alert_hooks()
        sym, _ = split_key(key)
        trade = self.positions.pop(key, None)
        if trade is None:
            log.warning("[TRACKER] close: key %s not found in active positions", key)
            return {}
        entry_price = float(trade.get("entry_price") or 0.0)
        side = trade.get("side", "LONG")
        pnl_pct = 0.0
        if entry_price > 0 and fill_price > 0:
            pnl_pct = ((fill_price - entry_price) / entry_price * 100.0
                       if side == "LONG"
                       else (entry_price - fill_price) / entry_price * 100.0)
        lev = float(trade.get("leverage") or 1.0)
        qty = float(trade.get("qty") or fill_qty or 0.0)
        notional = qty * entry_price
        pnl_usd = notional * pnl_pct / 100.0
        age_h = 0.0
        opened_at = trade.get("opened_at", "")
        if opened_at:
            try:
                dt = datetime.fromisoformat(str(opened_at).replace("Z", "+00:00"))
                age_h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
            except Exception:
                pass
        closed = {
            **trade,
            "closed_at": datetime.now(timezone.utc).isoformat(),
            "fill_price": fill_price,
            "fill_qty": fill_qty,
            "close_reason": reason,
            "pnl_pct": pnl_pct,
            "pnl_usd": pnl_usd,
            "hours_open": round(age_h, 3),
        }
        self.closed_trades.append(closed)
        self._save()
        try:
            _post_exit(
                pair=sym, side=side, exit_price=fill_price,
                age_minutes=age_h * 60.0, pnl_pct=pnl_pct,
                pnl_usd=pnl_usd, exit_reason=reason or "—",
            )
        except Exception:
            pass
        log.info("[TRACKER] CLOSE %s fill=%.4f pnl=%.2f%% (%.2fUSD) %s",
                 key, fill_price, pnl_pct, pnl_usd, reason)
        return closed

    def mark_partial_close(self, key: str, new_qty: float,
                           new_stop: Optional[float] = None) -> None:
        """Latch a PARTIAL_CLOSE: halve qty, move SL→entry, set the done flag.

        Called by the daemon (Session E) after exec.close_partial +
        move_stop_to_entry succeed. `partial_close_done` gates the exit cascade
        so the 50%-close fires exactly once per position.
        """
        trade = self.positions.get(key)
        if trade is None:
            log.warning("[TRACKER] mark_partial_close: key %s not found", key)
            return
        trade["partial_close_done"] = True
        trade["qty"] = new_qty
        if new_stop is not None:
            trade["exit_stop"] = new_stop
        trade["partial_closes"] = int(trade.get("partial_closes", 0)) + 1
        self._save()
        log.info("[TRACKER] PARTIAL %s qty→%.6f SL→%s", key, new_qty,
                 f"{new_stop:.6f}" if new_stop is not None else "—")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self.positions.get(key)

    def all(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.positions)

    def lanes_for_pair(self, symbol: str) -> Dict[str, Dict[str, Any]]:
        """Return {lane: trade} for all open lanes on this symbol."""
        sym, _ = split_key(symbol)
        return {split_key(k)[1]: t for k, t in self.positions.items() if split_key(k)[0] == sym}

    def sync_with_weex(self, exec_client: Any) -> None:
        """Reconcile in-memory state with exchange after daemon restart.

        Fetches /capi/v3/positions (via exec_client) and removes local
        positions that no longer exist on exchange (SL/TP hit while offline).
        """
        try:
            resp = exec_client._get("/capi/v3/account/position/allPosition")
            live = resp if isinstance(resp, list) else (resp.get("data") or [])
            live_syms = {
                p["symbol"] for p in live
                if abs(float(p.get("size") or 0.0)) > 1e-10
            }
            stale = [k for k in list(self.positions.keys()) if split_key(k)[0] not in live_syms]
            for k in stale:
                log.warning("[TRACKER] sync: stale %s not on exchange — removing", k)
                self.close(k, fill_price=0.0, fill_qty=0.0, reason="sync_stale_removal")
            log.info("[TRACKER] sync_with_weex done: %d live syms, %d stale removed",
                     len(live_syms), len(stale))
        except Exception as exc:
            log.error("[TRACKER] sync_with_weex failed: %s", exc)
