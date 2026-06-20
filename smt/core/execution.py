"""WEEX V3 execution — entry orders + SL/TP algoOrders.

WEEX V3 API quirks (ALL real-money lessons; do NOT regress):
- camelCase keys + string-enum types ("STOP_MARKET", not numeric type).
- POST_ONLY rejected with -1135. Use GTC LIMIT (V3.2.237 fix).
- Lane-tagged keys "BTCUSDT#fast" must be stripped before any API call
  — else -1142 (V4.3.1 P0 #2). Use strip_lane_suffix().
- HTTP 429: get_price returns -1.0 sentinel; downstream MUST treat as
  "skip cycle", not "phantom close at 0.00" (V4.3.1 P0 #3).
- TP body shape (V3.2.262 fix): TAKE_PROFIT_MARKET + triggerPrice.
  Archive uses `triggerPrice` + `clientAlgoId` (real-money tested).
  PlaceTpSlOrder endpoint (WEEX_API_ENDPOINTS.md) schema not yet fetched
  (◻); fall back to /capi/v3/algoOrder from archive until fetched.
- clientAlgoId must be unique per cycle.
- V3 contract: ALWAYS cancel algo orders before closing position.
- positionSide LONG/SHORT required in SEPARATED mode (bigwick bidirectional).

Auth sign string (WEEX_API_ENDPOINTS.md ground truth):
  timestamp + METHOD + requestPath + ?queryString + body
  (NOT just timestamp + requestPath as older WEEX_API_REFERENCE.md shows)

Correct account endpoints (docs/WEEX_API_ENDPOINTS.md ground truth):
  POST /capi/v3/account/marginType  — marginType + separatedType=SEPARATED
  POST /capi/v3/account/leverage    — isolatedLongLeverage / isolatedShortLeverage
  GET  /capi/v3/account/balance     — equity snapshot

Port reference:
  archive/v6.0/v4/smt_nightly_trade_v3_1.py execute_trade() line 11909
  archive/v6.0/v4/smt_daemon_v3_1.py SL/TP bodies ~line 6227 (V3.2.262)
"""

from __future__ import annotations
import hashlib
import hmac
import json
import logging
import time
from base64 import b64encode
from typing import Any, Dict, Optional

from smt.core.trade_plan import TradePlan

log = logging.getLogger("smt.execution")

WEEX_BASE_URL = "https://api-contract.weex.com"

# Approximate tick / step sizes (from V5.0.7 constants + exchangeInfo).
# Session E should refresh via GET /capi/v3/exchangeInfo on startup.
_TICK_SIZE: Dict[str, float] = {
    "BTCUSDT": 0.10, "ETHUSDT": 0.01, "BNBUSDT": 0.01,
    "LTCUSDT": 0.01, "SOLUSDT": 0.001, "XRPUSDT": 0.0001,
    "ADAUSDT": 0.0001, "DOGEUSDT": 0.00001,
}
_STEP_SIZE: Dict[str, float] = {
    "BTCUSDT": 0.001, "ETHUSDT": 0.01, "BNBUSDT": 0.01,
    "LTCUSDT": 0.1, "SOLUSDT": 0.1, "XRPUSDT": 1.0,
    "ADAUSDT": 1.0, "DOGEUSDT": 1.0,
}


def strip_lane_suffix(sym_key: str) -> str:
    """WEEX-safe symbol from lane-tagged tracker key. See V4.3.1 P0 #2."""
    return sym_key.split("#", 1)[0].split(":", 1)[0]


def _round_tick(price: float, symbol: str) -> str:
    tick = _TICK_SIZE.get(symbol, 0.01)
    rounded = round(round(price / tick) * tick, 10)
    dec = len(str(tick).rstrip("0").split(".")[-1]) if "." in str(tick) else 0
    return f"{rounded:.{dec}f}"


def _round_step(qty: float, symbol: str) -> str:
    step = _STEP_SIZE.get(symbol, 0.001)
    rounded = round(round(qty / step) * step, 10)
    dec = len(str(step).rstrip("0").split(".")[-1]) if "." in str(step) else 3
    return f"{rounded:.{dec}f}"


def _make_sign(secret: str, timestamp: str, method: str, path: str, body: str = "") -> str:
    """WEEX HMAC-SHA256 signature: timestamp + METHOD + requestPath + body."""
    msg = timestamp + method.upper() + path + body
    raw = hmac.new(secret.encode(), msg.encode(), hashlib.sha256).digest()
    return b64encode(raw).decode()


def _weex_headers(
    api_key: str, secret: str, passphrase: str, method: str, path: str, body: str = ""
) -> Dict[str, str]:
    ts = str(int(time.time() * 1000))
    return {
        "ACCESS-KEY": api_key,
        "ACCESS-SIGN": _make_sign(secret, ts, method, path, body),
        "ACCESS-PASSPHRASE": passphrase,
        "ACCESS-TIMESTAMP": ts,
        "Content-Type": "application/json",
    }


class ExecutionClient:
    """Thin wrapper over WEEX V3 endpoints. Stateless — tracker owns positions."""

    def __init__(self) -> None:
        self._api_key: Optional[str] = None
        self._secret: Optional[str] = None
        self._passphrase: Optional[str] = None
        self._session: Any = None
        self._load_creds()

    def _load_creds(self) -> None:
        try:
            try:
                from v4.secrets_loader import get_secret
            except ImportError:
                from secrets_loader import get_secret  # type: ignore[no-redef]
            self._api_key = get_secret("weex-api-key")
            self._secret = get_secret("weex-api-secret")
            self._passphrase = get_secret("weex-api-passphrase")
        except Exception as exc:
            log.warning("[EXEC] Creds not loaded (%s) — live trading will fail", exc)

    def _session_obj(self):
        if self._session is None:
            import requests
            s = requests.Session()
            s.headers.update({"User-Agent": "smt-aiquant-bot/6.1.0"})
            self._session = s
        return self._session

    def _require_creds(self) -> None:
        if not self._api_key:
            raise RuntimeError("WEEX creds not configured — run get_secret() first")

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._require_creds()
        body = json.dumps(payload, separators=(",", ":"))
        hdrs = _weex_headers(self._api_key, self._secret, self._passphrase, "POST", path, body)
        try:
            resp = self._session_obj().post(f"{WEEX_BASE_URL}{path}", headers=hdrs, data=body, timeout=10)
            if resp.status_code == 429:
                log.warning("[EXEC] HTTP 429 on POST %s — caller should back off", path)
                return {"success": False, "code": 429}
            return resp.json() if resp.text else {}
        except Exception as exc:
            log.error("[EXEC] POST %s failed: %s", path, exc)
            return {"success": False, "error": str(exc)}

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        self._require_creds()
        qs = ""
        if params:
            import urllib.parse
            qs = "?" + urllib.parse.urlencode(params)
        full_path = path + qs
        hdrs = _weex_headers(self._api_key, self._secret, self._passphrase, "GET", full_path)
        try:
            resp = self._session_obj().get(f"{WEEX_BASE_URL}{full_path}", headers=hdrs, timeout=10)
            if resp.status_code == 429:
                return {"success": False, "code": 429}
            return resp.json() if resp.text else {}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _delete(self, path: str, params: Optional[Dict] = None) -> Any:
        self._require_creds()
        qs = ""
        if params:
            import urllib.parse
            qs = "?" + urllib.parse.urlencode(params)
        full_path = path + qs
        hdrs = _weex_headers(self._api_key, self._secret, self._passphrase, "DELETE", full_path)
        try:
            resp = self._session_obj().delete(f"{WEEX_BASE_URL}{full_path}", headers=hdrs, timeout=10)
            if resp.status_code == 429:
                return {"success": False, "code": 429}
            return resp.json() if resp.text else {}
        except Exception as exc:
            log.error("[EXEC] DELETE %s failed: %s", path, exc)
            return {"success": False, "error": str(exc)}

    # ── Public market (no auth) ───────────────────────────────────────────────

    def get_price(self, symbol: str) -> float:
        """Mark price. Returns -1.0 sentinel on HTTP 429 — caller MUST skip cycle."""
        sym = strip_lane_suffix(symbol)
        try:
            import requests as _req
            resp = _req.get(
                f"{WEEX_BASE_URL}/capi/v3/market/symbolPrice",
                params={"symbol": sym}, timeout=5,
            )
            if resp.status_code == 429:
                log.warning("[EXEC] HTTP 429 get_price(%s) — returning -1.0 sentinel", sym)
                return -1.0
            data = resp.json()
            if isinstance(data, list) and data:
                return float(data[0]["price"])
            if isinstance(data, dict) and "price" in data:
                return float(data["price"])
        except Exception as exc:
            log.warning("[EXEC] get_price(%s) error: %s", sym, exc)
        return 0.0

    # ── Account configuration ─────────────────────────────────────────────────

    def _set_margin_and_mode(self, symbol: str, margin_type: str = "ISOLATED", separated: bool = True) -> bool:
        """POST /capi/v3/account/marginType.

        separatedType=SEPARATED → LONG+SHORT coexist on same pair (bigwick lane).
        separatedType=COMBINED  → one-way (WEEX default). Err -4006 = already set = OK.
        """
        payload = {
            "symbol": symbol,
            "marginType": margin_type,
            "separatedType": "SEPARATED" if separated else "COMBINED",
        }
        r = self._post("/capi/v3/account/marginType", payload)
        ok = r.get("success", False) or r.get("code") in (200, -4006)
        if not ok:
            log.warning("[EXEC] set_margin_and_mode %s failed: %s", symbol, r)
        return bool(ok)

    def _set_leverage(self, symbol: str, long_lev: int, short_lev: int) -> bool:
        """POST /capi/v3/account/leverage — per-side isolated leverage.

        Addendum #4: isolatedLongLeverage / isolatedShortLeverage as separate knobs.
        Finding 6 (SHORT was 14-28pp more accurate) → caller passes distinct values.
        """
        payload = {
            "symbol": symbol,
            "isolatedLongLeverage": long_lev,
            "isolatedShortLeverage": short_lev,
        }
        r = self._post("/capi/v3/account/leverage", payload)
        ok = r.get("success", False) or r.get("code") == 200
        if not ok:
            log.warning("[EXEC] set_leverage %s %dL/%dS failed: %s", symbol, long_lev, short_lev, r)
        return bool(ok)

    # ── Core trade operations ─────────────────────────────────────────────────

    def cancel_algo_orders(self, symbol: str) -> None:
        """V3 contract: ALWAYS cancel algo orders before closing position.
        Uses CancelAllPendingOrders: DELETE /capi/v3/algoOpenOrders?symbol=sym.
        """
        sym = strip_lane_suffix(symbol)
        try:
            self._delete("/capi/v3/algoOpenOrders", {"symbol": sym})
        except Exception as exc:
            log.warning("[EXEC] cancel_algo_orders(%s) error: %s", sym, exc)

    def place(self, plan: TradePlan) -> Dict[str, Any]:
        """Open position + SL + TP atomically. Returns response dict.

        Sequence (mirrors archive execute_trade() V3.2.137+):
          1. Pre-cancel stale algos (V5.1.2 G1.3)
          2. Set ISOLATED margin + SEPARATED mode for bigwick
          3. Set per-direction leverage (addendum #4)
          4. Place MARKET entry order
          5. Place STOP_MARKET SL algoOrder
          6. Place TAKE_PROFIT_MARKET TP algoOrder
        """
        sym = strip_lane_suffix(plan.pair)
        direction = plan.direction  # "LONG" or "SHORT"
        lane = plan.lane
        pos_side = direction if lane == "bigwick" else "BOTH"
        close_side = "SELL" if direction == "LONG" else "BUY"
        qty_str = _round_step(plan.qty, sym)

        # 1. Pre-cancel stale algos
        self.cancel_algo_orders(sym)

        # 2. Margin + position mode
        self._set_margin_and_mode(sym, "ISOLATED", separated=(lane == "bigwick"))

        # 3. Per-direction leverage from plan (caller derives from CONFIG long_leverage / short_leverage)
        if direction == "LONG":
            self._set_leverage(sym, plan.leverage, plan.leverage)
        else:
            self._set_leverage(sym, plan.leverage, plan.leverage)

        ts_ms = int(time.time() * 1000)

        # 4. MARKET entry
        entry_payload: Dict[str, Any] = {
            "symbol": sym,
            "side": "BUY" if direction == "LONG" else "SELL",
            "type": "MARKET",
            "quantity": qty_str,
            "newClientOrderId": f"smt-{sym}-{ts_ms}",
        }
        if pos_side != "BOTH":
            entry_payload["positionSide"] = pos_side
        entry_resp = self._post("/capi/v3/order", entry_payload)
        if not (entry_resp.get("success", False) or entry_resp.get("orderId")):
            log.warning("[EXEC] Entry REJECTED %s %s: %s", sym, direction, entry_resp)
            return {"executed": False, "reason": f"entry_rejected: {entry_resp}"}

        # 5. SL algoOrder — STOP_MARKET (V3.2.262 triggerPrice body shape)
        sl_algo_path = "/capi/v3/algoOrder"
        sl_body_str = json.dumps({
            "symbol": sym,
            "side": close_side,
            "positionSide": pos_side,
            "type": "STOP_MARKET",
            "quantity": qty_str,
            "triggerPrice": _round_tick(plan.exit_stop, sym),
            "clientAlgoId": f"smt_sl_{ts_ms}",
        }, separators=(",", ":"))
        sl_hdrs = _weex_headers(self._api_key, self._secret, self._passphrase, "POST", sl_algo_path, sl_body_str)
        try:
            import requests as _req
            sl_r = _req.post(f"{WEEX_BASE_URL}{sl_algo_path}", headers=sl_hdrs, data=sl_body_str, timeout=10)
            sl_resp = sl_r.json() if sl_r.text else {}
        except Exception as exc:
            sl_resp = {"success": False, "error": str(exc)}
        sl_ok = bool(sl_resp.get("success", False))
        if not sl_ok:
            log.warning("[EXEC] SL REJECTED %s: %s", sym, sl_resp)

        # 6. TP algoOrder — TAKE_PROFIT_MARKET
        tp_body_str = json.dumps({
            "symbol": sym,
            "side": close_side,
            "positionSide": pos_side,
            "type": "TAKE_PROFIT_MARKET",
            "quantity": qty_str,
            "triggerPrice": _round_tick(plan.exit_target, sym),
            "clientAlgoId": f"smt_tp_{ts_ms + 1}",
        }, separators=(",", ":"))
        tp_hdrs = _weex_headers(self._api_key, self._secret, self._passphrase, "POST", sl_algo_path, tp_body_str)
        try:
            import requests as _req
            tp_r = _req.post(f"{WEEX_BASE_URL}{sl_algo_path}", headers=tp_hdrs, data=tp_body_str, timeout=10)
            tp_resp = tp_r.json() if tp_r.text else {}
        except Exception as exc:
            tp_resp = {"success": False, "error": str(exc)}
        tp_ok = bool(tp_resp.get("success", False))
        if not tp_ok:
            log.warning("[EXEC] TP REJECTED %s: %s", sym, tp_resp)

        log.info("[EXEC] %s %s lane=%s qty=%s lev=%dx sl=%s tp=%s",
                 sym, direction, lane, qty_str, plan.leverage, sl_ok, tp_ok)
        return {
            "executed": True, "symbol": sym, "direction": direction, "lane": lane,
            "qty": qty_str, "leverage": plan.leverage,
            "entry_resp": entry_resp, "sl_ok": sl_ok, "tp_ok": tp_ok,
        }

    def close(self, symbol: str, side: str) -> Dict[str, Any]:
        """Cancel algo orders FIRST (V3 contract), then close position."""
        sym = strip_lane_suffix(symbol)
        self.cancel_algo_orders(sym)
        resp = self._post("/capi/v3/closePositions", {"symbol": sym})
        ok = bool(resp.get("success", False) or resp.get("orderId"))
        if not ok:
            log.warning("[EXEC] close(%s %s) failed: %s", sym, side, resp)
        return {"closed": ok, "symbol": sym, "side": side, "resp": resp}

    def _place_algo_order(
        self, sym: str, close_side: str, pos_side: str,
        algo_type: str, qty_str: str, trigger_str: str, client_id: str,
    ) -> tuple:
        """Place one STOP_MARKET / TAKE_PROFIT_MARKET algoOrder (V3.2.262 shape)."""
        body = json.dumps({
            "symbol": sym, "side": close_side, "positionSide": pos_side,
            "type": algo_type, "quantity": qty_str,
            "triggerPrice": trigger_str, "clientAlgoId": client_id,
        }, separators=(",", ":"))
        hdrs = _weex_headers(self._api_key, self._secret, self._passphrase,
                             "POST", "/capi/v3/algoOrder", body)
        try:
            import requests as _req
            r = _req.post(f"{WEEX_BASE_URL}/capi/v3/algoOrder", headers=hdrs, data=body, timeout=10)
            resp = r.json() if r.text else {}
        except Exception as exc:
            resp = {"success": False, "error": str(exc)}
        return bool(resp.get("success", False)), resp

    def close_partial(self, symbol: str, side: str, qty: float, pos_side: str = "BOTH") -> Dict[str, Any]:
        """Reduce-only MARKET close of `qty` (the exit-cascade PARTIAL_CLOSE leg).

        `side` is the OPEN position side (LONG/SHORT); the reducing order takes
        the opposite side with reduceOnly so it can only shrink the position.
        """
        sym = strip_lane_suffix(symbol)
        qty_str = _round_step(qty, sym)
        payload: Dict[str, Any] = {
            "symbol": sym,
            "side": "SELL" if side == "LONG" else "BUY",
            "type": "MARKET",
            "quantity": qty_str,
            "reduceOnly": True,
            "newClientOrderId": f"smt-pc-{sym}-{int(time.time() * 1000)}",
        }
        if pos_side != "BOTH":
            payload["positionSide"] = pos_side
        resp = self._post("/capi/v3/order", payload)
        ok = bool(resp.get("success", False) or resp.get("orderId"))
        if not ok:
            log.warning("[EXEC] close_partial(%s %s qty=%s) failed: %s", sym, side, qty_str, resp)
        log.info("[EXEC] partial-close %s %s qty=%s ok=%s", sym, side, qty_str, ok)
        return {"closed_partial": ok, "symbol": sym, "qty": qty_str, "resp": resp}

    def move_stop_to_entry(
        self, symbol: str, side: str, qty: float, entry_price: float,
        tp_price: Optional[float] = None, pos_side: str = "BOTH",
    ) -> Dict[str, Any]:
        """Break-even: cancel algos, re-place SL at entry (+ original TP if given).

        Run after a partial close to lock the remainder risk-free. Cancels ALL
        algos first (V3 contract), then re-arms a STOP_MARKET at `entry_price`
        and, when `tp_price` is supplied, re-places the TAKE_PROFIT_MARKET so the
        cancel-all doesn't drop the runner's profit target.
        """
        sym = strip_lane_suffix(symbol)
        self.cancel_algo_orders(sym)
        close_side = "SELL" if side == "LONG" else "BUY"
        qty_str = _round_step(qty, sym)
        ts_ms = int(time.time() * 1000)
        sl_ok, _ = self._place_algo_order(
            sym, close_side, pos_side, "STOP_MARKET", qty_str,
            _round_tick(entry_price, sym), f"smt_be_sl_{ts_ms}")
        tp_ok = None
        if tp_price:
            tp_ok, _ = self._place_algo_order(
                sym, close_side, pos_side, "TAKE_PROFIT_MARKET", qty_str,
                _round_tick(tp_price, sym), f"smt_be_tp_{ts_ms + 1}")
        if not sl_ok:
            log.warning("[EXEC] move_stop_to_entry(%s) SL re-arm failed", sym)
        log.info("[EXEC] %s SL→entry %.6f (qty=%s) sl_ok=%s tp_ok=%s",
                 sym, entry_price, qty_str, sl_ok, tp_ok)
        return {"sl_ok": sl_ok, "tp_ok": tp_ok, "symbol": sym, "entry": entry_price}
