"""GlobalContext — macro + cross-pair + regime + recent-fill snapshot.

Single object passed into every Persona.analyze() and Strategy.*_signal()
call. Refreshed by the daemon at the top of each cycle.

NOT in context: persona outputs (those are PRODUCED from context).

Fields wired live in Session E (Daemon.cycle wiring):
  klines per pair per timeframe (1m/5m/15m/1h/4h/1d) via WEEX
    GET /capi/v1/market/kline (cache in v4/)
  per-pair regime via compute_pair_regime(symbol) (V5.0.7 authoritative)
  BTC dominance (CMC), F&G (CMC, 15-min cache)
  funding rates (WEEX + Binance passthrough)
  cross-pair cascade state (B.10 detector V6.0.6)
  recent fills + open positions (tracker passthrough)

Session C: refresh() does NOT raise on missing live wiring — it sets the
no-data defaults so personas + JUDGE can still run end-to-end. Mark every
TODO so Session E (daemon end-to-end) knows where to plug real fetchers.
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger("smt.context")

PAIRS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "LTCUSDT",
         "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT")


@dataclass
class GlobalContext:
    timestamp: Optional[float] = None
    klines: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    regime: Dict[str, str] = field(default_factory=dict)
    fear_greed: Optional[int] = None
    btc_dominance_pct: Optional[float] = None
    funding_rates: Dict[str, float] = field(default_factory=dict)
    cascade_state: Dict[str, Any] = field(default_factory=dict)
    recent_fills: List[Dict[str, Any]] = field(default_factory=list)
    open_positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    prices: Dict[str, float] = field(default_factory=dict)
    equity_usd: float = 40_000.0

    # Per-cycle persona pre-computes (daemon-populated, Session E)
    flow_signal: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    technical_signal: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    whale_data: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    aggtrades: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    sentiment_signal: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    onchain_signal: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # JUDGE results per pair (populated by daemon after personas + judge run)
    judge: Dict[str, Any] = field(default_factory=dict)

    # Daemon-attached PositionTracker (Session E). Not part of as_dict().
    _tracker: Any = field(default=None, repr=False, compare=False)

    def attach_tracker(self, tracker: Any) -> None:
        """Daemon attaches the live PositionTracker BEFORE refresh() (Session E).

        refresh() then mirrors tracker.all() into open_positions and the recent
        closed trades into recent_fills, so personas + the exit cascade see the
        bot's own state without the tracker reaching into the context directly.
        """
        self._tracker = tracker

    def refresh(self) -> None:
        """Refresh self with the latest snapshot.

        Session C: scaffolded stubs — no raise. Session E adds tracker
        passthrough (open_positions + recent_fills) and derives a mark price
        per pair from the latest kline close, so Daemon.cycle() runs end-to-end
        without live network. The live WEEX kline / CMC / funding fetchers stay
        tagged "# TODO Session E: live" — wiring them is the remaining live-data
        step (Session J restart); everything else is exercisable today.
        """
        self.timestamp = time.time()
        try:
            self._refresh_klines()
            self._refresh_regimes()
            self._refresh_macro()
            self._refresh_funding()
            self._refresh_tracker_passthrough()
            self._refresh_prices()
        except Exception as e:
            log.exception("[CONTEXT] refresh() degraded: %s", e)

    def as_dict(self) -> Dict[str, Any]:
        """Strategy + Persona consumers prefer a plain dict view."""
        return {
            "timestamp": self.timestamp,
            "klines": self.klines,
            "regime": self.regime,
            "fear_greed": self.fear_greed,
            "btc_dominance_pct": self.btc_dominance_pct,
            "funding_rates": self.funding_rates,
            "cascade_state": self.cascade_state,
            "recent_fills": self.recent_fills,
            "open_positions": self.open_positions,
            "prices": self.prices,
            "equity_usd": self.equity_usd,
            "flow_signal": self.flow_signal,
            "technical_signal": self.technical_signal,
            "whale_data": self.whale_data,
            "aggtrades": self.aggtrades,
            "sentiment_signal": self.sentiment_signal,
            "onchain_signal": self.onchain_signal,
            "judge": self.judge,
        }

    # ── Sub-fetchers ─────────────────────────────────────────────────────────

    def _refresh_klines(self) -> None:
        # TODO Session E: live WEEX GET /capi/v1/market/kline per pair per TF.
        # For now: leave self.klines as-is (test injects). No-op safely.
        pass

    def _refresh_regimes(self) -> None:
        # TODO Session E: call compute_pair_regime(symbol) per pair.
        # Until then default any missing pair to "NORMAL" so HARD-BLOCK lookup
        # behaves predictably.
        for sym in PAIRS:
            self.regime.setdefault(sym, "NORMAL")

    def _refresh_macro(self) -> None:
        # TODO Session E: CMC F&G + BTC dominance (15-min cache, CMC scale).
        # Default to "neutral" band so JUDGE band logic is a no-op until live.
        if self.fear_greed is None:
            self.fear_greed = 50
        if self.btc_dominance_pct is None:
            self.btc_dominance_pct = 55.0

    def _refresh_funding(self) -> None:
        # TODO Session E: WEEX + Binance funding passthrough.
        # Default each pair's funding to 0.0 (neutral).
        for sym in PAIRS:
            self.funding_rates.setdefault(sym, 0.0)

    def _refresh_tracker_passthrough(self) -> None:
        # Session E: mirror the attached tracker's open positions + recent closes.
        tr = self._tracker
        if tr is None:
            return
        try:
            self.open_positions = tr.all()
            closed = list(getattr(tr, "closed_trades", []) or [])
            self.recent_fills = closed[-20:]
        except Exception as e:
            log.warning("[CONTEXT] tracker passthrough degraded: %s", e)

    def _refresh_prices(self) -> None:
        # Session E: derive a mark price per pair from the latest kline close.
        # Keeps any price already set (test-injected or daemon exec.get_price
        # fallback). Live WEEX symbolPrice wiring lands with live klines (Session J).
        for sym in PAIRS:
            if self.prices.get(sym):
                continue
            for key in (f"{sym}#1h", f"{sym}_1h", f"{sym}:1h",
                        f"{sym}#5m", f"{sym}#15m", f"{sym}#1m"):
                candles = self.klines.get(key) or []
                if candles:
                    try:
                        self.prices[sym] = float(candles[-1][4])
                    except (IndexError, ValueError, TypeError):
                        pass
                    break
