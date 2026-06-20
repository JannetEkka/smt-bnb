"""CoinMarketCap → SMT context adapter (BNB Hack: AI Trading Agent Edition).

The SMT personas read a fixed context schema (see smt/personas/*.py):
  - context["klines"][f"{PAIR}#1h"]   → OHLCV list (TechnicalPersona RSI/slope, FlowPersona fallback)
  - context["flow_signal"][PAIR]      → {direction, confidence}  (FlowPersona primary)
  - context["technical_signal"][PAIR] → {direction, confidence, reasoning}
  - context["regime"][PAIR]           → label string ("TRENDING_UP"/"RANGING"/…)
  - context["fear_greed"]             → int (CMC scale)
  - context["funding_rates"][PAIR]    → float

This adapter ONLY translates the data source — it populates those dicts from
CoinMarketCap, so the brain is untouched (CLAUDE.md contract #11). It is a pure,
offline-safe function: pass `quotes`/`ohlcv`/`fear_greed`/`funding` in directly
(tests + the demo do this), or let `CMCClient` fetch them live when a key is set.

Honesty note (alpha boundary): CMC's free surface has no L2 orderbook, so the FLOW
input here is a transparent **volume-weighted momentum proxy** derived from CMC's
24h % change + volume — NOT the WEEX/Binance L2 composite the production bot uses.
We label it as a proxy rather than pretend it's true order flow.
"""

from __future__ import annotations
import logging
import os
from typing import Any, Dict, List, Optional, Sequence

log = logging.getLogger("smt.hackathons.bnb.cmc")

# SMT's signal universe (8 pairs). The hackathon's eligible BEP-20 set (see
# twak_adapter.ELIGIBLE_TOKENS) excludes BTC/BNB/SOL, so the *tradeable* subset is
# {ETH, XRP, DOGE, ADA, LTC}; the rest stay signal-only on BSC. We still SCORE all 8.
SMT_SYMBOLS = ["BTC", "ETH", "BNB", "SOL", "LTC", "XRP", "ADA", "DOGE"]


def _regime_from_change(chg_pct: float) -> str:
    """CMC 24h % change → a coarse SMT regime label (refined by the bot's own classifier live)."""
    if chg_pct >= 3.0:
        return "TRENDING_UP"
    if chg_pct <= -3.0:
        return "TRENDING_DOWN"
    return "RANGING"


def _flow_proxy(chg_pct: float, vol_ratio: float) -> Optional[Dict[str, Any]]:
    """Volume-weighted momentum proxy → {direction, confidence}. None when too weak to vote."""
    if abs(chg_pct) < 0.5:
        return None
    direction = "LONG" if chg_pct > 0 else "SHORT"
    # Confidence grows with move size, scaled by how heavy volume is vs its own average.
    conf = min(0.75, 0.30 + abs(chg_pct) * 0.04) * max(0.5, min(1.5, vol_ratio))
    return {"direction": direction, "confidence": round(min(0.85, conf), 4),
            "reasoning": f"CMC momentum {chg_pct:+.2f}% (vol×{vol_ratio:.2f})"}


def build_context(
    symbols: Optional[Sequence[str]] = None,
    *,
    quotes: Optional[Dict[str, Dict[str, float]]] = None,
    ohlcv: Optional[Dict[str, List[Sequence[float]]]] = None,
    fear_greed: Optional[int] = None,
    funding: Optional[Dict[str, float]] = None,
    client: "Optional[CMCClient]" = None,
) -> Dict[str, Any]:
    """Assemble the SMT context dict from CMC data.

    `quotes[SYM]` = {"price", "percent_change_24h", "volume_ratio"} (volume_ratio
    optional, defaults 1.0). All inputs optional → missing pairs simply produce no
    signal (personas degrade to NEUTRAL), so this never raises and runs fully offline.
    """
    symbols = list(symbols or SMT_SYMBOLS)
    quotes = dict(quotes or {})
    if client is not None and not quotes:
        quotes = client.fetch_quotes(symbols) or {}
        if fear_greed is None:
            fear_greed = client.fetch_fear_greed()

    ctx: Dict[str, Any] = {
        "klines": {},
        "flow_signal": {},
        "technical_signal": {},
        "regime": {},
        "funding_rates": {},
        "prices": {},
        "fear_greed": 50 if fear_greed is None else int(fear_greed),
        "source": "coinmarketcap",
    }

    for sym in symbols:
        q = quotes.get(sym) or {}
        chg = float(q.get("percent_change_24h", 0.0) or 0.0)
        vol_ratio = float(q.get("volume_ratio", 1.0) or 1.0)
        if "price" in q:
            ctx["prices"][sym] = float(q["price"])
        ctx["regime"][sym] = _regime_from_change(chg)
        flow = _flow_proxy(chg, vol_ratio)
        if flow:
            ctx["flow_signal"][sym] = flow
        if ohlcv and sym in ohlcv:
            # Let TechnicalPersona compute RSI/slope itself from real candles (canonical key).
            ctx["klines"][f"{sym}#1h"] = list(ohlcv[sym])
        if funding and sym in funding:
            ctx["funding_rates"][sym] = float(funding[sym])

    return ctx


class CMCClient:
    """Thin CoinMarketCap client (lazy `requests`; graceful None without a key).

    The operator already holds `cmc-api-key` in GCP Secret Manager (CLAUDE.md
    lesson #12) — pass it as CMC_API_KEY. Live wiring to the CMC AI Agent Hub
    (MCP / x402 / CLI) is the production path; this REST shim keeps the demo
    self-contained and is the seam where the Agent Hub plugs in.
    """

    BASE = "https://pro-api.coinmarketcap.com"

    def __init__(self, api_key: Optional[str] = None, timeout: int = 10):
        self.api_key = api_key or os.getenv("CMC_API_KEY")
        self.timeout = timeout

    def _get(self, path: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.api_key:
            log.warning("[CMC] no API key — returning None (offline/signal-only)")
            return None
        try:
            import requests  # lazy
            r = requests.get(self.BASE + path, params=params, timeout=self.timeout,
                             headers={"X-CMC_PRO_API_KEY": self.api_key, "Accept": "application/json"})
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001 — never block the agent on a slow feed
            log.warning("[CMC] %s failed (%s) — None", path, exc)
            return None

    def fetch_quotes(self, symbols: Sequence[str]) -> Dict[str, Dict[str, float]]:
        """Latest quotes → {SYM: {price, percent_change_24h, volume_ratio}}. {} on failure."""
        data = self._get("/v1/cryptocurrency/quotes/latest", {"symbol": ",".join(symbols)})
        out: Dict[str, Dict[str, float]] = {}
        for sym in symbols:
            row = (((data or {}).get("data") or {}).get(sym) or {})
            usd = ((row.get("quote") or {}).get("USD") or {})
            if usd:
                out[sym] = {"price": float(usd.get("price", 0.0)),
                            "percent_change_24h": float(usd.get("percent_change_24h", 0.0)),
                            "volume_ratio": 1.0}
        return out

    def fetch_fear_greed(self) -> Optional[int]:
        """CMC Fear & Greed index (0..100). None on failure."""
        data = self._get("/v3/fear-and-greed/latest", {})
        try:
            return int(((data or {}).get("data") or {}).get("value"))
        except (TypeError, ValueError):
            return None
