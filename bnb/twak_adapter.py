"""Trust Wallet Agent Kit (TWAK) execution adapter — BNB Hack Track 1 (live BSC).

Mirrors ``smt.core.execution.ExecutionClient``'s ``place(plan)`` / ``close(symbol, side)``
surface, but signs + processes its own transactions on BSC via Trust Wallet (self-
custody, local signing — keys never leave the user). A drop-in SIBLING to the WEEX
adapter, not a fork (CLAUDE.md contract #11): the brain decides, this signs.

Everything routes through ``RiskGuardrails`` BEFORE any signing — the competition's
risk gate (blow past ~30% drawdown → DQ) plus the un-disableable SMT fee floor, a
token allowlist (the contest's eligible BEP-20 set), per-trade + daily caps, and
slippage protection. Offline (no signer) it returns a clean rejection and never
touches the network, so the demo + tests run with no keys.

Targets the **Best Use of TWAK** special prize: TWAK as the SOLE execution layer,
self-custody preserved end-to-end, autonomous signing inside hard guardrails.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

log = logging.getLogger("smt.hackathons.bnb.twak")

# Official eligible BEP-20 tokens (CoinMarketCap-listed) for the Track-1 competition —
# trades outside this set do not count. Source: the hackathon rules (149 symbols).
# Used here as the execution TOKEN ALLOWLIST guardrail. (Deduped; UTF-8 symbols kept.)
ELIGIBLE_TOKENS = frozenset({
    "ETH", "USDT", "USDC", "XRP", "TRX", "DOGE", "ZEC", "ADA", "LINK", "BCH", "DAI", "TON",
    "USD1", "USDe", "M", "LTC", "AVAX", "SHIB", "XAUt", "WLFI", "H", "DOT", "UNI", "ASTER",
    "DEXE", "USDD", "ETC", "AAVE", "ATOM", "U", "STABLE", "FIL", "INJ", "NIGHT", "FET", "TUSD",
    "BONK", "PENGU", "CAKE", "SIREN", "LUNC", "ZRO", "KITE", "FDUSD", "BEAT", "PIEVERSE", "BTT",
    "NFT", "EDGE", "FLOKI", "LDO", "B", "FF", "PENDLE", "NEX", "STG", "AXS", "TWT", "HOME", "RAY",
    "COMP", "GWEI", "XCN", "GENIUS", "XPL", "BAT", "SKYAI", "APE", "IP", "SFP", "TAG", "NXPC",
    "AB", "SAHARA", "1INCH", "CHEEMS", "BANANAS31", "RIVER", "MYX", "RAVE", "SNX", "FORM", "LAB",
    "HTX", "USDf", "CTM", "BDX", "SLX", "UB", "DUCKY", "FRAX", "BILL", "WFI", "KOGE", "ALE",
    "FRXUSD", "USDF", "GOMINING", "VCNT", "GUA", "DUSD", "SMILEK", "0G", "BEAM", "MY", "SOON",
    "REAL", "Q", "AIOZ", "ZIG", "YFI", "TAC", "lisUSD", "CYS", "ZAMA", "TRIA", "HUMA", "PLUME",
    "ZIL", "XPR", "ZETA", "BabyDoge", "NILA", "ROSE", "VELO", "UAI", "BRETT", "OPEN", "BSB",
    "TOSHI", "BAS", "ACH", "AXL", "LUR", "ELF", "KAVA", "APR", "IRYS", "EURI", "XUSD", "BARD",
    "DUSK", "SUSHI", "PEAQ", "COAI", "BDCA", "XAUM", "币安人生",
})

# SMT scores 8 pairs but only those listed on the contest count on BSC:
# {ETH, XRP, DOGE, ADA, LTC}. BTC/BNB/SOL stay signal-only here.
SMT_SYMBOLS = ("BTC", "ETH", "BNB", "SOL", "LTC", "XRP", "ADA", "DOGE")


def tradeable_symbols() -> list[str]:
    """SMT's signal universe ∩ the contest-eligible BEP-20 set."""
    return [s for s in SMT_SYMBOLS if s in ELIGIBLE_TOKENS]


@dataclass
class RiskGuardrails:
    """Hard limits enforced before any signing. Mirrors the contest's risk gate.

    ``drawdown_cap_pct`` is the DISQUALIFIER (the rules cite ~30%): if equity has
    fallen ≥ this far from its peak, the agent stops opening risk. The fee floor is
    SMT's own un-disableable check (net > fees). All caps are USD on capital deployed.
    """

    drawdown_cap_pct: float = 30.0
    max_per_trade_usd: float = 2_000.0
    max_daily_usd: float = 8_000.0
    max_slippage_bps: int = 50
    allowlist: frozenset = ELIGIBLE_TOKENS

    def check(
        self,
        plan: Any,
        *,
        risk_gate: Any,
        equity_usd: float,
        peak_equity_usd: float,
        deployed_today_usd: float = 0.0,
    ) -> "GuardrailResult":
        """Return a GuardrailResult(ok, reason, margin_usd). Never signs; never raises."""
        sym = _base_symbol(getattr(plan, "pair", ""))
        notional = float(getattr(plan, "qty", 0.0)) * float(getattr(plan, "entry_price", 0.0))
        margin = notional / max(1, int(getattr(plan, "leverage", 1) or 1))

        # 1. Fee floor — the ONE check no knob disables (delegated to SMT's RiskGate).
        if risk_gate is not None and not risk_gate.passes_fee_floor(plan):
            return GuardrailResult(False, "fee-floor: est_profit_net <= est_fees", margin)
        # 2. Token allowlist — trades outside the eligible set don't count → block.
        if sym not in self.allowlist:
            return GuardrailResult(False, f"token {sym} not in eligible BEP-20 allowlist", margin)
        # 3. Drawdown cap — the DQ gate.
        if peak_equity_usd > 0:
            dd = (peak_equity_usd - equity_usd) / peak_equity_usd * 100.0
            if dd >= self.drawdown_cap_pct:
                return GuardrailResult(False, f"drawdown {dd:.1f}% >= cap {self.drawdown_cap_pct:.0f}%", margin)
        # 4. Per-trade + daily caps (capital deployed).
        if margin > self.max_per_trade_usd:
            return GuardrailResult(False, f"per-trade ${margin:.0f} > cap ${self.max_per_trade_usd:.0f}", margin)
        if deployed_today_usd + margin > self.max_daily_usd:
            return GuardrailResult(False, f"daily ${deployed_today_usd + margin:.0f} > cap ${self.max_daily_usd:.0f}", margin)
        return GuardrailResult(True, "ok", margin)


@dataclass
class GuardrailResult:
    ok: bool
    reason: str
    margin_usd: float = 0.0


class TWAKExecutionAdapter:
    """Self-custodial BSC executor. Same surface as ExecutionClient.place/close.

    Offline (no signer) → returns ``{"executed": False, "reason": ...}`` and never
    hits the network. Live, ``signer`` is a Trust Wallet Agent Kit client that signs
    locally (autonomous mode) — keys stay with the user the whole way.
    """

    def __init__(
        self,
        risk_gate: Any = None,
        guardrails: Optional[RiskGuardrails] = None,
        signer: Any = None,
        equity_usd: float = 40_000.0,
    ):
        self.risk = risk_gate or _default_risk_gate()
        self.guardrails = guardrails or RiskGuardrails()
        self.signer = signer  # TWAK client; None → offline/dry-run
        self.equity_usd = equity_usd
        self.peak_equity_usd = equity_usd
        self.deployed_today_usd = 0.0

    def mark_equity(self, equity_usd: float) -> None:
        """Update equity + running peak (drives the drawdown guardrail)."""
        self.equity_usd = equity_usd
        self.peak_equity_usd = max(self.peak_equity_usd, equity_usd)

    def place(self, plan: Any) -> Dict[str, Any]:
        """Guardrail-check → (if signer) sign+send on BSC. Returns a result dict."""
        gr = self.guardrails.check(
            plan, risk_gate=self.risk, equity_usd=self.equity_usd,
            peak_equity_usd=self.peak_equity_usd, deployed_today_usd=self.deployed_today_usd,
        )
        if not gr.ok:
            log.info("[TWAK] BLOCKED %s %s — %s", getattr(plan, "pair", "?"),
                     getattr(plan, "direction", "?"), gr.reason)
            return {"executed": False, "reason": gr.reason, "guardrail": True}
        if self.signer is None:
            return {"executed": False, "reason": "TWAK signer not configured (offline dry-run)",
                    "would_execute": True, "margin_usd": gr.margin_usd}
        try:
            # TWAK autonomous mode: local self-custody signing on BSC. reduce_only=False = open.
            tx = self.signer.sign_and_send(
                symbol=_base_symbol(plan.pair), side=plan.direction,
                qty=plan.qty, reduce_only=False, max_slippage_bps=self.guardrails.max_slippage_bps,
            )
            self.deployed_today_usd += gr.margin_usd
            log.info("[TWAK] placed %s %s qty=%.6f → %s", plan.pair, plan.direction, plan.qty, tx)
            return {"executed": True, "tx": tx, "margin_usd": gr.margin_usd}
        except Exception as exc:  # noqa: BLE001 — surface, never crash the loop
            log.warning("[TWAK] place failed (%s)", exc)
            return {"executed": False, "reason": f"signer error: {exc}"}

    def close(self, symbol: str, side: str) -> Dict[str, Any]:
        """Reduce-only close on BSC (self-custody signed)."""
        if self.signer is None:
            return {"closed": False, "reason": "TWAK signer not configured (offline dry-run)"}
        try:
            tx = self.signer.sign_and_send(symbol=_base_symbol(symbol), side=side,
                                           reduce_only=True, max_slippage_bps=self.guardrails.max_slippage_bps)
            return {"closed": True, "tx": tx}
        except Exception as exc:  # noqa: BLE001
            log.warning("[TWAK] close failed (%s)", exc)
            return {"closed": False, "reason": f"signer error: {exc}"}


def _base_symbol(pair: str) -> str:
    """'ETHUSDT' / 'ETH-USDT' → 'ETH'."""
    s = (pair or "").upper().replace("-", "").replace("_", "")
    for quote in ("USDT", "USDC", "USD"):
        if s.endswith(quote) and len(s) > len(quote):
            return s[: -len(quote)]
    return s


def _default_risk_gate() -> Any:
    """SMT RiskGate when importable (fee floor); else a minimal net>fees stand-in."""
    try:
        from smt.core.risk import RiskGate
        return RiskGate()
    except Exception:  # noqa: BLE001 — keep the adapter usable standalone
        class _Floor:
            @staticmethod
            def passes_fee_floor(plan: Any) -> bool:
                return float(getattr(plan, "est_profit_net", 0.0)) > float(getattr(plan, "est_fees", 0.0))
        return _Floor()
