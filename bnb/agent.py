"""SMT BNB trading agent — runner for both tracks (offline-safe demo).

Wires the pieces without touching the SMT brain:
  CMC data  →  cmc_adapter.build_context
            →  strategy_skill.generate_strategy_spec   (Track 2 deliverable)
            →  TWAKExecutionAdapter.place               (Track 1, only if a signer is set)
            →  BscBridge.record_decision                (optional on-chain proof)

Run the offline demo (no keys, no network):
    python3 bnb/agent.py            # (or hackathons/bnb-ai-trading-agent/agent.py in the main repo)

Personas degrade to NEUTRAL without data, so it always runs; pass a CMC key
(CMC_API_KEY) + a TWAK signer + BSC creds to go live.
"""

from __future__ import annotations
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

# Make both `smt` (one/two levels up) AND sibling modules importable, whether this
# file lives in hackathons/bnb-ai-trading-agent/ (main repo) or top-level bnb/ (submission).
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)                      # sibling modules: cmc_adapter, twak_adapter, …
for _cand in (os.path.join(_here, ".."), os.path.join(_here, "..", "..")):
    if os.path.isdir(os.path.join(_cand, "smt")):
        sys.path.insert(0, os.path.abspath(_cand))  # the SMT brain
        break

import cmc_adapter          # noqa: E402
import strategy_skill       # noqa: E402
from twak_adapter import TWAKExecutionAdapter, tradeable_symbols  # noqa: E402


def run_symbol(
    symbol: str,
    context: Dict[str, Any],
    *,
    executor: Optional[TWAKExecutionAdapter] = None,
    bridge: Any = None,
) -> Dict[str, Any]:
    """One symbol: spec (Track 2) + optional execution (Track 1) + optional on-chain record."""
    spec = strategy_skill.generate_strategy_spec(symbol, context)
    action = spec["signal"]["action"]

    out: Dict[str, Any] = {"symbol": symbol, "spec": spec, "executed": None, "tx": None}

    # Track 1: only attempt execution for an actionable, contest-eligible symbol.
    if executor is not None and action in ("LONG", "SHORT") and symbol in tradeable_symbols():
        plan = _spec_to_plan(spec, context)
        if plan is not None:
            out["executed"] = executor.place(plan)

    # Optional on-chain decision record (verifiable reputation).
    if bridge is not None and action in ("LONG", "SHORT"):
        out["tx"] = bridge.record_decision(symbol, action, spec["signal"]["confidence"], spec["reasoning"])

    return out


def _spec_to_plan(spec: Dict[str, Any], context: Dict[str, Any]) -> Any:
    """Turn a strategy spec into a minimal TradePlan for the guardrail/execution path.

    Uses the SMT TradePlan when importable; sizing/targets are illustrative (the
    production daemon owns real sizing). Returns None if there's no usable price.
    """
    sym = spec["symbol"]
    price = float((context.get("prices") or {}).get(sym, 0.0) or 0.0)
    if price <= 0:
        return None
    direction = spec["signal"]["action"]
    tp = price * (1.01 if direction == "LONG" else 0.99)
    sl = price * (0.99 if direction == "LONG" else 1.01)
    leverage = 10
    qty = (40_000.0 * 0.02 * leverage) / price        # 2% equity × leverage / price
    notional = qty * price
    est_fees = notional * 0.0012
    est_profit_net = notional * 0.01 - est_fees       # ~1% target move, net of fees
    try:
        from smt.core.trade_plan import TradePlan
        return TradePlan(
            pair=sym, lane=spec["signal"].get("lane", "fast"), direction=direction,
            entry_price=price, exit_target=tp, exit_stop=sl, hold_max=2.0, qty=qty,
            leverage=leverage, est_fees=est_fees, est_profit_net=est_profit_net,
            est_time_hours=1.0, decision_confidence=spec["signal"]["confidence"],
            reasoning=spec["reasoning"],
        )
    except Exception:  # noqa: BLE001 — keep a duck-typed fallback so the adapter still runs
        from types import SimpleNamespace
        return SimpleNamespace(pair=sym, direction=direction, entry_price=price,
                               qty=qty, leverage=leverage, est_fees=est_fees,
                               est_profit_net=est_profit_net)


def _demo_context() -> Dict[str, Any]:
    """Synthetic CMC snapshot (offline): ETH trending up, XRP mild, DOGE flat, fear=55."""
    quotes = {
        "ETH": {"price": 3200.0, "percent_change_24h": 4.2, "volume_ratio": 1.3},
        "XRP": {"price": 0.62, "percent_change_24h": 1.1, "volume_ratio": 1.0},
        "DOGE": {"price": 0.16, "percent_change_24h": -0.2, "volume_ratio": 0.8},
    }
    return cmc_adapter.build_context(["ETH", "XRP", "DOGE"], quotes=quotes, fear_greed=55)


def _strong_context() -> Dict[str, Any]:
    """Illustrative 'clean setup' (clearly synthetic): FLOW + TECHNICAL both fire LONG on
    ETH so the Judge clears its floor — shows the Track-1 guardrail/execution path end-to-end."""
    ctx = cmc_adapter.build_context(
        ["ETH"], quotes={"ETH": {"price": 3200.0, "percent_change_24h": 5.0, "volume_ratio": 1.5}},
        fear_greed=58)
    # Inject a strong TECHNICAL read (a real kline feed would produce this; hard-coded for the offline demo).
    ctx["technical_signal"] = {"ETH": {"direction": "LONG", "confidence": 0.8, "reasoning": "reclaimed range high"}}
    return ctx


def _demo() -> None:
    logging.basicConfig(level=logging.INFO)
    execu = TWAKExecutionAdapter()   # offline (no signer) → guardrail-checked dry-run, no network

    print("### Realistic CMC snapshot — SMT stays disciplined (WAIT logged with its reason) ###")
    ctx = _demo_context()
    results: List[Dict[str, Any]] = [run_symbol(s, ctx, executor=execu) for s in ("ETH", "XRP", "DOGE")]
    for r in results:
        sig = r["spec"]["signal"]
        print(f"\n=== {r['symbol']} ===")
        print(f"  signal : {sig['action']} (conf {sig['confidence']:.2f}, lane {sig['lane']})")
        print(f"  why    : {r['spec']['reasoning']}")
        print(f"  exec   : {r['executed']}")

    print("\n### Illustrative clean setup — Judge clears the floor → Track-1 guardrail dry-run ###")
    strong = run_symbol("ETH", _strong_context(), executor=execu)
    print(f"  signal : {strong['spec']['signal']['action']} (conf {strong['spec']['signal']['confidence']:.2f})")
    print(f"  why    : {strong['spec']['reasoning']}")
    print(f"  exec   : {strong['executed']}   # offline: passes guardrails, no signer → would_execute")

    print("\n--- full strategy spec (ETH, clean setup) ---")
    print(json.dumps(strong["spec"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _demo()
