"""BNB Hack integration map — how the thin `bnb/` adapter wraps the SMT brain.

NOT imported by the main package or tests (hackathons/ is excluded from the build).
The real adapter now lives in sibling modules; this file is the one-screen manifest
of what reuses `smt.*` vs what's folder-local (CLAUDE.md contract #11).

Adapter modules (this folder):
  - cmc_adapter.py     → CoinMarketCap → the SMT context dicts (brain untouched)
  - strategy_skill.py  → Track 2: backtestable strategy spec from personas + Judge
  - twak_adapter.py    → Track 1: Trust-Wallet self-custody execution + risk guardrails
  - onchain.py         → BscBridge over SMTAgentRegistry.sol (identity + reputation)
  - agent.py           → runner tying them together (offline demo: `python3 agent.py`)

Reused UNCHANGED from the SMT brain:
  - smt.personas.{flow,technical,whale,onchain,sentiment,regime}  (the six voters)
  - smt.personas.judge.JudgePersona                                (aggregation + "why")
  - smt.core.risk.RiskGate                                         (un-disableable fee floor)
  - smt.core.trade_plan.TradePlan                                  (execution intent shape)
"""

from __future__ import annotations
from typing import Any, Dict


def demo() -> Dict[str, Any]:
    """Smallest end-to-end: synthetic CMC quote → context → Track-2 strategy spec."""
    import cmc_adapter
    import strategy_skill
    ctx = cmc_adapter.build_context(
        ["ETH"], quotes={"ETH": {"price": 3200.0, "percent_change_24h": 4.2, "volume_ratio": 1.3}},
        fear_greed=55,
    )
    return strategy_skill.generate_strategy_spec("ETH", ctx)


# Track 1 surface (mirrors smt.core.execution.ExecutionClient.place/close):
#   from twak_adapter import TWAKExecutionAdapter
#   ex = TWAKExecutionAdapter()          # offline dry-run (no signer) → guardrail-checked, no network
#   ex.place(plan)                       # self-custody sign on BSC when a TWAK signer is wired
#
# On-chain proof (optional):
#   from onchain import BscBridge
#   BscBridge().record_decision("ETHUSDT", "LONG", 0.72, "<=500-char why")
