"""RiskGate — fee-floor (hard) + sizing (Session D learnable margin).

The fee floor (`est_profit_net > est_fees`) is the ONE check the
learner can never disable. Fees are real money on every flip;
without this floor the bot can churn small losses to death.

Sizing is initially a fixed-pct placeholder (Session B); it becomes
learnable in Session D via smt.learning.optimizer (TPE) inside
smt.learning.reward (net-fees + fat-tail bonus − overtrading penalty).

drawdown_guardian and gcp_budget_pause are EXTERNAL (in v4/); the
daemon polls them in its main loop. RiskGate only does per-plan
accept/reject + size.
"""

from __future__ import annotations
import logging
from typing import Optional

from smt.core.trade_plan import TradePlan

log = logging.getLogger("smt.risk")

# Round-trip taker fee (V5.0.7 baseline; see CLAUDE.md "Global constants").
ROUND_TRIP_FEE_PCT = 0.0012


DEFAULT_POSITION_PCT = 0.02


class RiskGate:
    def __init__(self, fee_pct: float = ROUND_TRIP_FEE_PCT,
                 position_pct: float = DEFAULT_POSITION_PCT):
        self.fee_pct = fee_pct
        # Session D made this learnable; the daemon (Session E) constructs the
        # gate with learned_params["position_pct"] when present, else the default.
        self.position_pct = position_pct

    def passes_fee_floor(self, plan: TradePlan) -> bool:
        """Hard floor: est_profit_net MUST exceed est_fees. No knob disables this."""
        return plan.est_profit_net > plan.est_fees

    def size_position(self, plan: TradePlan, equity_usd: float) -> Optional[TradePlan]:
        """Size position at `position_pct` of equity, capped by per-pair leverage.

        Returns None if notional falls below WEEX min-lot equivalent.
        """
        margin_usd = equity_usd * self.position_pct
        notional_usd = margin_usd * plan.leverage
        if plan.entry_price <= 0:
            log.warning("[RISK] size_position: entry_price=0 for %s — rejecting", plan.pair)
            return None
        qty = notional_usd / plan.entry_price
        if qty <= 0:
            log.warning("[RISK] size_position: qty<=0 for %s — rejecting", plan.pair)
            return None
        est_fees = notional_usd * self.fee_pct
        est_profit_net = notional_usd * (
            abs(plan.exit_target - plan.entry_price) / plan.entry_price
        ) - est_fees
        from dataclasses import replace
        sized = replace(
            plan,
            qty=qty,
            est_fees=est_fees,
            est_profit_net=est_profit_net,
        )
        if not self.passes_fee_floor(sized):
            log.info("[RISK] REJECT %s after sizing — net %.4f <= fees %.4f",
                     plan.pair, est_profit_net, est_fees)
            return None
        return sized

    def accept(self, plan: TradePlan, equity_usd: float) -> Optional[TradePlan]:
        """Run all gates. Return sized plan if accepted, None if rejected."""
        if not self.passes_fee_floor(plan):
            log.info(
                "[RISK] REJECT %s — net %.2f <= fees %.2f",
                plan.pair, plan.est_profit_net, plan.est_fees,
            )
            return None
        return self.size_position(plan, equity_usd)
