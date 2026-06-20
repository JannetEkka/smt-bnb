"""Smoke tests for the BNB Hack adapter (hackathons/bnb-ai-trading-agent/).

The hackathon folder is hyphenated (not an importable package), so we load the
sibling modules by file path. Everything here is pure-Python + offline: the on-chain
bridge must DEGRADE (no web3/RPC/key → available() False), the Track-2 strategy spec
must build with no network, and the TWAK guardrails must reject on the fee floor,
the token allowlist, the drawdown cap, and the per-trade/daily caps.
"""

from __future__ import annotations
import importlib.util
import os
import sys
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The adapter lives in hackathons/bnb-ai-trading-agent/ (main repo) or top-level bnb/ (submission repo).
BNB = next((os.path.join(REPO, c) for c in ("bnb", "hackathons/bnb-ai-trading-agent")
            if os.path.isdir(os.path.join(REPO, c))), os.path.join(REPO, "hackathons", "bnb-ai-trading-agent"))
if BNB not in sys.path:
    sys.path.insert(0, BNB)   # let sibling-importing modules (agent.py) resolve cmc_adapter etc.


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(BNB, filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # dataclasses need the module registered before exec
    spec.loader.exec_module(mod)
    return mod


onchain = _load("smt_bnb_onchain", "onchain.py")
cmc_adapter = _load("cmc_adapter", "cmc_adapter.py")
strategy_skill = _load("strategy_skill", "strategy_skill.py")
twak = _load("twak_adapter", "twak_adapter.py")


def _plan(pair="ETHUSDT", entry=3200.0, qty=0.1, leverage=10, net=5.0, fees=1.0):
    return SimpleNamespace(pair=pair, direction="LONG", entry_price=entry, qty=qty,
                           leverage=leverage, est_profit_net=net, est_fees=fees)


# ── pure encoders match the Solidity types ────────────────────────────────────

def test_pair_to_bytes32_is_32_bytes_and_padded():
    b = onchain.pair_to_bytes32("ETHUSDT")
    assert len(b) == 32 and b.startswith(b"ETHUSDT") and b.endswith(b"\x00")


def test_direction_and_conviction_encoders():
    assert onchain.direction_to_int("LONG") == 1
    assert onchain.direction_to_int("SHORT") == -1
    assert onchain.direction_to_int("WAIT") == 0
    assert onchain.conviction_to_bps(0.55) == 5500
    assert onchain.conviction_to_bps(1.7) == 10000      # clamp high
    assert onchain.conviction_to_bps(-1.0) == 0         # clamp low


def test_agent_card_points_at_bnb_and_competition():
    card = onchain.build_agent_card(registry_address="0xABC", agent_id=1, wallet="0xWALLET")
    assert card["name"] and card["skills"] and card["transparency"]
    assert card["registry"]["contract"] == "0xABC" and card["registry"]["agentId"] == 1
    assert card["registry"]["competition"] == onchain.COMPETITION_REGISTRY
    assert card["endpoints"]["repo"].endswith("smt-bnb")
    assert "Trust Wallet" in card["integrations"]["execution"]


# ── on-chain bridge degrades gracefully (no web3 / no config) ──────────────────

def test_bridge_unavailable_without_web3_or_config():
    b = onchain.BscBridge(onchain.OnchainConfig(registry_address=None, private_key=None))
    assert b.available() is False
    assert b.record_decision("ETHUSDT", "LONG", 0.8, "why") is None   # never raises, returns None


# ── CMC adapter maps onto the SMT context schema (offline) ─────────────────────

def test_cmc_adapter_builds_context_schema():
    ctx = cmc_adapter.build_context(
        ["ETH", "DOGE"],
        quotes={"ETH": {"price": 3200.0, "percent_change_24h": 4.2, "volume_ratio": 1.3},
                "DOGE": {"price": 0.16, "percent_change_24h": -0.1}},
        fear_greed=55,
    )
    assert ctx["fear_greed"] == 55 and ctx["source"] == "coinmarketcap"
    assert ctx["prices"]["ETH"] == 3200.0
    assert ctx["regime"]["ETH"] == "TRENDING_UP"            # +4.2% → trending up
    assert ctx["flow_signal"]["ETH"]["direction"] == "LONG"  # momentum proxy votes LONG
    assert "DOGE" not in ctx["flow_signal"]                  # -0.1% too weak to vote


def test_cmc_adapter_offline_without_key_is_empty_not_raising():
    client = cmc_adapter.CMCClient(api_key=None)
    assert client.fetch_quotes(["ETH"]) == {}
    assert client.fetch_fear_greed() is None


# ── Track 2: strategy spec builds offline, is backtestable + auditable ─────────

def test_strategy_spec_offline_is_valid_and_reason_bounded():
    ctx = cmc_adapter.build_context(
        ["ETH"], quotes={"ETH": {"price": 3200.0, "percent_change_24h": 4.2, "volume_ratio": 1.3}},
        fear_greed=55)
    spec = strategy_skill.generate_strategy_spec("ETH", ctx)
    assert spec["skill"] == strategy_skill.SKILL_NAME and spec["symbol"] == "ETH"
    assert spec["signal"]["action"] in ("LONG", "SHORT", "WAIT", "BLOCK")
    assert 0.0 <= spec["signal"]["confidence"] <= 5.0       # weighted score, not necessarily <=1
    assert set(spec["persona_votes"]) == {"flow", "technical", "whale", "onchain", "sentiment", "regime"}
    assert len(spec["reasoning"]) <= 500 and "ETH" in spec["reasoning"]
    assert spec["risk"]["fee_floor"].startswith("net > fees")
    assert spec["backtest"]["horizons_h"] == [2, 4]


# ── Track 1: TWAK guardrails reject on every hard limit; offline never signs ────

def test_tradeable_subset_is_eligible_intersection():
    ts = twak.tradeable_symbols()
    assert "ETH" in ts and "XRP" in ts and "DOGE" in ts
    assert "BTC" not in ts and "BNB" not in ts and "SOL" not in ts   # not on the contest list


def test_guardrail_passes_clean_plan_but_blocks_each_violation():
    g = twak.RiskGuardrails()
    rg = twak._default_risk_gate()
    base = dict(risk_gate=rg, equity_usd=40_000.0, peak_equity_usd=40_000.0)

    assert g.check(_plan(), **base).ok is True

    # fee floor: net <= fees
    assert g.check(_plan(net=0.5, fees=1.0), **base).ok is False
    # token allowlist: BTC is not an eligible BEP-20 on the contest list
    assert g.check(_plan(pair="BTCUSDT"), **base).ok is False
    # drawdown cap: 32.5% down from peak ≥ 30% DQ gate
    assert g.check(_plan(), risk_gate=rg, equity_usd=27_000.0, peak_equity_usd=40_000.0).ok is False
    # per-trade cap: margin = qty*price/lev = 10*3200/1 = 32000 > 2000
    assert g.check(_plan(qty=10.0, leverage=1, net=100.0), **base).ok is False


def test_twak_offline_place_is_guardrail_checked_dry_run():
    ex = twak.TWAKExecutionAdapter()           # no signer → offline
    out = ex.place(_plan())
    assert out["executed"] is False and out.get("would_execute") is True   # passed guardrails, no signer
    blocked = ex.place(_plan(pair="BTCUSDT"))
    assert blocked["executed"] is False and blocked.get("guardrail") is True
