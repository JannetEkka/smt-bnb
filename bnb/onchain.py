"""BSC on-chain bridge for SMT — BNB Hack: AI Trading Agent Edition.

Thin web3.py client over ``SMTAgentRegistry.sol`` (the SAME contract we deployed on
Mantle — BSC is EVM, so it ports unchanged). The SMT brain stays off-chain Python;
this writes each Judge decision (and its graded +2h/+4h outcome) on-chain so the
agent accrues a verifiable reputation, and it points at the hackathon's Track-1
competition registry on BSC.

Bare-container safe: web3 is imported LAZILY inside methods. With no web3 / no RPC /
no key, ``available()`` is False and the agent runs SIGNAL-ONLY — never blocks. The
pure encoders (pair/direction/conviction) + the agent-card builder need no web3 and
are unit-tested.

Verify (Etherscan V2, ONE key for every EVM chain incl. BSC — no per-explorer key,
no customChains; CLAUDE.md lesson #12):
    export ETHERSCAN_API_KEY=<etherscan.io key>   # GCP secret `etherscan-api-key`
    npx hardhat verify --network bscTestnet <address>
"""

from __future__ import annotations
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

log = logging.getLogger("smt.hackathons.bnb.onchain")

# BSC endpoints (web-verified 2026-06-20: chainId 97 testnet / 56 mainnet).
BSC_TESTNET = {"rpc": "https://data-seed-prebsc-1-s1.bnbchain.org:8545", "chain_id": 97,
               "explorer": "https://testnet.bscscan.com", "faucet": "https://www.bnbchain.org/en/testnet-faucet"}
BSC_MAINNET = {"rpc": "https://bsc-dataseed.bnbchain.org", "chain_id": 56,
               "explorer": "https://bscscan.com"}

# Track-1 live-trading registration contract (BSC) — the hackathon's own registry.
# Registering the agent wallet here before 2026-06-22 is how Track 1 entry works
# (`twak compete register` / MCP `competition_register` resolve to a tx against this).
COMPETITION_REGISTRY = "0x212c61b9b72c95d95bf29cf032f5e5635629aed5"


# ── pure encoders (match the Solidity types; no web3 needed) ───────────────────

def pair_to_bytes32(pair: str) -> bytes:
    """'BTCUSDT' → 32-byte right-padded ASCII (the contract's bytes32 pair field)."""
    b = (pair or "").upper().encode()[:32]
    return b + b"\x00" * (32 - len(b))


def direction_to_int(direction: str) -> int:
    """LONG → +1, SHORT → -1, anything else (WAIT/BLOCK/NEUTRAL) → 0."""
    d = (direction or "").upper()
    return 1 if d == "LONG" else -1 if d == "SHORT" else 0


def conviction_to_bps(confidence: float) -> int:
    """JUDGE confidence (0..1) → basis points (0..10000), clamped."""
    return max(0, min(10000, int(round(float(confidence or 0.0) * 10000))))


def build_agent_card(
    *,
    registry_address: Optional[str] = None,
    agent_id: Optional[int] = None,
    wallet: Optional[str] = None,
    network: str = "bscTestnet",
) -> Dict[str, Any]:
    """Agent card (identity + endpoints + on-chain registry pointer + agent wallet)."""
    return {
        "name": "Smart Money Trading (SMT)",
        "description": ("Transparent multi-persona AI trading agent: reads CoinMarketCap data, "
                        "decides via 6 personas + a learned Judge, explains every call, and trades "
                        "on BNB Chain via Trust Wallet — inside hard risk guardrails."),
        "version": "6.1.0",
        "agentType": "ai-trading-agent",
        "owner": "@EkkaJanny96",
        "skills": ["cmc-strategy-spec", "multi-persona-judge", "self-custody-execution", "explainable-signals"],
        "endpoints": {
            "dashboard": "https://jannetekka.github.io/smt-bnb/",
            "repo": "https://github.com/JannetEkka/smt-bnb",
            "wallet": wallet or "<agent-wallet-address>",
        },
        "integrations": {
            "data": "CoinMarketCap AI Agent Hub (MCP / x402 / CLI)",
            "execution": "Trust Wallet Agent Kit (self-custody local signing, BSC)",
            "sdk": "BNB AI Agent SDK",
        },
        "registry": {"chain": network, "contract": registry_address or "<deployed-address>",
                     "agentId": agent_id, "competition": COMPETITION_REGISTRY},
        "reputation": "on-chain correct/graded from +2h/+4h direction accuracy",
        "transparency": "white-box persona votes + counterfactual faithfulness check",
    }


@dataclass
class OnchainConfig:
    rpc_url: str = BSC_TESTNET["rpc"]
    chain_id: int = BSC_TESTNET["chain_id"]
    registry_address: Optional[str] = None
    private_key: Optional[str] = None

    @classmethod
    def from_env(cls) -> "OnchainConfig":
        """Read RPC / registry / key from env (BSC_RPC_URL, SMT_REGISTRY_ADDRESS, BSC_PRIVATE_KEY)."""
        return cls(
            rpc_url=os.getenv("BSC_RPC_URL", BSC_TESTNET["rpc"]),
            chain_id=int(os.getenv("BSC_CHAIN_ID", BSC_TESTNET["chain_id"])),
            registry_address=os.getenv("SMT_REGISTRY_ADDRESS"),
            private_key=os.getenv("BSC_PRIVATE_KEY"),
        )


# Minimal ABI — only the functions the bridge calls.
REGISTRY_ABI = [
    {"inputs": [{"name": "cardURI", "type": "string"}], "name": "registerAgent",
     "outputs": [{"name": "agentId", "type": "uint256"}], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [{"name": "pair", "type": "bytes32"}, {"name": "direction", "type": "int8"},
                {"name": "convictionBps", "type": "uint16"}, {"name": "reasoningHash", "type": "bytes32"}],
     "name": "recordDecision", "outputs": [{"name": "decisionId", "type": "uint256"}],
     "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [{"name": "decisionId", "type": "uint256"}, {"name": "correct", "type": "bool"}],
     "name": "gradeDecision", "outputs": [], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [{"name": "agentId", "type": "uint256"}], "name": "reputationBps",
     "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
]


class BscBridge:
    """web3.py client; degrades to a no-op (returns None) when web3/RPC/key absent."""

    def __init__(self, config: Optional[OnchainConfig] = None):
        self.config = config or OnchainConfig.from_env()
        self._w3 = None
        self._contract = None

    def _connect(self) -> bool:
        if self._w3 is not None:
            return True
        try:
            from web3 import Web3  # lazy: bare container has no web3
        except ImportError:
            log.warning("[BSC] web3 not installed — signal-only (no on-chain write)")
            return False
        if not (self.config.registry_address and self.config.private_key):
            log.warning("[BSC] missing registry address or key — signal-only")
            return False
        try:
            self._w3 = Web3(Web3.HTTPProvider(self.config.rpc_url))
            self._contract = self._w3.eth.contract(
                address=Web3.to_checksum_address(self.config.registry_address), abi=REGISTRY_ABI)
            return self._w3.is_connected()
        except Exception as exc:  # noqa: BLE001 — degrade, never block the agent
            log.warning("[BSC] connect failed (%s) — signal-only", exc)
            return False

    def available(self) -> bool:
        return self._connect()

    def _send(self, fn) -> Optional[str]:
        """Sign + send a contract call; return tx hash hex, or None on any failure."""
        try:
            from web3 import Web3
            acct = self._w3.eth.account.from_key(self.config.private_key)
            tx = fn.build_transaction({
                "from": acct.address,
                "nonce": self._w3.eth.get_transaction_count(acct.address),
                "chainId": self.config.chain_id,
            })
            signed = acct.sign_transaction(tx)
            tx_hash = self._w3.eth.send_raw_transaction(signed.raw_transaction)
            return Web3.to_hex(tx_hash)
        except Exception as exc:  # noqa: BLE001
            log.warning("[BSC] tx failed (%s) — skipped", exc)
            return None

    def record_decision(self, pair: str, direction: str, confidence: float, reasoning: str) -> Optional[str]:
        """Write one Judge decision on-chain. Returns tx hash, or None if unavailable."""
        if not self._connect():
            return None
        from web3 import Web3
        fn = self._contract.functions.recordDecision(
            pair_to_bytes32(pair), direction_to_int(direction),
            conviction_to_bps(confidence), Web3.keccak(text=reasoning or ""))
        tx = self._send(fn)
        if tx:
            log.info("[BSC] recordDecision %s %s conf=%.2f → %s", pair, direction, confidence, tx)
        return tx

    def grade_decision(self, decision_id: int, correct: bool) -> Optional[str]:
        if not self._connect():
            return None
        return self._send(self._contract.functions.gradeDecision(int(decision_id), bool(correct)))
