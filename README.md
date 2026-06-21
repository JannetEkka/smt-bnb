# Smart Money Trading (SMT) · BNB Hack: AI Trading Agent Edition

> **Read the market. Decide. Explain every call. Trade it on BNB Chain.** A *transparent*,
> multi-persona AI trading agent: six specialist personas (order-flow · technical · on-chain/whale ·
> sentiment · regime) + a learned **Judge** turn **CoinMarketCap** data into one **auditable**
> decision — with a plain-English "why" on every call — and a **Trust Wallet Agent Kit** adapter signs
> + executes it self-custodially on **BSC**, inside hard risk guardrails.

**The thesis:** every agent team rebuilds the same data + execution plumbing before writing a line of
real agent logic. SMT ships the *brain* — and its edge isn't a secret indicator, it's **a decision you
can audit**.

**Tracks (both):** **Track 1 — Autonomous Trading** (live PnL on BSC via TWAK) · **Track 2 — Strategy
Skills** (a backtestable CoinMarketCap Skill). Plus the **Best Use of TWAK** special prize.

---

## Why it wins on the criteria
| What judges score | What SMT does |
|---|---|
| **Technical execution / on-chain is real** | `SMTAgentRegistry` on BSC records every decision (`recordDecision`) + accrues a +2h/+4h accuracy reputation; TWAK signs trades self-custodially. Not cosmetic. |
| **Track 1 — live PnL inside a risk gate** | un-disableable **fee floor** + **30% drawdown DQ cap** + per-trade/daily caps + the contest's **eligible-token allowlist** + slippage — all enforced *before* any signing (`bnb/twak_adapter.py`). |
| **Track 2 — backtestable strategy** | `bnb/strategy_skill.py` emits a structured spec (entry/exit/risk + per-persona votes + a ≤500-char "why"), validated by **Deflated Sharpe / PBO / CPCV / conformal** (`smt/learning/validation/`). |
| **Originality / transparency** | white-box persona votes + a **counterfactual faithfulness check** (flip a vote → the decision must move, or the "why" is rejected). The agent explains *every* call — and every *wait*. |
| **Best Use of TWAK** | TWAK is the **sole** execution layer; keys never leave the user (self-custody end-to-end); autonomous signing inside the guardrails above. |

## Verify it yourself (no API keys needed)
```bash
pip install pytest
pytest -q                                # full suite green (personas, Judge, learning + validation gates)
python3 bnb/agent.py                      # offline demo: CMC → personas → Judge → Track-2 spec
                                          #               + Track-1 TWAK guardrail dry-run (no keys)
```
- Track 2 Skill: `bnb/strategy_skill.py` · Track 1 execution: `bnb/twak_adapter.py`
- CMC → context adapter: `bnb/cmc_adapter.py` (brain untouched) · runner: `bnb/agent.py`
- Architecture + Mermaid diagrams: **`docs/ARCHITECTURE.md`**

## Integrations
- **CoinMarketCap AI Agent Hub** — market data → the personas' context (MCP / x402 / CLI seam in `bnb/cmc_adapter.py`).
- **Trust Wallet Agent Kit (TWAK)** — self-custody local signing on BSC (`bnb/twak_adapter.py`, same `place/close` surface as the WEEX executor).
- **BNB AI Agent SDK** — the path to a live BSC agent; on-chain identity via `bnb/contracts/SMTAgentRegistry.sol`.

## On-chain (BSC)
- Contract: **`bnb/contracts/SMTAgentRegistry.sol`** — identity + reputation + **`recordDecision`** (the
  AI function callable on-chain) + `gradeDecision` (reputation from realized +2h/+4h accuracy). Same
  contract we ship on Mantle — BSC is EVM, so it ports unchanged. Deploy/verify runbook + the Python
  bridge (`bnb/onchain.py`) are in that folder.
- **Track-1 registration:** the agent wallet registers on the hackathon's BSC competition contract
  [`0x212c…aed5`](https://bsctrace.com/address/0x212c61b9b72c95d95bf29cf032f5e5635629aed5) before the
  Jun 22 trading window (`twak compete register` / MCP `competition_register`).
- Eligible BEP-20 set excludes BTC/BNB/SOL, so SMT's **tradeable** subset on BSC is **{ETH, XRP, DOGE,
  ADA, LTC}** (we still SCORE all 8; the allowlist guardrail blocks off-list orders).

## Live & links
- **Public dashboard (SMT World):** https://jannetekka.github.io/smt-bnb/
- **X:** https://x.com/JTechSMT · built solo by [@EkkaJanny96](https://x.com/EkkaJanny96)

## Track record & alpha boundary
SMT isn't a paper sketch — a live, multi-persona daemon ran in **real time from January to May 2026**,
making thousands of logged, executed decisions across 8 perpetual-futures pairs (every win *and* loss
recorded). That history is what the learning loop trains on and the validation gates guard against — we
treat honesty about drawdowns as a feature, not something to bury. The **architecture, learning, XAI,
and validation methodology are fully open** (this repo); the **tuned parameters and raw equity curve
stay private** — that's the edge. Judges verify from the **open method + the on-chain decision
reputation**, not an exposed ledger.
