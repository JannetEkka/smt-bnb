# BNB Hack: AI Trading Agent — BUIDL submission (paste-ready)

> Tracks: **2 (Strategy Skill, do first)** + **1 (Autonomous Trading on BSC)**. Submit by 2026-06-21.

## Profile
- **BUIDL name:** `Smart Money Trading (SMT)`
- **Logo:** the 480×480 PNG.
- **Category:** AI / Robotics · **Is this an AI Agent?** **Yes**
- **GitHub:** `https://github.com/JannetEkka/smt-bnb` (public submission repo)
- **Project website:** `https://jannetekka.github.io/smt-bnb/`
- **Demo video:** *(optional — a 2-min agent walkthrough; add after submit)*
- **Social:** `https://x.com/JTechSMT` · LinkedIn

**Vision (≤256 chars):**
> AI trading agents waste weeks rebuilding data + execution plumbing. Smart Money Trading ships the brain: 6 AI personas + a learned Judge read CoinMarketCap data, decide, explain every call, and trade autonomously on BNB Chain via Trust Wallet.

## Details
**Smart Money Trading (SMT)** is a transparent, multi-persona AI trading agent. Six specialist
personas (order-flow, technical, on-chain/whale, sentiment, regime) + a learned **Judge** turn market
data into one auditable decision, with a ≤500-char "why" on every call.

- **Track 2 — Strategy Skill:** a **CoinMarketCap-data Skill** that turns market state into a
  backtestable strategy spec — our regime + multi-persona scoring authored as an LLM Skill.
- **Track 1 — Autonomous Trading:** the SMT brain reads CMC data, decides, and a **Trust Wallet
  Agent Kit** adapter signs + executes on **BSC**, fully self-custodial, inside hard risk guardrails:
  an **un-disableable fee floor**, a drawdown cap, per-trade + daily limits, and a token allowlist.

**Why it wins on the criteria:** real on-chain execution (BSC tx hashes), a genuinely hands-off
self-custodial loop (TWAK signing end-to-end), and a learning loop (Bayesian opt + a regime bandit
seeded from logged history) that tunes the engine from realized PnL. Transparency is the
differentiator — every trade ships with its reasoning.

## Team
**Jannet Akanksha Ekka** — Google Cloud AI Engineer; 4+ yrs enterprise software (Deloitte) → AI/ML
& agentic systems (Rank 1, 4.09/5 GPA, PG AI/ML). Built SMT v6.1 end-to-end: multi-persona engine,
learned Judge, from-scratch learning + validation stack, XAI layer. Fluent in the agent stack (MCP,
A2A/AP2, x402) + Vertex AI / BigQuery ML.

## Contact
`jtech26smt@gmail.com` · X `@JTechSMT` · LinkedIn (above)

## Submission notes
- Public repo (showcase); tuned params / per-pair research kept private (the edge).
- Track 1 on-chain proof = agent wallet address + BSC tx hash. Integration surfaces: CMC Agent Hub,
  Trust Wallet Agent Kit, BNB AI Agent SDK. See `integration_stub.py`.
