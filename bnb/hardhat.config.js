// Hardhat config — deploy + verify SMTAgentRegistry on BNB Smart Chain (BSC).
// SMTAgentRegistry.sol is the SAME contract deployed on Mantle — BSC is EVM, so it
// ports unchanged (CLAUDE.md contract #12). Verify chainId/RPC at docs.bnbchain.org.
require("@nomicfoundation/hardhat-toolbox");

const PK = process.env.BSC_PRIVATE_KEY; // set in your shell; never commit it

module.exports = {
  solidity: { version: "0.8.24", settings: { optimizer: { enabled: true, runs: 200 } } },
  networks: {
    bscTestnet: {
      url: process.env.BSC_RPC_URL || "https://data-seed-prebsc-1-s1.bnbchain.org:8545",
      chainId: 97,
      accounts: PK ? [PK] : [],
    },
    bsc: {
      url: "https://bsc-dataseed.bnbchain.org",
      chainId: 56,
      accounts: PK ? [PK] : [],
    },
  },
  // Etherscan V2 (multichain): ONE etherscan.io key verifies on every supported chain
  // (BSC, Mantle, … 60+) — routed by chainId. Use the operator's EXISTING key
  // (GCP secret `etherscan-api-key`), NOT a BscScan-specific key, and NO customChains
  // (CLAUDE.md lesson #12). In Cloud Shell:
  //   export ETHERSCAN_API_KEY=<your etherscan.io key>
  //   npx hardhat verify --network bscTestnet <deployed-address>
  etherscan: {
    apiKey: process.env.ETHERSCAN_API_KEY || "",
  },
};
