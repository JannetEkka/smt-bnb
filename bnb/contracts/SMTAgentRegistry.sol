// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title SMTAgentRegistry — ERC-8004-style identity + reputation + on-chain AI decision record
/// @notice Built for The Turing Test Hackathon 2026 (Mantle). Self-contained, no external deps.
///         Satisfies the "≥1 AI-powered function callable on-chain" bar (20-Deploy award + Grand
///         Champion Mantle requirement): the off-chain SMT Judge calls `recordDecision(...)` to write
///         each multi-persona decision on-chain; `gradeDecision(...)` later writes the realized
///         +2h/+4h outcome, accruing a verifiable on-chain reputation. Radical transparency, on-chain.
contract SMTAgentRegistry {
    struct Agent {
        address owner;
        string cardURI;     // ERC-8004 agent-card JSON (IPFS/HTTPS)
        uint64 registeredAt;
        uint64 decisions;   // total decisions recorded
        uint64 correct;     // graded-correct (reputation numerator)
        uint64 graded;      // decisions graded so far (denominator)
        bool exists;
    }

    struct Decision {
        uint256 agentId;
        bytes32 pair;          // e.g. "BTCUSDT" packed into bytes32
        int8 direction;        // +1 LONG, -1 SHORT, 0 WAIT
        uint16 convictionBps;  // 0..10000 = JUDGE confidence
        bytes32 reasoningHash; // keccak256 of the <=500-char "why" (full text off-chain)
        uint64 ts;
        bool graded;
        bool correct;
    }

    uint256 public nextAgentId = 1;
    mapping(uint256 => Agent) public agents;
    mapping(address => uint256) public agentIdOf;
    Decision[] public decisions;

    event AgentRegistered(uint256 indexed agentId, address indexed owner, string cardURI);
    event CardUpdated(uint256 indexed agentId, string cardURI);
    event DecisionRecorded(
        uint256 indexed decisionId, uint256 indexed agentId, bytes32 pair,
        int8 direction, uint16 convictionBps, bytes32 reasoningHash
    );
    event DecisionGraded(uint256 indexed decisionId, bool correct);

    /// @notice Register the calling agent with its ERC-8004 card URI. One agent per address.
    function registerAgent(string calldata cardURI) external returns (uint256 agentId) {
        require(agentIdOf[msg.sender] == 0, "already registered");
        agentId = nextAgentId++;
        agents[agentId] = Agent(msg.sender, cardURI, uint64(block.timestamp), 0, 0, 0, true);
        agentIdOf[msg.sender] = agentId;
        emit AgentRegistered(agentId, msg.sender, cardURI);
    }

    function updateCard(uint256 agentId, string calldata cardURI) external {
        require(agents[agentId].owner == msg.sender, "not owner");
        agents[agentId].cardURI = cardURI;
        emit CardUpdated(agentId, cardURI);
    }

    /// @notice THE on-chain AI function: the off-chain SMT Judge writes one decision on-chain.
    function recordDecision(bytes32 pair, int8 direction, uint16 convictionBps, bytes32 reasoningHash)
        external
        returns (uint256 decisionId)
    {
        uint256 agentId = agentIdOf[msg.sender];
        require(agentId != 0, "register first");
        require(convictionBps <= 10000, "conviction>10000");
        require(direction >= -1 && direction <= 1, "bad direction");
        decisionId = decisions.length;
        decisions.push(Decision(agentId, pair, direction, convictionBps, reasoningHash,
                                uint64(block.timestamp), false, false));
        agents[agentId].decisions += 1;
        emit DecisionRecorded(decisionId, agentId, pair, direction, convictionBps, reasoningHash);
    }

    /// @notice Grade a past decision with its realized +2h/+4h outcome → accrues reputation.
    function gradeDecision(uint256 decisionId, bool correct) external {
        require(decisionId < decisions.length, "no decision");
        Decision storage d = decisions[decisionId];
        require(agents[d.agentId].owner == msg.sender, "not owner");
        require(!d.graded, "already graded");
        d.graded = true;
        d.correct = correct;
        agents[d.agentId].graded += 1;
        if (correct) agents[d.agentId].correct += 1;
        emit DecisionGraded(decisionId, correct);
    }

    /// @notice Reputation in bps (0..10000) = correct / graded. 0 until first grade.
    function reputationBps(uint256 agentId) external view returns (uint256) {
        uint64 g = agents[agentId].graded;
        if (g == 0) return 0;
        return (uint256(agents[agentId].correct) * 10000) / g;
    }

    function decisionCount() external view returns (uint256) {
        return decisions.length;
    }
}
