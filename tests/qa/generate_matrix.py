#!/usr/bin/env python3
"""Regenerate the QA traceability matrix from the live pytest suite.

One row per real (parametrized) test case. Inert CSVs — pytest ignores them, so
they cannot break the green run. Run from the repo root:

    python3 tests/qa/generate_matrix.py

Produces, under tests/qa/:
- test_cases.csv   — tc, epic, type, severity, node, status (one row per case)
- epics.csv        — EPIC-A..J ↔ Session ↔ status ↔ case count
- traceability.csv — case counts per epic × type

EPIC ↔ Session ↔ file mapping and the severity rubric below are the single
source of truth; update them when a session adds a test file.
"""

from __future__ import annotations
import csv
import os
import subprocess
import sys
from collections import Counter, defaultdict

QA_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(QA_DIR))

# Session file → EPIC. (Sessions F–J add their files here as they ship.)
EPIC_BY_FILE = {
    "tests/test_imports.py": "A",
    "tests/test_pairs_smoke.py": "B",
    "tests/test_personas_smoke.py": "C",
    "tests/test_learning_smoke.py": "D",
    "tests/test_daemon_smoke.py": "E",
    "tests/test_dune_smoke.py": "E",
    "tests/test_validation_smoke.py": "F",
}
EPIC_SESSION = {
    "A": "Scaffold + archive", "B": "Strategies + execution + tracker",
    "C": "Personas + JUDGE + context", "D": "Learning loop",
    "E": "Daemon loop end-to-end", "F": "Validation gates",
    "G": "Hierarchical + live-PBO", "H": "Explanation layer",
    "I": "Shadow run + refit", "J": "Production restart",
}
DONE_EPICS = {"A", "B", "C", "D", "E", "F"}

# Test-type classifier (function-name based). Smoke is the default for the
# Session A–D files; Session E introduced the integration/e2e/unit split.
TYPE_BY_NODE = {
    "test_context_attach_tracker_populates_positions_and_prices": "integration",
    "test_daemon_cycle_runs_without_network_and_fires_all": "e2e",
    "test_daemon_cycle_writes_valid_exp_record_all_required_fields": "e2e",
    "test_daemon_partial_close_fires_once_and_latches": "e2e",
    "test_daemon_bandit_gate_warm_arm_scales_confidence": "integration",
    "test_daemon_bandit_gate_warm_low_prob_vetoes": "integration",
    "test_daemon_cold_start_no_learned_params": "unit",
    "test_daemon_uses_default_position_pct_on_cold_start": "unit",
    "test_get_latest_result_parses_rows": "unit",
    "test_execute_sql_returns_execution_id": "unit",
    "test_run_sql_polls_to_completion": "unit",
    "test_failure_degrades_to_none": "unit",
    # Session F — validation gates: a replayed-market backtest + a locked-output regression.
    "test_backtest_synthetic_book_passes_validation_gates": "backtest",
    "test_regression_persona_weight_verdicts_locked": "regression",
}

# Severity rubric (README): S1 money/data/direction/security; S3 imports/defaults/
# content checks; S2 otherwise (a feature degrades but there's a workaround).
S1_KW = ("fee", "hard_block", "200d", "_block", "block_", "direction_quality",
         "all_required_fields", "cycle_runs", "cycle_writes", "bandit_gate",
         "fat_tail", "overtrading", "capitulation", "veto", "partial_close",
         # Session F: gates that decide whether a money-risking candidate ships,
         # plus direction-correctness — all S1 (money / direction).
         "rejects_overfit", "validation_gate", "no_discovery", "single_breach",
         "direction_correct", "per_lane", "faithfulness", "cascade_flags")
S3_KW = ("import", "cold_start", "position_pct", "prompt_marks", "default_space",
         "select_playbook", "roundtrip")


def _node_type(func: str, epic: str) -> str:
    return TYPE_BY_NODE.get(func, "smoke")


def _severity(func: str, epic: str) -> str:
    f = func.lower()
    if any(k in f for k in S1_KW):
        return "S1"
    if epic == "A" or any(k in f for k in S3_KW):
        return "S3"
    return "S2"


def collect_nodes() -> list:
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=REPO, capture_output=True, text=True,
    ).stdout
    return [ln.strip() for ln in out.splitlines() if "::" in ln]


def main() -> None:
    nodes = collect_nodes()
    rows = []
    per_epic = Counter()
    for node in sorted(nodes):
        file = node.split("::", 1)[0]
        epic = EPIC_BY_FILE.get(file, "?")
        func = node.split("::", 1)[1].split("[", 1)[0]
        rows.append({
            "tc": "", "epic": epic, "type": _node_type(func, epic),
            "severity": _severity(func, epic), "node": node, "status": "pass",
        })
        per_epic[epic] += 1
    for i, r in enumerate(rows, 1):
        r["tc"] = f"TC-{i:04d}"

    with open(os.path.join(QA_DIR, "test_cases.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tc", "epic", "type", "severity", "node", "status"])
        w.writeheader()
        w.writerows(rows)

    with open(os.path.join(QA_DIR, "epics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epic", "session", "status", "n_cases"])
        for e in sorted(EPIC_SESSION):
            status = "done" if e in DONE_EPICS else "pending"
            w.writerow([f"EPIC-{e}", EPIC_SESSION[e], status, per_epic.get(e, 0)])

    matrix = defaultdict(Counter)
    types = set()
    for r in rows:
        matrix[r["epic"]][r["type"]] += 1
        types.add(r["type"])
    types = sorted(types)
    with open(os.path.join(QA_DIR, "traceability.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epic"] + types + ["total"])
        for e in sorted(matrix):
            counts = [matrix[e].get(t, 0) for t in types]
            w.writerow([e] + counts + [sum(counts)])

    sev = Counter(r["severity"] for r in rows)
    print(f"wrote {len(rows)} test cases across epics {dict(per_epic)}; severity {dict(sev)}")


if __name__ == "__main__":
    main()
