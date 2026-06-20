"""SMT AIQuant Bot — multi-persona quant trading + AI learning loop.

Thin daemon orchestrator: `smt.daemon`. Per-pair strategies:
`smt.pairs.*`. Persona vote sources (consumed by JUDGE):
`smt.personas.*`. Execution / tracker / risk: `smt.core.*`. Learning
stack (overfit gates / optimizer / conformal / hierarchical):
`smt.learning.*`. Per-cycle market+regime snapshot: `smt.context.*`.

Version: bumped in v4/smt_settings.json (single source of truth).
"""

__version__ = "6.1.0"
