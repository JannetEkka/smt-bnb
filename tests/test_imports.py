"""Import smoke test — all smt.* modules must import without error.

Session A only requires the import surface to work. Subsequent sessions
fill bodies. Methods that raise NotImplementedError are FINE at import
time; this test never CALLS them.
"""

import importlib


SMT_MODULES = [
    "smt",
    "smt.daemon",
    "smt.core",
    "smt.core.trade_plan",
    "smt.core.execution",
    "smt.core.tracker",
    "smt.core.exit_cascade",
    "smt.core.risk",
    "smt.core.experience",
    "smt.context",
    "smt.context.global_context",
    "smt.context.fetchers",
    "smt.context.fetchers.dune",
    "smt.pairs",
    "smt.pairs.base",
    "smt.pairs.btc",
    "smt.pairs.eth",
    "smt.pairs.bnb",
    "smt.pairs.ltc",
    "smt.pairs.sol",
    "smt.pairs.xrp",
    "smt.pairs.ada",
    "smt.pairs.doge",
    "smt.personas",
    "smt.personas.base",
    "smt.personas.whale",
    "smt.personas.sentiment",
    "smt.personas.flow",
    "smt.personas.technical",
    "smt.personas.regime",
    "smt.personas.onchain",
    "smt.personas.judge",
    "smt.learning",
    "smt.learning.validation",
    "smt.learning.validation.cpcv",
    "smt.learning.validation.dsr",
    "smt.learning.validation.pbo",
    "smt.learning.validation.fdr",
    "smt.learning.validation.conformal",
    "smt.learning.validation.kde",
    "smt.learning.validation.gate",
    "smt.learning.optimizer",
    "smt.learning.bandit",
    "smt.learning.reward",
    "smt.learning.corpus",
    "smt.learning.hierarchical",
    "smt.learning.synthetic",
    "smt.learning.faithfulness",
    "smt.learning.groundtruth",
]


def test_all_modules_import():
    for mod_name in SMT_MODULES:
        importlib.import_module(mod_name)
