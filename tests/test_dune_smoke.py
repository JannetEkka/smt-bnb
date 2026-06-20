"""Dune client — mocked, network-free. Verifies the endpoint shapes match docs
(rows under result.rows; execute → execution_id; run_sql polls to completion) and
that any failure degrades gracefully to None (→ persona NEUTRAL).
"""

from __future__ import annotations
from unittest.mock import MagicMock

from smt.context.fetchers.dune import DuneClient


def _resp(status=200, payload=None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload or {}
    return r


def test_get_latest_result_parses_rows():
    rows = [{"token": "ETH", "netflow_usd": -1_200_000}, {"token": "BTC", "netflow_usd": 800_000}]
    sess = MagicMock()
    sess.get.return_value = _resp(200, {"state": "QUERY_STATE_COMPLETED", "result": {"rows": rows}})
    c = DuneClient(api_key="k", session=sess)
    out = c.get_latest_result(123456)
    assert out == rows
    # auth header sent
    assert "X-Dune-Api-Key" in sess.get.call_args.kwargs["headers"]


def test_execute_sql_returns_execution_id():
    sess = MagicMock()
    sess.post.return_value = _resp(200, {"execution_id": "abc-123", "state": "QUERY_STATE_PENDING"})
    c = DuneClient(api_key="k", session=sess)
    assert c.execute_sql("select 1") == "abc-123"


def test_run_sql_polls_to_completion():
    rows = [{"x": 1}]
    sess = MagicMock()
    sess.post.return_value = _resp(200, {"execution_id": "e1", "state": "QUERY_STATE_PENDING"})
    # status: pending → completed; then results
    sess.get.side_effect = [
        _resp(200, {"state": "QUERY_STATE_EXECUTING"}),
        _resp(200, {"state": "QUERY_STATE_COMPLETED"}),
        _resp(200, {"result": {"rows": rows}}),
    ]
    c = DuneClient(api_key="k", session=sess)
    assert c.run_sql("select 1", poll_interval=0) == rows


def test_failure_degrades_to_none():
    sess = MagicMock()
    sess.get.return_value = _resp(429, {})           # rate-limited
    c = DuneClient(api_key="k", session=sess)
    assert c.get_latest_result(1) is None             # → persona NEUTRAL, never a fabricated side
    sess.post.side_effect = RuntimeError("network down")
    assert c.execute_sql("select 1") is None
