"""Dune Analytics REST client — free tier; key in GCP secret `dune-analytics`.

Endpoints (verified against docs.dune.com 2026-06-15):
- Get Latest Query Result : GET  /api/v1/query/{query_id}/results   (cheap; reads last result)
- Execute SQL            : POST /api/v1/sql/execute                 (raw SQL → execution_id)
- Execute Query          : POST /api/v1/query/{query_id}/execute    (saved query → execution_id)
- Get Execution Status   : GET  /api/v1/execution/{id}/status
- Get Execution Result   : GET  /api/v1/execution/{id}/results
Auth header: `X-Dune-Api-Key`. Rows come back under result.rows (list of dicts).

The ONCHAIN persona's adapter (Session F/J) calls this, then maps rows →
`{direction, confidence}`. This client stays generic (returns rows) + graceful:
every method returns None on any failure so the persona degrades to NEUTRAL.
"""

from __future__ import annotations
import logging
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger("smt.context.fetchers.dune")

DUNE_BASE = "https://api.dune.com/api/v1"
# Verified vs docs.dune.com 2026-06-15. CANCELED has one L; PARTIAL still has rows.
_DONE = {"QUERY_STATE_COMPLETED", "QUERY_STATE_COMPLETED_PARTIAL"}
_FAIL = {"QUERY_STATE_FAILED", "QUERY_STATE_CANCELED", "QUERY_STATE_EXPIRED"}


def _load_key() -> Optional[str]:
    try:
        try:
            from v4.secrets_loader import get_secret
        except ImportError:
            from secrets_loader import get_secret  # type: ignore[no-redef]
        return get_secret("dune-analytics")
    except Exception as exc:
        log.warning("[DUNE] key not loaded (%s) — onchain persona will be NEUTRAL", exc)
        return None


class DuneClient:
    """Thin, graceful Dune REST wrapper. Inject `session` in tests (no network)."""

    def __init__(self, api_key: Optional[str] = None, timeout: int = 15, session: Any = None):
        self.api_key = api_key if api_key is not None else _load_key()
        self.timeout = timeout
        self._session = session

    def _sess(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    def _headers(self) -> Dict[str, str]:
        return {"X-Dune-Api-Key": self.api_key or "", "Content-Type": "application/json"}

    @staticmethod
    def _rows(payload: Optional[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        if not isinstance(payload, dict):
            return None
        result = payload.get("result")
        if isinstance(result, dict) and isinstance(result.get("rows"), list):
            return result["rows"]
        return None

    # ── reads ──────────────────────────────────────────────────────────────────

    def get_latest_result(self, query_id: int) -> Optional[List[Dict[str, Any]]]:
        """GET the latest stored result of a saved query (the cheap bot path)."""
        try:
            r = self._sess().get(f"{DUNE_BASE}/query/{query_id}/results",
                                  headers=self._headers(), timeout=self.timeout)
            if r.status_code != 200:
                log.warning("[DUNE] get_latest_result(%s) HTTP %s", query_id, r.status_code)
                return None
            return self._rows(r.json())
        except Exception as exc:
            log.warning("[DUNE] get_latest_result(%s) error: %s", query_id, exc)
            return None

    def execution_status(self, execution_id: str) -> Optional[str]:
        try:
            r = self._sess().get(f"{DUNE_BASE}/execution/{execution_id}/status",
                                 headers=self._headers(), timeout=self.timeout)
            return r.json().get("state") if r.status_code == 200 else None
        except Exception as exc:
            log.warning("[DUNE] execution_status(%s) error: %s", execution_id, exc)
            return None

    def execution_result(self, execution_id: str) -> Optional[List[Dict[str, Any]]]:
        try:
            r = self._sess().get(f"{DUNE_BASE}/execution/{execution_id}/results",
                                 headers=self._headers(), timeout=self.timeout)
            return self._rows(r.json()) if r.status_code == 200 else None
        except Exception as exc:
            log.warning("[DUNE] execution_result(%s) error: %s", execution_id, exc)
            return None

    # ── executions ──────────────────────────────────────────────────────────────

    def execute_sql(self, sql: str, performance: str = "medium") -> Optional[str]:
        """POST raw SQL → returns execution_id (poll status, then fetch result)."""
        return self._execute(f"{DUNE_BASE}/sql/execute", {"sql": sql, "performance": performance})

    def execute_query(self, query_id: int, query_parameters: Optional[Dict] = None) -> Optional[str]:
        """POST a saved query by id → returns execution_id."""
        body = {"query_parameters": query_parameters} if query_parameters else {}
        return self._execute(f"{DUNE_BASE}/query/{query_id}/execute", body)

    def _execute(self, url: str, body: Dict[str, Any]) -> Optional[str]:
        try:
            import json
            r = self._sess().post(url, headers=self._headers(),
                                  data=json.dumps(body), timeout=self.timeout)
            if r.status_code not in (200, 201):
                log.warning("[DUNE] execute HTTP %s (%s)", r.status_code, url)
                return None
            return r.json().get("execution_id")
        except Exception as exc:
            log.warning("[DUNE] execute error (%s): %s", url, exc)
            return None

    # ── convenience: execute → poll → result ─────────────────────────────────────

    def run_sql(self, sql: str, performance: str = "medium",
                poll_interval: float = 1.0, max_polls: int = 30) -> Optional[List[Dict[str, Any]]]:
        """Execute raw SQL and block until results (or None on failure/timeout)."""
        exec_id = self.execute_sql(sql, performance)
        if not exec_id:
            return None
        for _ in range(max_polls):
            state = self.execution_status(exec_id)
            if state in _DONE:                      # completed (incl. partial → has rows)
                return self.execution_result(exec_id)
            if state in _FAIL:                      # failed / canceled / expired
                log.warning("[DUNE] run_sql terminal state=%s", state)
                return None
            time.sleep(poll_interval)
        log.warning("[DUNE] run_sql timed out after %d polls", max_polls)
        return None
