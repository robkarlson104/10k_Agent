"""
audit.py — Full auditability for the 10-K agent.

Every agent session, tool call, and raw database query is written to three
tables in the same tenk_rag Postgres database.

  audit_sessions   — one row per question asked
  audit_events     — one row per tool call (input + output + duration)
  audit_db_queries — one row per SQL query executed inside a tool
"""

import json
import time
from contextvars import ContextVar
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from db import get_connection

# Carries the active session id into tool functions without threading it through
# every function signature. ContextVar is safe under async and threaded execution.
_session_id_var: ContextVar[int | None] = ContextVar("audit_session_id", default=None)


def init_audit_tables() -> None:
    """Create audit tables if they don't already exist."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS audit_sessions (
                    id           SERIAL PRIMARY KEY,
                    thread_id    TEXT        NOT NULL,
                    question     TEXT        NOT NULL,
                    final_answer TEXT,
                    started_at   TIMESTAMPTZ DEFAULT NOW(),
                    completed_at TIMESTAMPTZ
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    id          SERIAL PRIMARY KEY,
                    session_id  INTEGER REFERENCES audit_sessions(id),
                    tool_name   TEXT        NOT NULL,
                    input       TEXT,
                    output      TEXT,
                    error       TEXT,
                    duration_ms INTEGER,
                    created_at  TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS audit_db_queries (
                    id            SERIAL PRIMARY KEY,
                    session_id    INTEGER REFERENCES audit_sessions(id),
                    tool_name     TEXT,
                    sql           TEXT        NOT NULL,
                    params        TEXT,
                    rows_returned INTEGER,
                    duration_ms   INTEGER,
                    created_at    TIMESTAMPTZ DEFAULT NOW()
                )
            """)
        conn.commit()


def create_session(thread_id: str, question: str) -> int:
    """Insert a session row and set the context var. Returns the new session id."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO audit_sessions (thread_id, question) VALUES (%s, %s) RETURNING id",
                (thread_id, question),
            )
            session_id: int = cur.fetchone()[0]
        conn.commit()
    _session_id_var.set(session_id)
    return session_id


def close_session(session_id: int, final_answer: str) -> None:
    """Stamp the session row with the final answer and completion time."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE audit_sessions
                SET final_answer = %s, completed_at = NOW()
                WHERE id = %s
                """,
                (final_answer, session_id),
            )
        conn.commit()


def log_db_query(
    sql: str,
    params: Any,
    rows_returned: int,
    tool_name: str = "",
    duration_ms: int = 0,
) -> None:
    """
    Write one row to audit_db_queries for a single SQL execution.

    params should be a human-readable dict — do not pass the raw embedding vector.
    Silently no-ops if no session is active or if the write fails, so audit
    failures never crash the agent.
    """
    session_id = _session_id_var.get()
    if session_id is None:
        return
    try:
        params_str = json.dumps(params) if isinstance(params, dict) else str(params)
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO audit_db_queries
                        (session_id, tool_name, sql, params, rows_returned, duration_ms)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (session_id, tool_name, sql.strip(), params_str, rows_returned, duration_ms),
                )
            conn.commit()
    except Exception:
        pass


class AuditCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler that logs every tool call to audit_events.

    on_tool_start inserts a row and stores the run_id → event_id mapping.
    on_tool_end / on_tool_error update that row with output and duration.
    All DB writes are wrapped in try/except so audit failures never surface
    to the user.
    """

    def __init__(self, session_id: int) -> None:
        self.session_id = session_id
        self._pending: dict[str, tuple[int, float]] = {}

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        tool_name = serialized.get("name", "unknown")
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO audit_events (session_id, tool_name, input)
                        VALUES (%s, %s, %s) RETURNING id
                        """,
                        (self.session_id, tool_name, input_str),
                    )
                    event_id: int = cur.fetchone()[0]
                conn.commit()
            self._pending[str(run_id)] = (event_id, time.perf_counter())
        except Exception:
            pass

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        pending = self._pending.pop(str(run_id), None)
        if pending is None:
            return
        event_id, t0 = pending
        duration_ms = int((time.perf_counter() - t0) * 1000)
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE audit_events
                        SET output = %s, duration_ms = %s
                        WHERE id = %s
                        """,
                        (str(output), duration_ms, event_id),
                    )
                conn.commit()
        except Exception:
            pass

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        pending = self._pending.pop(str(run_id), None)
        if pending is None:
            return
        event_id, t0 = pending
        duration_ms = int((time.perf_counter() - t0) * 1000)
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE audit_events
                        SET error = %s, duration_ms = %s
                        WHERE id = %s
                        """,
                        (str(error), duration_ms, event_id),
                    )
                conn.commit()
        except Exception:
            pass
