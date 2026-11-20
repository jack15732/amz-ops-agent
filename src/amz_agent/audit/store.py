from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..models import AgentResponse, AuditTrace, ComplianceResult, ToolCallRecord


class AuditStore:
    """SQLite audit trail: every decision, tool call and LLM tag is durable."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._closed = False
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    endpoint TEXT NOT NULL,
                    site TEXT,
                    input TEXT,
                    output TEXT,
                    latency_ms REAL,
                    status TEXT,
                    created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS tool_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    seq INTEGER,
                    name TEXT,
                    arguments TEXT,
                    result TEXT,
                    latency_ms REAL,
                    error TEXT
                );
                CREATE TABLE IF NOT EXISTS compliance_flags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT,
                    site TEXT,
                    decision TEXT,
                    rule_ids TEXT,
                    llm_label TEXT,
                    confidence REAL,
                    text_hash TEXT,
                    created_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_traces_created ON traces(created_at);
                CREATE INDEX IF NOT EXISTS idx_flags_trace ON compliance_flags(trace_id);
                """
            )
            self._conn.commit()

    def record_trace(self, trace: AuditTrace) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO traces VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    trace.trace_id,
                    trace.endpoint,
                    trace.site,
                    _dumps(trace.input),
                    _dumps(trace.output),
                    trace.latency_ms,
                    trace.status,
                    trace.created_at or _now(),
                ),
            )
            self._conn.commit()
        for call in trace.tool_calls:
            self.record_tool_call(trace.trace_id, call)

    def record_tool_call(self, trace_id: str, call: ToolCallRecord) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO tool_calls (trace_id, seq, name, arguments, result, latency_ms, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    trace_id,
                    call.seq,
                    call.name,
                    _dumps(call.arguments),
                    _dumps(call.result),
                    call.latency_ms,
                    call.error,
                ),
            )
            self._conn.commit()

    def record_compliance(
        self, trace_id: str, result: ComplianceResult, text_hash: str = ""
    ) -> None:
        review = result.llm_review
        with self._lock:
            self._conn.execute(
                "INSERT INTO compliance_flags "
                "(trace_id, site, decision, rule_ids, llm_label, confidence, text_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    trace_id,
                    result.site.value,
                    result.decision.value,
                    json.dumps(sorted({m.rule_id for m in result.matches})),
                    review.label.value if review else None,
                    review.confidence if review else None,
                    text_hash,
                    _now(),
                ),
            )
            self._conn.commit()

    def record_agent(self, trace_id: str, request: Any, response: AgentResponse) -> None:
        self.record_trace(
            AuditTrace(
                trace_id=trace_id,
                endpoint="agent",
                site=response.site.value,
                input=request.model_dump(mode="json"),
                output=response.model_dump(mode="json"),
                latency_ms=response.latency_ms,
                created_at=_now(),
                tool_calls=response.tool_calls,
            )
        )

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
            if row is None:
                return None
            calls = self._conn.execute(
                "SELECT * FROM tool_calls WHERE trace_id = ? ORDER BY seq", (trace_id,)
            ).fetchall()
            flags = self._conn.execute(
                "SELECT * FROM compliance_flags WHERE trace_id = ? ORDER BY id", (trace_id,)
            ).fetchall()
        return {
            "trace_id": row["trace_id"],
            "endpoint": row["endpoint"],
            "site": row["site"],
            "input": _loads(row["input"]),
            "output": _loads(row["output"]),
            "latency_ms": row["latency_ms"],
            "status": row["status"],
            "created_at": row["created_at"],
            "tool_calls": [
                {
                    "seq": call["seq"],
                    "name": call["name"],
                    "arguments": _loads(call["arguments"]),
                    "result": _loads(call["result"]),
                    "latency_ms": call["latency_ms"],
                    "error": call["error"],
                }
                for call in calls
            ],
            "compliance_flags": [
                {
                    "decision": flag["decision"],
                    "rule_ids": _loads(flag["rule_ids"]),
                    "llm_label": flag["llm_label"],
                    "confidence": flag["confidence"],
                    "created_at": flag["created_at"],
                }
                for flag in flags
            ],
        }

    def recent_traces(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT trace_id, endpoint, site, latency_ms, status, created_at "
                "FROM traces ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            traces = self._conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
            blocked = self._conn.execute(
                "SELECT COUNT(*) FROM compliance_flags WHERE decision = 'blocked'"
            ).fetchone()[0]
            review = self._conn.execute(
                "SELECT COUNT(*) FROM compliance_flags WHERE decision = 'review'"
            ).fetchone()[0]
            llm_tags = self._conn.execute(
                "SELECT COUNT(*) FROM compliance_flags WHERE llm_label IS NOT NULL"
            ).fetchone()[0]
        return {
            "traces": traces,
            "blocked": blocked,
            "review": review,
            "llm_tags": llm_tags,
        }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _loads(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value
