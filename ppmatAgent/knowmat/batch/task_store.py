"""SQLite-based persistent task state for batch processing."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ppmatAgent.knowmat.batch.models import TaskRecord, TaskStatus

_SCHEMA_VERSION = 1

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    pdf_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    ocr_vendor TEXT,
    ocr_job_id TEXT,
    md_path TEXT,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    api_key_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_ocr_job_id ON tasks(ocr_job_id);

CREATE TABLE IF NOT EXISTS schema_info (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


class TaskStore:
    """SQLite-backed task state manager.

    Thread-safe via check_same_thread=False. WAL mode for concurrent reads
    and crash safety. All mutations use explicit transactions.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(db_path),
            check_same_thread=False,
            isolation_level="DEFERRED",
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_CREATE_SQL)
        self._conn.execute(
            "INSERT OR REPLACE INTO schema_info VALUES (?, ?)",
            ("version", str(_SCHEMA_VERSION)),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Insert / Update
    # ------------------------------------------------------------------

    def bulk_insert_pending(
        self, pdf_paths: List[Path], max_retries: int = 3
    ) -> int:
        """Insert tasks for PDFs not already tracked. Returns count of new inserts."""
        now = time.time()
        inserted = 0
        with self._conn:
            for path in pdf_paths:
                task_id = path.stem
                try:
                    self._conn.execute(
                        """INSERT OR IGNORE INTO tasks
                           (task_id, pdf_path, status, max_retries, created_at, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (task_id, str(path), TaskStatus.PENDING.value, max_retries, now, now),
                    )
                    inserted += self._conn.execute(
                        "SELECT changes()"
                    ).fetchone()[0]
                except sqlite3.IntegrityError:
                    pass
        return inserted

    def update_status(self, task_id: str, status: TaskStatus, **kwargs: Any) -> None:
        """Update task status and optional extra fields atomically."""
        now = time.time()
        sets = ["status = ?", "updated_at = ?"]
        vals: list = [status.value, now]
        for col, val in kwargs.items():
            sets.append(f"{col} = ?")
            vals.append(val)
        vals.append(task_id)
        sql = f"UPDATE tasks SET {', '.join(sets)} WHERE task_id = ?"
        with self._conn:
            self._conn.execute(sql, vals)

    def mark_failed(self, task_id: str, error: str) -> None:
        """Mark task as FAILED with error message."""
        self.update_status(
            task_id, TaskStatus.FAILED, error_message=error
        )

    def increment_retry(self, task_id: str) -> bool:
        """Increment retry_count and reset to PENDING if under max. Returns True if retried."""
        with self._conn:
            row = self._conn.execute(
                "SELECT retry_count, max_retries FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if not row:
                return False
            if row["retry_count"] >= row["max_retries"]:
                return False
            now = time.time()
            self._conn.execute(
                """UPDATE tasks SET status = ?, retry_count = retry_count + 1,
                   error_message = NULL, updated_at = ? WHERE task_id = ?""",
                (TaskStatus.PENDING.value, now, task_id),
            )
        return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        row = self._conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
        return self._row_to_record(row) if row else None

    def get_tasks_by_status(
        self, status: TaskStatus, limit: int = 200
    ) -> List[TaskRecord]:
        rows = self._conn.execute(
            "SELECT * FROM tasks WHERE status = ? ORDER BY created_at LIMIT ?",
            (status.value, limit),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_all_submitted(self) -> List[TaskRecord]:
        """Get all OCR_SUBMITTED tasks (for polling/recovery)."""
        return self.get_tasks_by_status(TaskStatus.OCR_SUBMITTED, limit=10000)

    def get_retryable_failed(self) -> List[TaskRecord]:
        """Get FAILED tasks that can still be retried."""
        rows = self._conn.execute(
            """SELECT * FROM tasks
               WHERE status = ? AND retry_count < max_retries
               ORDER BY updated_at LIMIT 200""",
            (TaskStatus.FAILED.value,),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_statistics(self) -> Dict[str, int]:
        """Return count per status."""
        rows = self._conn.execute(
            "SELECT status, COUNT(*) as cnt FROM tasks GROUP BY status"
        ).fetchall()
        stats = {s.value: 0 for s in TaskStatus}
        for row in rows:
            stats[row["status"]] = row["cnt"]
        stats["total"] = sum(stats.values())
        return stats

    def reset_stuck_tasks(self) -> int:
        """Reset llm_processing tasks back to ocr_done.

        Tasks stuck in llm_processing indicate a prior crash mid-enrichment.
        They have valid OCR output and just need to be re-enriched.
        Returns the number of tasks reset.
        """
        cur = self._conn.execute(
            "UPDATE tasks SET status=?, updated_at=datetime('now') WHERE status=?",
            (TaskStatus.OCR_DONE.value, TaskStatus.LLM_PROCESSING.value),
        )
        self._conn.commit()
        return cur.rowcount

    def count_by_status(self, status: TaskStatus) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM tasks WHERE status = ?",
            (status.value,),
        ).fetchone()
        return row["cnt"] if row else 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _row_to_record(self, row: sqlite3.Row) -> TaskRecord:
        return TaskRecord(
            task_id=row["task_id"],
            pdf_path=row["pdf_path"],
            status=TaskStatus(row["status"]),
            ocr_vendor=row["ocr_vendor"],
            ocr_job_id=row["ocr_job_id"],
            md_path=row["md_path"],
            error_message=row["error_message"],
            retry_count=row["retry_count"],
            max_retries=row["max_retries"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            api_key_id=row["api_key_id"],
        )
