# Copyright (c) Microsoft. All rights reserved

"""SQLite-based state persistence for the job framework."""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
from typing import Any

from pydantic import BaseModel

from eval_recipes.benchmarking.job_framework.base import JobState, JobStatus


class JobStateStore:
    """
    SQLite-based persistent storage for job state.

    Provides atomic operations for updating job state and supports
    concurrent access through SQLite's built-in locking.

    State can be scoped by run_id to support multiple independent runs
    in the same database.
    """

    def __init__(self, db_path: Path, run_id: str | None = None) -> None:
        """
        Initialize the state store.

        Args:
            db_path: Path to the SQLite database file
            run_id: Optional run identifier to scope state. If provided, all
                   operations are scoped to this run, allowing multiple
                   independent runs in the same database.
        """
        self.db_path = db_path
        self.run_id = run_id or "default"
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create database schema if it doesn't exist."""
        with self._connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    run_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    outputs TEXT NOT NULL DEFAULT '{}',
                    error TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    PRIMARY KEY (run_id, job_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_run_status ON jobs(run_id, status)
            """)
            conn.commit()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _row_to_state(self, row: sqlite3.Row) -> JobState:
        """Convert a database row to a JobState object."""
        return JobState(
            job_id=row["job_id"],
            status=JobStatus(row["status"]),
            outputs=json.loads(row["outputs"]),
            error=row["error"],
            attempt_count=row["attempt_count"],
            created_at=datetime.fromisoformat(row["created_at"]),
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
        )

    def get(self, job_id: str) -> JobState | None:
        """
        Get the state of a job.

        Args:
            job_id: The job identifier

        Returns:
            JobState if found, None otherwise
        """
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM jobs WHERE run_id = ? AND job_id = ?",
                (self.run_id, job_id),
            )
            row = cursor.fetchone()
            return self._row_to_state(row) if row else None

    def get_all(self) -> list[JobState]:
        """
        Get all job states for the current run.

        Returns:
            List of all JobState objects
        """
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM jobs WHERE run_id = ? ORDER BY created_at",
                (self.run_id,),
            )
            return [self._row_to_state(row) for row in cursor.fetchall()]

    def get_by_status(self, status: JobStatus) -> list[JobState]:
        """
        Get all jobs with a specific status for the current run.

        Args:
            status: The status to filter by

        Returns:
            List of JobState objects with the given status
        """
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM jobs WHERE run_id = ? AND status = ? ORDER BY created_at",
                (self.run_id, status.value),
            )
            return [self._row_to_state(row) for row in cursor.fetchall()]

    def create(self, job_id: str) -> JobState:
        """
        Create a new job state entry.

        Args:
            job_id: The job identifier

        Returns:
            The created JobState

        Raises:
            ValueError: If job already exists in this run
        """
        now = datetime.now(UTC).isoformat()
        with self._connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO jobs (run_id, job_id, status, outputs, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (self.run_id, job_id, JobStatus.PENDING.value, "{}", now),
                )
                conn.commit()
            except sqlite3.IntegrityError as e:
                raise ValueError(f"Job {job_id} already exists in run {self.run_id}") from e
        return self.get(job_id)  # type: ignore[return-value]

    def create_or_get(self, job_id: str) -> JobState:
        """
        Get existing job state or create a new one.

        Args:
            job_id: The job identifier

        Returns:
            The existing or newly created JobState
        """
        existing = self.get(job_id)
        if existing:
            return existing
        return self.create(job_id)

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        outputs: dict[str, Any] | BaseModel | None = None,
        error: str | None = None,
    ) -> JobState:
        """
        Update the status of a job.

        Args:
            job_id: The job identifier
            status: New status
            outputs: Optional outputs to store (dict or Pydantic model)
            error: Optional error message

        Returns:
            Updated JobState

        Raises:
            ValueError: If job doesn't exist
        """
        now = datetime.now(UTC).isoformat()
        with self._connection() as conn:
            # Build update query dynamically
            updates = ["status = ?"]
            params: list[Any] = [status.value]

            if status == JobStatus.RUNNING:
                updates.append("started_at = ?")
                params.append(now)
                updates.append("attempt_count = attempt_count + 1")
            elif status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.SKIPPED):
                updates.append("completed_at = ?")
                params.append(now)

            if outputs is not None:
                updates.append("outputs = ?")
                if isinstance(outputs, BaseModel):
                    params.append(outputs.model_dump_json())
                else:
                    params.append(json.dumps(outputs))

            if error is not None:
                updates.append("error = ?")
                params.append(error)

            params.extend([self.run_id, job_id])
            query = f"UPDATE jobs SET {', '.join(updates)} WHERE run_id = ? AND job_id = ?"
            cursor = conn.execute(query, params)
            conn.commit()

            if cursor.rowcount == 0:
                raise ValueError(f"Job {job_id} not found in run {self.run_id}")

        return self.get(job_id)  # type: ignore[return-value]

    def reset_running_jobs(self) -> int:
        """
        Reset all running jobs back to pending for the current run.

        This is useful for resuming after a crash where jobs were
        left in running state.

        Returns:
            Number of jobs reset
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs
                SET status = ?, started_at = NULL
                WHERE run_id = ? AND status = ?
                """,
                (JobStatus.PENDING.value, self.run_id, JobStatus.RUNNING.value),
            )
            conn.commit()
            return cursor.rowcount

    def reset_failed_jobs(self) -> int:
        """
        Reset all failed jobs back to pending for retry in the current run.

        Returns:
            Number of jobs reset
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs
                SET status = ?, error = NULL, completed_at = NULL
                WHERE run_id = ? AND status = ?
                """,
                (JobStatus.PENDING.value, self.run_id, JobStatus.FAILED.value),
            )
            conn.commit()
            return cursor.rowcount

    def delete(self, job_id: str) -> bool:
        """
        Delete a job state entry.

        Args:
            job_id: The job identifier

        Returns:
            True if deleted, False if not found
        """
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM jobs WHERE run_id = ? AND job_id = ?",
                (self.run_id, job_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def clear(self) -> int:
        """
        Delete all job states for the current run.

        Returns:
            Number of jobs deleted
        """
        with self._connection() as conn:
            cursor = conn.execute("DELETE FROM jobs WHERE run_id = ?", (self.run_id,))
            conn.commit()
            return cursor.rowcount

    def list_runs(self) -> list[str]:
        """
        List all run IDs in the database.

        Returns:
            List of run IDs
        """
        with self._connection() as conn:
            cursor = conn.execute("SELECT DISTINCT run_id FROM jobs ORDER BY run_id")
            return [row["run_id"] for row in cursor.fetchall()]
