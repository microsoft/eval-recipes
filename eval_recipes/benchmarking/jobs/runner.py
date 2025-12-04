# Copyright (c) Microsoft. All rights reserved

"""Job runner with DAG resolution and parallel execution."""

import asyncio
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeVar

from loguru import logger

from eval_recipes.benchmarking.jobs.base import Job, JobContext, JobResult, JobState, JobStatus
from eval_recipes.benchmarking.jobs.state import JobStateStore

T = TypeVar("T")


def _run_coro_in_new_loop(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run a coroutine in a new event loop (for use in threads).

    This allows async job.run() methods to be executed in a thread pool
    without blocking the main event loop.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class CyclicDependencyError(Exception):
    """Raised when a cycle is detected in job dependencies."""


class MissingDependencyError(Exception):
    """Raised when a job depends on a job that doesn't exist."""


class JobRunner:
    """
    Executes jobs respecting their dependencies with configurable parallelism.

    The runner:
    1. Builds a DAG from job dependency declarations
    2. Persists state to SQLite for resume capability
    3. Executes jobs in parallel when dependencies allow
    4. Supports configurable retry logic
    """

    def __init__(
        self,
        state_path: Path,
        max_parallel: int = 10,
        config: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> None:
        """
        Initialize the job runner.

        Args:
            state_path: Path to the SQLite database for state persistence
            max_parallel: Maximum number of jobs to run in parallel
            config: Optional configuration dict passed to all jobs via JobContext
            run_id: Optional run identifier. Each unique run_id gets independent
                   state, allowing you to run the same jobs fresh each day without
                   resuming from previous runs. If not provided, uses "default".
        """
        self.state_path = state_path
        self.max_parallel = max_parallel
        self.config = config or {}
        self.run_id = run_id or "default"
        self._store = JobStateStore(state_path, run_id=self.run_id)
        self._jobs: dict[str, Job] = {}
        self._completed_outputs: dict[str, dict[str, Any]] = {}

    def add_job(self, job: Job) -> None:
        """
        Register a job with the runner.

        Args:
            job: The job to register
        """
        self._jobs[job.job_id] = job

    def add_jobs(self, jobs: list[Job]) -> None:
        """
        Register multiple jobs with the runner.

        Args:
            jobs: List of jobs to register
        """
        for job in jobs:
            self.add_job(job)

    def _validate_dag(self) -> None:
        """
        Validate the job DAG has no cycles and all dependencies exist.

        Raises:
            CyclicDependencyError: If a cycle is detected
            MissingDependencyError: If a dependency doesn't exist
        """
        # Check all dependencies exist
        for job_id, job in self._jobs.items():
            for dep_id in job.dependencies:
                if dep_id not in self._jobs:
                    raise MissingDependencyError(f"Job '{job_id}' depends on '{dep_id}' which doesn't exist")

        # Detect cycles using DFS
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def has_cycle(job_id: str) -> bool:
            visited.add(job_id)
            rec_stack.add(job_id)

            job = self._jobs[job_id]
            for dep_id in job.dependencies:
                if dep_id not in visited:
                    if has_cycle(dep_id):
                        return True
                elif dep_id in rec_stack:
                    return True

            rec_stack.remove(job_id)
            return False

        for job_id in self._jobs:
            if job_id not in visited and has_cycle(job_id):
                raise CyclicDependencyError(f"Cycle detected involving job '{job_id}'")

    def _get_ready_jobs(self) -> list[Job]:
        """
        Get jobs that are ready to run (all dependencies satisfied).

        Returns:
            List of jobs ready for execution
        """
        ready = []
        for job_id, job in self._jobs.items():
            state = self._store.get(job_id)
            if state is None or state.status != JobStatus.PENDING:
                continue

            # Check all dependencies are completed or skipped
            deps_satisfied = True
            for dep_id in job.dependencies:
                dep_state = self._store.get(dep_id)
                if dep_state is None or dep_state.status not in (JobStatus.COMPLETED, JobStatus.SKIPPED):
                    deps_satisfied = False
                    break

            if deps_satisfied:
                ready.append(job)

        return ready

    def _build_context(self, job: Job) -> JobContext:
        """
        Build the execution context for a job.

        Args:
            job: The job to build context for

        Returns:
            JobContext with dependency outputs
        """
        dependency_outputs: dict[str, dict[str, Any]] = {}
        for dep_id in job.dependencies:
            if dep_id in self._completed_outputs:
                dependency_outputs[dep_id] = self._completed_outputs[dep_id]
            else:
                # Load from state store
                dep_state = self._store.get(dep_id)
                if dep_state:
                    dependency_outputs[dep_id] = dep_state.outputs
                    self._completed_outputs[dep_id] = dep_state.outputs

        return JobContext(dependency_outputs, self.config)

    async def _execute_job(self, job: Job, executor: ThreadPoolExecutor) -> JobResult:
        """
        Execute a single job with retry logic.

        Jobs are run in a thread pool to prevent blocking the event loop,
        ensuring true parallelism even with blocking I/O operations.

        Args:
            job: The job to execute
            executor: Thread pool executor to run the job in

        Returns:
            JobResult from the job execution
        """
        state = self._store.get(job.job_id)
        if state is None:
            state = self._store.create(job.job_id)

        context = self._build_context(job)

        # Execute with retries
        max_attempts = job.max_retries + 1
        last_error: str | None = None

        for attempt in range(max_attempts):
            if attempt > 0:
                logger.info(f"Retrying job {job.job_id} (attempt {attempt + 1}/{max_attempts})")

            # Mark as running
            self._store.update_status(job.job_id, JobStatus.RUNNING)

            try:
                # Run in thread pool to prevent blocking the event loop.
                # This ensures true parallelism even with blocking I/O
                # (file ops, subprocess calls, Docker, synchronous HTTP, etc.)
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(executor, _run_coro_in_new_loop, job.run(context))

                if result.status == JobStatus.COMPLETED:
                    self._store.update_status(job.job_id, JobStatus.COMPLETED, outputs=result.outputs, error=None)
                    self._completed_outputs[job.job_id] = result.outputs
                    logger.info(f"Job completed: {job.job_id}")
                    return result
                elif result.status == JobStatus.FAILED:
                    last_error = result.error
                    if attempt < max_attempts - 1:
                        # Will retry
                        self._store.update_status(job.job_id, JobStatus.PENDING, error=result.error)
                    else:
                        # Final failure
                        self._store.update_status(job.job_id, JobStatus.FAILED, error=result.error)
                        logger.error(f"Job failed: {job.job_id} - {result.error}")
                        return result
                else:
                    # SKIPPED or other status
                    self._store.update_status(job.job_id, result.status, outputs=result.outputs, error=result.error)
                    return result

            except Exception as e:
                last_error = str(e)
                logger.exception(f"Job {job.job_id} raised exception: {e}")
                if attempt < max_attempts - 1:
                    self._store.update_status(job.job_id, JobStatus.PENDING, error=last_error)
                else:
                    self._store.update_status(job.job_id, JobStatus.FAILED, error=last_error)

        return JobResult(status=JobStatus.FAILED, error=last_error)

    def _load_completed_outputs(self) -> None:
        """Load outputs from already completed jobs into memory."""
        completed_states = self._store.get_by_status(JobStatus.COMPLETED)
        for state in completed_states:
            self._completed_outputs[state.job_id] = state.outputs

        skipped_states = self._store.get_by_status(JobStatus.SKIPPED)
        for state in skipped_states:
            self._completed_outputs[state.job_id] = state.outputs

    async def run(self, reset_running: bool = True) -> dict[str, JobState]:
        """
        Execute all registered jobs respecting dependencies.

        Args:
            reset_running: If True, reset jobs left in RUNNING state to PENDING
                          (useful for resume after crash)

        Returns:
            Dictionary mapping job_id to final JobState
        """
        if not self._jobs:
            logger.warning("No jobs registered")
            return {}

        # Validate DAG before execution
        self._validate_dag()

        # Initialize job states in store
        for job_id in self._jobs:
            self._store.create_or_get(job_id)

        # Reset running jobs if requested (for resume after crash)
        if reset_running:
            reset_count = self._store.reset_running_jobs()
            if reset_count > 0:
                logger.info(f"Reset {reset_count} jobs from RUNNING to PENDING")

        # Load outputs from previously completed jobs
        self._load_completed_outputs()

        # Main execution loop with explicit thread pool for clean shutdown
        semaphore = asyncio.Semaphore(self.max_parallel)
        running_tasks: dict[str, asyncio.Task[JobResult]] = {}
        executor = ThreadPoolExecutor(max_workers=self.max_parallel)

        try:
            while True:
                # Get jobs ready to run
                ready_jobs = self._get_ready_jobs()

                # Start new jobs up to parallelism limit
                for job in ready_jobs:
                    if job.job_id in running_tasks:
                        continue

                    async def run_with_semaphore(j: Job, ex: ThreadPoolExecutor) -> JobResult:
                        async with semaphore:
                            return await self._execute_job(j, ex)

                    task = asyncio.create_task(run_with_semaphore(job, executor))
                    running_tasks[job.job_id] = task

                # If no tasks running and no ready jobs, we're done
                if not running_tasks:
                    break

                # Wait for at least one task to complete
                done, _ = await asyncio.wait(
                    running_tasks.values(),
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Remove completed tasks
                completed_job_ids = []
                for job_id, task in running_tasks.items():
                    if task in done:
                        completed_job_ids.append(job_id)
                for job_id in completed_job_ids:
                    del running_tasks[job_id]
        finally:
            # Properly shut down the thread pool to avoid cleanup warnings
            executor.shutdown(wait=True)

        # Return final states
        return {job_id: self._store.get(job_id) for job_id in self._jobs if self._store.get(job_id)}  # type: ignore[misc]

    def get_state(self, job_id: str) -> JobState | None:
        """
        Get the current state of a job.

        Args:
            job_id: The job identifier

        Returns:
            JobState if found, None otherwise
        """
        return self._store.get(job_id)

    def get_all_states(self) -> list[JobState]:
        """
        Get all job states.

        Returns:
            List of all JobState objects
        """
        return self._store.get_all()

    def reset_failed(self) -> int:
        """
        Reset all failed jobs to pending for retry.

        Returns:
            Number of jobs reset
        """
        return self._store.reset_failed_jobs()

    def clear_state(self) -> int:
        """
        Clear all job state (useful for fresh start).

        Returns:
            Number of jobs cleared
        """
        self._completed_outputs.clear()
        return self._store.clear()
