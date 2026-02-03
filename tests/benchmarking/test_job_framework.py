# Copyright (c) Microsoft. All rights reserved

"""Tests for the job framework: concurrency, DAG execution, resuming, and retries."""

import asyncio
from pathlib import Path
import time
from typing import Any, ClassVar

from pydantic import BaseModel
import pytest

from eval_recipes.benchmarking.job_framework.base import Job, JobContext, JobResult, JobStatus
from eval_recipes.benchmarking.job_framework.runner import CyclicDependencyError, JobRunner, MissingDependencyError
from eval_recipes.benchmarking.job_framework.state import JobStateStore


class SimpleOutput(BaseModel):
    """Simple output model for tests."""

    value: str


class MockJob(Job[SimpleOutput]):
    """Configurable test job with execution tracking."""

    output_model = SimpleOutput
    execution_order: ClassVar[list[str]] = []
    concurrent_count: ClassVar[int] = 0
    max_concurrent: ClassVar[int] = 0

    def __init__(
        self,
        job_id: str,
        deps: list["Job[Any]"] | None = None,
        soft_deps: list["Job[Any]"] | None = None,
        delay: float = 0,
        fail_until_attempt: int = 0,
        raise_exception: bool = False,
        max_retries_override: int | None = None,
    ) -> None:
        self._job_id = job_id
        self._deps = deps or []
        self._soft_deps = soft_deps or []
        self._delay = delay
        self._fail_until_attempt = fail_until_attempt
        self._raise_exception = raise_exception
        self._max_retries_override = max_retries_override
        self._attempt = 0

    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def dependencies(self) -> list["Job[Any]"]:
        return self._deps

    @property
    def soft_dependencies(self) -> list["Job[Any]"]:
        return self._soft_deps

    @property
    def max_retries(self) -> int:
        if self._max_retries_override is not None:
            return self._max_retries_override
        # Need fail_until_attempt retries to eventually succeed on attempt fail_until_attempt+1
        return self._fail_until_attempt

    async def run(self, context: JobContext) -> JobResult[SimpleOutput]:
        self._attempt += 1
        MockJob.execution_order.append(self._job_id)

        # Track concurrency
        MockJob.concurrent_count += 1
        MockJob.max_concurrent = max(MockJob.max_concurrent, MockJob.concurrent_count)

        try:
            if self._delay > 0:
                await asyncio.sleep(self._delay)

            if self._raise_exception:
                raise RuntimeError(f"Exception in {self._job_id}")

            if self._attempt <= self._fail_until_attempt:
                return JobResult(status=JobStatus.FAILED, error=f"fail attempt {self._attempt}")

            return JobResult(status=JobStatus.COMPLETED, output=SimpleOutput(value=f"result_{self._job_id}"))
        finally:
            MockJob.concurrent_count -= 1


@pytest.fixture(autouse=True)
def reset_test_job_state() -> None:
    """Reset MockJob class state before each test."""
    MockJob.execution_order = []
    MockJob.concurrent_count = 0
    MockJob.max_concurrent = 0


async def test_dag_execution_with_concurrency(tmp_path: Path) -> None:
    """Test DAG execution order and concurrency limits with a diamond DAG."""
    # Create diamond DAG: A → B, A → C, B → D, C → D
    job_a = MockJob("A", delay=0.05)
    job_b = MockJob("B", deps=[job_a], delay=0.05)
    job_c = MockJob("C", deps=[job_a], delay=0.05)
    job_d = MockJob("D", deps=[job_b, job_c], delay=0.05)

    runner = JobRunner(state_path=tmp_path / "jobs.db", max_parallel=2)
    runner.add_jobs([job_a, job_b, job_c, job_d])

    start = time.perf_counter()
    results = await runner.run()
    elapsed = time.perf_counter() - start

    # All jobs should complete
    assert all(state.status == JobStatus.COMPLETED for state in results.values())

    # A must execute first
    assert MockJob.execution_order[0] == "A"

    # D must execute last
    assert MockJob.execution_order[-1] == "D"

    # B and C should be in the middle (either order)
    middle = set(MockJob.execution_order[1:3])
    assert middle == {"B", "C"}

    # Total time should be ~0.15s (3 layers), not 0.2s (sequential)
    assert elapsed < 0.25, f"Expected ~0.15s, got {elapsed:.2f}s"

    # Verify D's dependencies were available
    d_state = results["D"]
    assert d_state.outputs["value"] == "result_D"


async def test_dag_validation_errors(tmp_path: Path) -> None:
    """Test all DAG validation error cases."""
    # Test 1: Missing dependency
    job_a = MockJob("A")
    job_b = MockJob("B", deps=[job_a])

    runner = JobRunner(state_path=tmp_path / "jobs1.db")
    runner.add_job(job_b)  # Only add B, not A

    with pytest.raises(MissingDependencyError, match="depends on 'A'"):
        await runner.run()

    # Test 2: Self-cycle (job depends on itself)
    class SelfDepJob(Job[SimpleOutput]):
        output_model = SimpleOutput

        @property
        def job_id(self) -> str:
            return "self"

        @property
        def dependencies(self) -> list[Job[Any]]:
            return [self]

        async def run(self, context: JobContext) -> JobResult[SimpleOutput]:
            return JobResult(status=JobStatus.COMPLETED, output=SimpleOutput(value="x"))

    runner2 = JobRunner(state_path=tmp_path / "jobs2.db")
    runner2.add_job(SelfDepJob())

    with pytest.raises(CyclicDependencyError):
        await runner2.run()

    # Test 3: Simple cycle A → B → A
    job_a3 = MockJob("A3")
    job_b3 = MockJob("B3", deps=[job_a3])
    # Manually create cycle by modifying deps
    job_a3._deps = [job_b3]  # type: ignore[list-item]

    runner3 = JobRunner(state_path=tmp_path / "jobs3.db")
    runner3.add_jobs([job_a3, job_b3])

    with pytest.raises(CyclicDependencyError):
        await runner3.run()

    # Test 4: Complex cycle A → B → C → A
    job_a4 = MockJob("A4")
    job_b4 = MockJob("B4", deps=[job_a4])
    job_c4 = MockJob("C4", deps=[job_b4])
    job_a4._deps = [job_c4]  # type: ignore[list-item]

    runner4 = JobRunner(state_path=tmp_path / "jobs4.db")
    runner4.add_jobs([job_a4, job_b4, job_c4])

    with pytest.raises(CyclicDependencyError):
        await runner4.run()


async def test_resume_after_crash(tmp_path: Path) -> None:
    """Test crash recovery and state persistence."""
    db_path = tmp_path / "jobs.db"

    # Create jobs: A → B, C (independent)
    job_a = MockJob("A")
    job_b = MockJob("B", deps=[job_a])
    job_c = MockJob("C")

    # First run - complete all
    runner1 = JobRunner(state_path=db_path, max_parallel=5)
    runner1.add_jobs([job_a, job_b, job_c])
    await runner1.run()

    initial_order = MockJob.execution_order.copy()
    assert len(initial_order) == 3
    MockJob.execution_order.clear()

    # Simulate crash: manually set C back to RUNNING in DB
    store = JobStateStore(db_path)
    store.update_status("C", JobStatus.RUNNING)

    # Create new runner (simulating restart after crash)
    job_a2 = MockJob("A")
    job_b2 = MockJob("B", deps=[job_a2])
    job_c2 = MockJob("C")

    runner2 = JobRunner(state_path=db_path, max_parallel=5)
    runner2.add_jobs([job_a2, job_b2, job_c2])
    results = await runner2.run()

    # Only C should have re-executed (was RUNNING, reset to PENDING)
    assert MockJob.execution_order == ["C"], f"Expected only C to run, got {MockJob.execution_order}"

    # All should be completed now
    assert all(state.status == JobStatus.COMPLETED for state in results.values())


async def test_run_id_isolation(tmp_path: Path) -> None:
    """Test that different run_ids have isolated state."""
    db_path = tmp_path / "jobs.db"

    # Run 1
    job_a1 = MockJob("A")
    job_b1 = MockJob("B")

    runner1 = JobRunner(state_path=db_path, run_id="run1", max_parallel=5)
    runner1.add_jobs([job_a1, job_b1])
    await runner1.run()

    assert set(MockJob.execution_order) == {"A", "B"}
    MockJob.execution_order.clear()

    # Run 2 with same job IDs but different run_id
    job_a2 = MockJob("A")
    job_b2 = MockJob("B")

    runner2 = JobRunner(state_path=db_path, run_id="run2", max_parallel=5)
    runner2.add_jobs([job_a2, job_b2])
    await runner2.run()

    # Both jobs should execute fresh in run2
    assert set(MockJob.execution_order) == {"A", "B"}

    # Verify both runs have independent state
    store1 = JobStateStore(db_path, run_id="run1")
    store2 = JobStateStore(db_path, run_id="run2")

    assert len(store1.get_all()) == 2
    assert len(store2.get_all()) == 2
    assert all(s.status == JobStatus.COMPLETED for s in store1.get_all())
    assert all(s.status == JobStatus.COMPLETED for s in store2.get_all())


async def test_retry_and_failure_handling(tmp_path: Path) -> None:
    """Test retry logic and exception handling."""
    # Job A: fails first 2 attempts, succeeds on 3rd (max_retries=2 by default)
    job_a = MockJob("A", fail_until_attempt=2)

    # Job B: always fails - fails on all attempts, max_retries=1 (2 total attempts)
    job_b = MockJob("B", fail_until_attempt=999, max_retries_override=1)

    # Job C: raises exception
    job_c = MockJob("C", raise_exception=True)

    runner = JobRunner(state_path=tmp_path / "jobs.db", max_parallel=5)
    runner.add_jobs([job_a, job_b, job_c])
    results = await runner.run()

    # A should complete after retries (3 total attempts)
    assert results["A"].status == JobStatus.COMPLETED
    assert MockJob.execution_order.count("A") == 3

    # B should fail after 2 attempts (1 initial + 1 retry)
    assert results["B"].status == JobStatus.FAILED
    assert MockJob.execution_order.count("B") == 2

    # C should fail with exception captured
    assert results["C"].status == JobStatus.FAILED
    assert "Exception in C" in (results["C"].error or "")


async def test_parallel_execution_limits(tmp_path: Path) -> None:
    """Test that max_parallel is strictly respected."""
    # Create 6 independent jobs with delay
    jobs = [MockJob(f"job_{i}", delay=0.1) for i in range(6)]

    runner = JobRunner(state_path=tmp_path / "jobs.db", max_parallel=2)
    runner.add_jobs(jobs)

    start = time.perf_counter()
    results = await runner.run()
    elapsed = time.perf_counter() - start

    # All jobs should complete
    assert all(state.status == JobStatus.COMPLETED for state in results.values())

    # Max concurrent should never exceed 2
    assert MockJob.max_concurrent <= 2, f"Max concurrent was {MockJob.max_concurrent}, expected <= 2"

    # Total time should be ~0.3s (3 batches of 2), not 0.6s (sequential)
    assert elapsed < 0.5, f"Expected ~0.3s, got {elapsed:.2f}s"
    assert elapsed >= 0.25, f"Expected at least 0.25s (3 batches), got {elapsed:.2f}s"


async def test_dependent_job_not_run_when_dependency_fails(tmp_path: Path) -> None:
    """Test that dependent jobs don't run when their dependency fails."""
    # Job A always fails (no retries)
    job_a = MockJob("A", fail_until_attempt=999, max_retries_override=0)

    # Job B depends on A
    job_b = MockJob("B", deps=[job_a])

    # Job C is independent (should still run)
    job_c = MockJob("C")

    runner = JobRunner(state_path=tmp_path / "jobs.db", max_parallel=5)
    runner.add_jobs([job_a, job_b, job_c])
    results = await runner.run()

    # A should fail
    assert results["A"].status == JobStatus.FAILED

    # B should remain PENDING (never ran because A failed)
    assert results["B"].status == JobStatus.PENDING
    assert "B" not in MockJob.execution_order

    # C should complete (independent of A)
    assert results["C"].status == JobStatus.COMPLETED
    assert "C" in MockJob.execution_order


async def test_soft_dependency_runs_when_dependency_fails(tmp_path: Path) -> None:
    """Test that jobs with soft dependencies run even when soft dependency fails."""
    # Job A always fails
    job_a = MockJob("A", fail_until_attempt=999, max_retries_override=0)

    # Job B has A as a soft dependency
    job_b = MockJob("B", soft_deps=[job_a])

    runner = JobRunner(state_path=tmp_path / "jobs.db", max_parallel=5)
    runner.add_jobs([job_a, job_b])
    results = await runner.run()

    # A should fail
    assert results["A"].status == JobStatus.FAILED

    # B should COMPLETE (soft dependency allows running even when A failed)
    assert results["B"].status == JobStatus.COMPLETED
    assert "B" in MockJob.execution_order


async def test_soft_dependency_waits_for_completion(tmp_path: Path) -> None:
    """Test that soft dependencies wait for dependent jobs to complete."""
    # Job A takes time
    job_a = MockJob("A", delay=0.1)

    # Job B has A as soft dependency
    job_b = MockJob("B", soft_deps=[job_a])

    runner = JobRunner(state_path=tmp_path / "jobs.db", max_parallel=5)
    runner.add_jobs([job_a, job_b])
    results = await runner.run()

    # Both complete, but A must run before B
    assert results["A"].status == JobStatus.COMPLETED
    assert results["B"].status == JobStatus.COMPLETED
    assert MockJob.execution_order.index("A") < MockJob.execution_order.index("B")


async def test_mixed_hard_and_soft_dependencies(tmp_path: Path) -> None:
    """Test jobs with both hard and soft dependencies."""
    # A succeeds, B fails, C has hard dep on A and soft dep on B
    job_a = MockJob("A")
    job_b = MockJob("B", fail_until_attempt=999, max_retries_override=0)
    job_c = MockJob("C", deps=[job_a], soft_deps=[job_b])

    runner = JobRunner(state_path=tmp_path / "jobs.db", max_parallel=5)
    runner.add_jobs([job_a, job_b, job_c])
    results = await runner.run()

    assert results["A"].status == JobStatus.COMPLETED
    assert results["B"].status == JobStatus.FAILED
    # C runs because hard dep (A) succeeded, even though soft dep (B) failed
    assert results["C"].status == JobStatus.COMPLETED
    assert "C" in MockJob.execution_order


async def test_hard_dependency_blocks_even_with_soft_deps(tmp_path: Path) -> None:
    """Test that hard dependency failure blocks job even if soft deps succeed."""
    # A fails, B succeeds, C has hard dep on A and soft dep on B
    job_a = MockJob("A", fail_until_attempt=999, max_retries_override=0)
    job_b = MockJob("B")
    job_c = MockJob("C", deps=[job_a], soft_deps=[job_b])

    runner = JobRunner(state_path=tmp_path / "jobs.db", max_parallel=5)
    runner.add_jobs([job_a, job_b, job_c])
    results = await runner.run()

    assert results["A"].status == JobStatus.FAILED
    assert results["B"].status == JobStatus.COMPLETED
    # C should NOT run because hard dep (A) failed
    assert results["C"].status == JobStatus.PENDING
    assert "C" not in MockJob.execution_order
