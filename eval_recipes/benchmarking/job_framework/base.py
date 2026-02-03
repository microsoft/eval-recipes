# Copyright (c) Microsoft. All rights reserved

"""Base classes and types for the job framework."""

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Generic, TypeVar, cast

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Status of a job in the execution pipeline."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# TypeVar for job output types, bound to BaseModel
TOutput = TypeVar("TOutput", bound=BaseModel)


class JobResult(BaseModel, Generic[TOutput]):
    """Result returned by a job after execution."""

    status: JobStatus
    output: TOutput | None = None
    error: str | None = None


class JobState(BaseModel):
    """Persisted state of a job."""

    job_id: str
    status: JobStatus = JobStatus.PENDING
    outputs: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    attempt_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None


class JobContext:
    """
    Context passed to jobs during execution.

    Provides typed access to outputs from dependency jobs and shared configuration.
    """

    def __init__(
        self,
        dependency_outputs: dict[str, dict[str, Any]],
        dependency_jobs: dict[str, "Job[Any]"],
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize job context.

        Args:
            dependency_outputs: Map of job_id -> outputs dict from completed dependencies
            dependency_jobs: Map of job_id -> Job object for typed access
            config: Optional shared configuration accessible to all jobs
        """
        self._dependency_outputs = dependency_outputs
        self._dependency_jobs = dependency_jobs
        self._config = config or {}

    def get_output(self, job: "Job[TOutput]") -> TOutput:
        """
        Get typed output from a dependency job.

        Args:
            job: The dependency job to get output from

        Returns:
            The typed output model parsed from stored dict
        """
        raw = self._dependency_outputs.get(job.job_id, {})
        return cast(TOutput, job.output_model(**raw))

    def try_get_output(self, job: "Job[TOutput]") -> TOutput | None:
        """
        Get typed output from a dependency job, returning None if unavailable.

        This is useful for soft dependencies where the job may have failed.
        If the job didn't complete successfully, its output dict will be empty
        and parsing will fail.

        Args:
            job: The dependency job to get output from

        Returns:
            The typed output model, or None if output is unavailable or invalid
        """
        raw = self._dependency_outputs.get(job.job_id, {})
        if not raw:
            return None
        try:
            return cast(TOutput, job.output_model(**raw))
        except Exception:
            return None

    @property
    def config(self) -> dict[str, Any]:
        """Access shared configuration."""
        return self._config


class Job(ABC, Generic[TOutput]):
    """
    Abstract base class for all jobs with typed output.

    Jobs are units of work that can declare dependencies on other jobs.
    The JobRunner uses these declarations to build a DAG and execute
    jobs in the correct order with proper parallelization.

    Example:
        class MyJob(Job[MyJobOutput]):
            output_model = MyJobOutput

            def __init__(self, name: str, dep_job: OtherJob):
                self._name = name
                self._dep_job = dep_job

            @property
            def job_id(self) -> str:
                return f"my_job:{self._name}"

            @property
            def dependencies(self) -> list[Job[Any]]:
                return [self._dep_job]

            async def run(self, context: JobContext) -> JobResult[MyJobOutput]:
                dep_output = context.get_output(self._dep_job)  # Typed!
                return JobResult(status=JobStatus.COMPLETED, output=MyJobOutput(...))
    """

    # Subclasses must set this to their output model class
    output_model: type[BaseModel]

    @property
    @abstractmethod
    def job_id(self) -> str:
        """
        Unique identifier for this job instance.

        The ID should be deterministic and based on the job's parameters
        so that the same job can be identified across runs for resume support.

        Returns:
            A unique string identifier
        """

    @property
    def dependencies(self) -> list["Job[Any]"]:
        """
        List of job objects that must complete before this job can run.

        Override this property to declare dependencies. The JobRunner will
        ensure all dependencies are satisfied before executing this job.

        Returns:
            List of Job objects this job depends on
        """
        return []

    @property
    def dependency_ids(self) -> list[str]:
        """
        Derived list of job IDs from dependencies.

        Used internally by JobRunner for DAG operations.

        Returns:
            List of job IDs this job depends on
        """
        return [dep.job_id for dep in self.dependencies]

    @property
    def soft_dependencies(self) -> list["Job[Any]"]:
        """
        List of jobs that must complete before this job runs, but can fail.

        Unlike hard dependencies, soft dependencies allow this job to run even
        if the dependency failed. The job just needs to wait for the soft
        dependency to reach a terminal state (completed, failed, or skipped).

        Useful for aggregation jobs that should run with whatever results are available.

        Returns:
            List of Job objects this job soft-depends on
        """
        return []

    @property
    def soft_dependency_ids(self) -> list[str]:
        """
        Derived list of job IDs from soft dependencies.

        Used internally by JobRunner for DAG operations.

        Returns:
            List of job IDs this job soft-depends on
        """
        return [dep.job_id for dep in self.soft_dependencies]

    @property
    def max_retries(self) -> int:
        """
        Maximum number of retry attempts for this job.

        Override to customize retry behavior. Default is 0 (no retries).

        Returns:
            Maximum retry count
        """
        return 0

    @abstractmethod
    async def run(self, context: JobContext) -> JobResult[TOutput]:
        """
        Execute the job.

        Args:
            context: JobContext providing typed access to dependency outputs

        Returns:
            JobResult with status, typed output, and optional error
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(job_id={self.job_id!r})"
