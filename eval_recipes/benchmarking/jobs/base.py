# Copyright (c) Microsoft. All rights reserved

"""Base classes and types for the job framework."""

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Status of a job in the execution pipeline."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class JobResult(BaseModel):
    """Result returned by a job after execution."""

    status: JobStatus
    outputs: dict[str, Any] = Field(default_factory=dict)
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

    Provides access to outputs from dependency jobs and shared configuration.
    """

    def __init__(self, dependency_outputs: dict[str, dict[str, Any]], config: dict[str, Any] | None = None) -> None:
        """
        Initialize job context.

        Args:
            dependency_outputs: Map of job_id -> outputs dict from completed dependencies
            config: Optional shared configuration accessible to all jobs
        """
        self._dependency_outputs = dependency_outputs
        self._config = config or {}

    def get_output(self, job_id: str, output_name: str, default: Any = None) -> Any:
        """
        Get a specific output from a dependency job.

        Args:
            job_id: ID of the dependency job
            output_name: Name of the output to retrieve
            default: Default value if output not found

        Returns:
            The output value or default
        """
        job_outputs = self._dependency_outputs.get(job_id, {})
        return job_outputs.get(output_name, default)

    def get_all_outputs(self, job_id: str) -> dict[str, Any]:
        """
        Get all outputs from a dependency job.

        Args:
            job_id: ID of the dependency job

        Returns:
            Dictionary of all outputs from the job
        """
        return self._dependency_outputs.get(job_id, {})

    @property
    def config(self) -> dict[str, Any]:
        """Access shared configuration."""
        return self._config


class Job(ABC):
    """
    Abstract base class for all jobs.

    Jobs are units of work that can declare dependencies on other jobs.
    The JobRunner uses these declarations to build a DAG and execute
    jobs in the correct order with proper parallelization.

    Example:
        class MyJob(Job):
            def __init__(self, name: str, deps: list[str]):
                self._name = name
                self._deps = deps

            @property
            def job_id(self) -> str:
                return f"my_job:{self._name}"

            @property
            def dependencies(self) -> list[str]:
                return self._deps

            async def run(self, context: JobContext) -> JobResult:
                # Do work here
                return JobResult(status=JobStatus.COMPLETED, outputs={"result": 42})
    """

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
    def dependencies(self) -> list[str]:
        """
        List of job IDs that must complete before this job can run.

        Override this property to declare dependencies. The JobRunner will
        ensure all dependencies are satisfied before executing this job.

        Returns:
            List of job IDs this job depends on
        """
        return []

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
    async def run(self, context: JobContext) -> JobResult:
        """
        Execute the job.

        Args:
            context: JobContext providing access to dependency outputs

        Returns:
            JobResult with status, outputs, and optional error
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(job_id={self.job_id!r})"
