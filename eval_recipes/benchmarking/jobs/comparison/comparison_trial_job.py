# Copyright (c) Microsoft. All rights reserved.

"""Job for executing a complete comparison trial (agent execution and project extraction)."""

from typing import Any

import docker
import docker.errors
from loguru import logger

from eval_recipes.benchmarking.job_framework.base import Job, JobContext, JobResult, JobStatus
from eval_recipes.benchmarking.jobs.comparison.extract_project_job import ExtractProjectJob
from eval_recipes.benchmarking.jobs.execute_agent_job import ExecuteAgentJob
from eval_recipes.benchmarking.schemas import (
    ComparisonTrialJobInput,
    ComparisonTrialJobOutput,
    ExecuteAgentJobInput,
    ExtractProjectJobInput,
)


class ComparisonTrialJob(Job[ComparisonTrialJobOutput]):
    """Job that wraps agent execution and project extraction for comparison benchmarks.

    This job combines ExecuteAgentJob and ExtractProjectJob into a single job to ensure
    the Docker container lifecycle is contained within a single job execution. This prevents
    containers from sitting idle in the queue when parallelism is limited.
    """

    output_model = ComparisonTrialJobOutput

    def __init__(self, job_input: ComparisonTrialJobInput) -> None:
        self._input = job_input

        # Create internal jobs
        execute_input = ExecuteAgentJobInput(
            agent=job_input.agent,
            task=job_input.task,
            trial_number=job_input.trial_number,
            continuation_provider=job_input.continuation_provider,
            continuation_model=job_input.continuation_model,
        )
        self._execute_job = ExecuteAgentJob(execute_input)

        extract_input = ExtractProjectJobInput(
            agent_id=job_input.agent.id,
            task_name=job_input.task.name,
            trial_number=job_input.trial_number,
        )
        self._extract_job = ExtractProjectJob(extract_input, self._execute_job)

    @property
    def job_id(self) -> str:
        return f"comparison_trial:{self._input.agent.id}:{self._input.task.name}:{self._input.trial_number}"

    async def run(self, context: JobContext) -> JobResult[ComparisonTrialJobOutput]:
        logger.info(f"Starting job: {self.job_id}")

        container_id: str | None = None
        image_tag: str | None = None

        try:
            # Phase 1: Execute agent
            execute_result = await self._execute_job.run(context)

            if execute_result.status != JobStatus.COMPLETED or execute_result.output is None:
                error_msg = execute_result.error or "ExecuteAgentJob failed without error message"
                logger.error(f"ExecuteAgentJob failed: {error_msg}")
                return JobResult(status=JobStatus.FAILED, error=error_msg)

            execute_output = execute_result.output
            container_id = execute_output.container_id
            image_tag = execute_output.image_tag

            # Phase 2: Extract project (has cleanup in finally block)
            extract_context = self._build_context_with_outputs(
                context,
                [(self._execute_job, execute_output)],
            )
            extract_result = await self._extract_job.run(extract_context)

            # Note: ExtractProjectJob handles Docker cleanup in its finally block,
            # so we don't need to clean up here even if it fails

            if extract_result.status != JobStatus.COMPLETED or extract_result.output is None:
                error_msg = extract_result.error or "ExtractProjectJob failed without error message"
                logger.error(f"ExtractProjectJob failed: {error_msg}")
                return JobResult(status=JobStatus.FAILED, error=error_msg)

            extract_output = extract_result.output

            # Combine outputs
            combined_output = ComparisonTrialJobOutput(
                # From ExecuteAgentJobOutput
                agent_console_log=execute_output.agent_console_log,
                agent_duration_seconds=execute_output.agent_duration_seconds,
                continuation_occurred=execute_output.continuation_occurred,
                continuation_prompt=execute_output.continuation_prompt,
                continuation_error=execute_output.continuation_error,
                # From ExtractProjectJobOutput
                project_dir=extract_output.project_dir,
                agent_id=extract_output.agent_id,
                task_name=extract_output.task_name,
            )

            logger.info(f"Completed job: {self.job_id}")
            return JobResult(status=JobStatus.COMPLETED, output=combined_output)

        except Exception as e:
            error_msg = f"Unexpected error in ComparisonTrialJob: {e}"
            logger.exception(error_msg)
            # Ensure cleanup on unexpected errors
            if container_id and image_tag:
                self._cleanup_docker_resources(container_id, image_tag)
            return JobResult(status=JobStatus.FAILED, error=error_msg)

    def _build_context_with_outputs(
        self,
        base_context: JobContext,
        job_outputs: list[tuple[Job[Any], Any]],
    ) -> JobContext:
        """Build a new JobContext with outputs from completed internal jobs.

        Args:
            base_context: The original context passed to this job
            job_outputs: List of (job, output) tuples to include in the new context

        Returns:
            A new JobContext with the job outputs available
        """
        dependency_outputs: dict[str, dict[str, Any]] = {}
        dependency_jobs: dict[str, Job[Any]] = {}

        for job, output in job_outputs:
            dependency_outputs[job.job_id] = output.model_dump()
            dependency_jobs[job.job_id] = job

        return JobContext(dependency_outputs, dependency_jobs, base_context.config)

    def _cleanup_docker_resources(self, container_id: str, image_tag: str) -> None:
        """Clean up Docker container and image.

        This is a fallback cleanup method for when an unexpected exception occurs
        before ExtractProjectJob can run its cleanup.
        """
        try:
            docker_client = docker.from_env()

            # Remove container
            try:
                container = docker_client.containers.get(container_id)
                container.remove(force=True)
                logger.info(f"Removed container: {container_id}")
            except docker.errors.NotFound:
                logger.debug(f"Container already removed: {container_id}")
            except Exception as e:
                logger.warning(f"Failed to remove container {container_id}: {e}")

            # Remove image
            try:
                docker_client.images.remove(image_tag, force=True)
                logger.info(f"Removed image: {image_tag}")
            except docker.errors.ImageNotFound:
                logger.debug(f"Image already removed: {image_tag}")
            except Exception as e:
                logger.warning(f"Failed to remove image {image_tag}: {e}")

            docker_client.close()

        except Exception as e:
            logger.warning(f"Failed to initialize Docker client for cleanup: {e}")
