# Copyright (c) Microsoft. All rights reserved.

"""Job for executing a complete trial (agent execution, evaluation, and analysis)."""

from typing import Any

import docker
import docker.errors
from loguru import logger

from eval_recipes.benchmarking.job_framework.base import Job, JobContext, JobResult, JobStatus
from eval_recipes.benchmarking.jobs.execute_agent_job import ExecuteAgentJob
from eval_recipes.benchmarking.jobs.score.execute_evaluations_job import ExecuteEvaluationsJob
from eval_recipes.benchmarking.jobs.score.task_analysis_job import TaskAnalysisJob
from eval_recipes.benchmarking.schemas import (
    ExecuteAgentJobInput,
    ExecuteEvaluationsJobInput,
    TaskAnalysisJobInput,
    TrialExecutionJobInput,
    TrialExecutionJobOutput,
)


class TrialExecutionJob(Job[TrialExecutionJobOutput]):
    """Job that wraps the complete trial execution lifecycle.

    This job combines ExecuteAgentJob, ExecuteEvaluationsJob, and TaskAnalysisJob
    into a single job to ensure the Docker container lifecycle is contained within
    a single job execution. This prevents containers from sitting idle in the queue
    when parallelism is limited.
    """

    output_model = TrialExecutionJobOutput

    def __init__(self, job_input: TrialExecutionJobInput) -> None:
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

        eval_input = ExecuteEvaluationsJobInput(
            task=job_input.task,
            trial_number=job_input.trial_number,
            agent_log_hint=job_input.agent_log_hint,
        )
        self._eval_job = ExecuteEvaluationsJob(eval_input, self._execute_job)

        analysis_input = TaskAnalysisJobInput(
            task=job_input.task,
            trial_number=job_input.trial_number,
            analysis_score_threshold=job_input.analysis_score_threshold,
        )
        self._analysis_job = TaskAnalysisJob(analysis_input, self._execute_job, self._eval_job)

    @property
    def job_id(self) -> str:
        return f"trial:{self._input.agent.id}:{self._input.task.name}:{self._input.trial_number}"

    async def run(self, context: JobContext) -> JobResult[TrialExecutionJobOutput]:
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

            # Phase 2: Execute evaluations
            eval_context = self._build_context_with_outputs(
                context,
                [(self._execute_job, execute_output)],
            )
            eval_result = await self._eval_job.run(eval_context)

            if eval_result.status != JobStatus.COMPLETED or eval_result.output is None:
                error_msg = eval_result.error or "ExecuteEvaluationsJob failed without error message"
                logger.error(f"ExecuteEvaluationsJob failed: {error_msg}")
                # Clean up Docker resources before returning
                self._cleanup_docker_resources(container_id, image_tag)
                return JobResult(status=JobStatus.FAILED, error=error_msg)

            eval_output = eval_result.output

            # Phase 3: Task analysis (includes cleanup in finally block)
            analysis_context = self._build_context_with_outputs(
                context,
                [(self._execute_job, execute_output), (self._eval_job, eval_output)],
            )
            analysis_result = await self._analysis_job.run(analysis_context)

            if analysis_result.status != JobStatus.COMPLETED or analysis_result.output is None:
                error_msg = analysis_result.error or "TaskAnalysisJob failed without error message"
                logger.error(f"TaskAnalysisJob failed: {error_msg}")
                # Note: TaskAnalysisJob handles its own cleanup in finally block,
                # but we add this for safety in case it fails before reaching cleanup
                return JobResult(status=JobStatus.FAILED, error=error_msg)

            analysis_output = analysis_result.output

            # Combine all outputs
            combined_output = TrialExecutionJobOutput(
                # From ExecuteAgentJobOutput
                agent_console_log=execute_output.agent_console_log,
                agent_duration_seconds=execute_output.agent_duration_seconds,
                continuation_occurred=execute_output.continuation_occurred,
                continuation_prompt=execute_output.continuation_prompt,
                continuation_error=execute_output.continuation_error,
                # From EvaluateJobOutput
                score=eval_output.score,
                rubric=eval_output.rubric,
                test_console_log=eval_output.test_console_log,
                test_duration_seconds=eval_output.test_duration_seconds,
                # From TaskAnalysisJobOutput
                valid_trial=analysis_output.valid_trial,
                analysis_skipped=analysis_output.analysis_skipped,
                failure_report=analysis_output.failure_report,
                failure_category=analysis_output.failure_category,
            )

            logger.info(f"Completed job: {self.job_id} (score: {eval_output.score:.1f})")
            return JobResult(status=JobStatus.COMPLETED, output=combined_output)

        except Exception as e:
            error_msg = f"Unexpected error in TrialExecutionJob: {e}"
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

        This is a fallback cleanup method for when ExecuteEvaluationsJob fails
        before TaskAnalysisJob can run its cleanup. TaskAnalysisJob has its own
        cleanup in a finally block, so this is only called when we fail before
        reaching that job.
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
