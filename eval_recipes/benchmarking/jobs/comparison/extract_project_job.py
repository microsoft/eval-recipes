# Copyright (c) Microsoft. All rights reserved.

"""Job for extracting /project directory from container and cleaning up Docker resources."""

from pathlib import Path
from typing import Any

import docker
import docker.errors
from loguru import logger

from eval_recipes.benchmarking.docker_manager import DockerManager
from eval_recipes.benchmarking.job_framework.base import Job, JobContext, JobResult, JobStatus
from eval_recipes.benchmarking.jobs.execute_agent_job import ExecuteAgentJob
from eval_recipes.benchmarking.schemas import ExtractProjectJobInput, ExtractProjectJobOutput


class ExtractProjectJob(Job[ExtractProjectJobOutput]):
    """Job that extracts /project directory from container and cleans up Docker resources."""

    output_model = ExtractProjectJobOutput

    def __init__(
        self,
        job_input: ExtractProjectJobInput,
        execute_agent_job: ExecuteAgentJob,
    ) -> None:
        self._input = job_input
        self._execute_agent_job = execute_agent_job

    @property
    def job_id(self) -> str:
        return f"extract:{self._input.agent_id}:{self._input.task_name}:{self._input.trial_number}"

    @property
    def dependencies(self) -> list[Job[Any]]:
        return [self._execute_agent_job]

    async def run(self, context: JobContext) -> JobResult[ExtractProjectJobOutput]:
        output_dir: Path = context.config.get("output_dir", Path.cwd() / ".benchmark_results_v2")

        logger.info(f"Starting job: {self.job_id}")

        # Get container info from execute job
        execute_output = context.get_output(self._execute_agent_job)
        container_id = execute_output.container_id
        image_tag = execute_output.image_tag

        # Create output directory for this agent/task
        project_output_dir = output_dir / self._input.agent_id / self._input.task_name / "project"
        project_output_dir.mkdir(parents=True, exist_ok=True)

        client: docker.DockerClient | None = None
        try:
            client = docker.from_env()
            container = client.containers.get(container_id)

            # Create a minimal DockerManager just for extraction
            docker_manager = DockerManager(
                log_dir=output_dir / self._input.agent_id / self._input.task_name,
                dockerfile="",  # Not needed for extraction
            )
            docker_manager.client = client

            # Extract /project from container
            logger.info(f"Extracting /project to {project_output_dir}")
            docker_manager.extract_directory_from_container(
                container=container,
                src_path="/project",
                dest_path=project_output_dir,
                exclude_dotfiles=True,
            )

            logger.info(f"Successfully extracted project to {project_output_dir}")

            return JobResult(
                status=JobStatus.COMPLETED,
                output=ExtractProjectJobOutput(
                    project_dir=str(project_output_dir),
                    agent_id=self._input.agent_id,
                    task_name=self._input.task_name,
                ),
            )

        except docker.errors.NotFound as e:
            error_msg = f"Container {container_id} not found: {e}"
            logger.error(error_msg)
            return JobResult(status=JobStatus.FAILED, error=error_msg)

        except Exception as e:
            error_msg = f"Failed to extract project: {e}"
            logger.exception(error_msg)
            return JobResult(status=JobStatus.FAILED, error=error_msg)

        finally:
            # Always cleanup: stop container and remove image
            if client is not None:
                try:
                    container = client.containers.get(container_id)
                    container.remove(force=True)
                    logger.info(f"Removed container {container_id}")
                except Exception as e:
                    logger.warning(f"Failed to remove container {container_id}: {e}")

                try:
                    client.images.remove(image_tag, force=True)
                    logger.info(f"Removed image {image_tag}")
                except Exception as e:
                    logger.warning(f"Failed to remove image {image_tag}: {e}")

                client.close()
