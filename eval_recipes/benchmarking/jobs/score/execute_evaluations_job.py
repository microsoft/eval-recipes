# Copyright (c) Microsoft. All rights reserved.

"""Job for running evaluations/tests on a completed agent task."""

import json
from pathlib import Path
import time
from typing import Any
import uuid

import docker
import docker.errors
from loguru import logger

from eval_recipes.benchmarking.docker_manager import DockerManager, collect_eval_recipes_package
from eval_recipes.benchmarking.job_framework.base import Job, JobContext, JobResult, JobStatus
from eval_recipes.benchmarking.jobs.execute_agent_job import ExecuteAgentJob, collect_directory_files
from eval_recipes.benchmarking.schemas import EvaluateJobOutput, ExecuteEvaluationsJobInput, ScoreEvalConfig


class ExecuteEvaluationsJob(Job[EvaluateJobOutput]):
    """Job that runs test evaluations in an existing container after agent execution."""

    output_model = EvaluateJobOutput

    def __init__(
        self,
        job_input: ExecuteEvaluationsJobInput,
        execute_agent_job: ExecuteAgentJob,
    ) -> None:
        self._input = job_input
        self._execute_agent_job = execute_agent_job

    @property
    def job_id(self) -> str:
        agent_id = self._execute_agent_job._input.agent.id
        return f"evaluate:{agent_id}:{self._input.task.name}:{self._input.trial_number}"

    @property
    def dependencies(self) -> list[Job[Any]]:
        return [self._execute_agent_job]

    async def run(self, context: JobContext) -> JobResult[EvaluateJobOutput]:
        output_dir: Path = context.config.get("output_dir", Path.cwd() / ".benchmark_results_v2")

        task = self._input.task
        trial_number = self._input.trial_number

        # Get execute job output to access container_id
        execute_output = context.get_output(self._execute_agent_job)
        container_id = execute_output.container_id

        # Derive trial_dir from execute job's agent info
        # We need to get agent_id from the execute job's input
        agent_id = self._execute_agent_job._input.agent.id
        trial_dir = output_dir / agent_id / task.name / f"trial_{trial_number}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting job: {self.job_id}")

        try:
            # Connect to existing container
            docker_client = docker.from_env()
            container = docker_client.containers.get(container_id)

            # Create a DockerManager instance for utility methods
            # We're reusing an existing container, so we don't need build parameters
            docker_manager = DockerManager(
                log_dir=trial_dir,
                dockerfile="",  # Not used - container already exists
                image_tag="",  # Not used - container already exists
            )
            docker_manager.client = docker_client
            docker_manager.container = container
            docker_manager.container_id = container_id

            # Copy test-time files to container
            for mapping in task.test_time_files:
                if mapping.source.exists():
                    files = collect_directory_files(mapping.source)
                    if files:
                        docker_manager.copy_files_to_container(
                            container=container,
                            files=files,
                            dest_path=mapping.dest,
                        )
                        logger.info(f"Copied {len(files)} test-time file(s) to {mapping.dest}")

            # Get the score evaluation config
            score_eval = None
            for eval_config in task.evaluation_configs:
                if isinstance(eval_config, ScoreEvalConfig):
                    score_eval = eval_config
                    break

            if score_eval is None:
                return JobResult(
                    status=JobStatus.FAILED,
                    error=f"Task '{task.name}' does not have a ScoreEvalConfig",
                )

            # Prepare test ID for result file identification
            test_id = str(uuid.uuid4())

            # Write agent metadata file for semantic tests to optionally read
            agent_metadata = {"agent_log_hint": self._input.agent_log_hint}
            agent_metadata_content = json.dumps(agent_metadata, indent=2)

            # Clean up any existing injected files/dirs to ensure fresh state
            container.exec_run(["rm", "-rf", "/project/_eval_recipes", "/project/test.py", "/project/instructions.txt"])

            # Prepare files to copy: test script, instructions, and agent metadata
            files_to_copy: dict[str, bytes] = {
                "test.py": score_eval.test_script.read_bytes(),
                ".agent_metadata.json": agent_metadata_content.encode("utf-8"),
            }
            if task.instructions:
                files_to_copy["instructions.txt"] = task.instructions.encode("utf-8")

            docker_manager.copy_files_to_container(
                container=container,
                files=files_to_copy,
                dest_path="/project",
            )
            logger.info("Copied test script and metadata to container")

            # Copy eval_recipes package to container subdirectory
            container.exec_run(["mkdir", "-p", "/project/_eval_recipes"])
            runtime_files = collect_eval_recipes_package()
            docker_manager.copy_files_to_container(
                container=container,
                files=runtime_files,
                dest_path="/project/_eval_recipes",
            )
            logger.info(f"Copied {len(runtime_files)} eval_recipes files to /project/_eval_recipes")

            # Build test command - uv reads dependencies from pyproject.toml in _eval_recipes
            test_command_parts = ["uv", "run", "--project", "/project/_eval_recipes", "/project/test.py"]

            logger.info(f"Running test: {' '.join(test_command_parts)}")
            test_start_time = time.perf_counter()
            _exec_result, test_output = docker_manager.exec_command(
                container=container,
                command=test_command_parts,
                log_filename="test_output.log",
                timeout=task.timeout,
                environment={
                    "EVAL_RECIPES_TEST_ID": test_id,
                    "PYTHONPATH": "/project/_eval_recipes",
                },
                workdir="/project",
            )
            test_duration = time.perf_counter() - test_start_time
            logger.info(f"Test completed in {test_duration:.1f}s")

            # Parse result file
            result_file_path = f"/project/.eval_recipes_test_results_{test_id}.json"
            result_bytes = docker_manager.read_file_from_container(container, result_file_path)

            if result_bytes:
                # Save raw result file to output directory for debugging/auditing
                result_output_path = trial_dir / "test_results.json"
                result_output_path.write_bytes(result_bytes)
                logger.info(f"Test results saved to: {result_output_path}")

                result_data = json.loads(result_bytes.decode("utf-8"))
                score = float(result_data.get("score", 0))
                rubric = result_data.get("metadata", {})
                logger.info(f"Test score: {score}, metadata: {rubric}")
            else:
                logger.warning(f"Could not read results file: {result_file_path}")
                score = 0.0
                rubric = {"error": "No results file found"}

            output = EvaluateJobOutput(
                score=score,
                rubric=rubric,
                test_console_log=test_output,
                test_duration_seconds=test_duration,
            )
            return JobResult(status=JobStatus.COMPLETED, output=output)

        except docker.errors.NotFound as e:
            error_msg = f"Container not found: {e}"
            logger.error(error_msg)
            return JobResult(status=JobStatus.FAILED, error=error_msg)

        except docker.errors.APIError as e:
            error_msg = f"Docker API error: {e}"
            logger.error(error_msg)
            return JobResult(status=JobStatus.FAILED, error=error_msg)

        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            logger.exception(error_msg)
            return JobResult(status=JobStatus.FAILED, error=error_msg)
