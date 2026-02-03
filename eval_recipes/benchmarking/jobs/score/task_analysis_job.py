# Copyright (c) Microsoft. All rights reserved.

"""Job for analyzing completed trials and generating failure reports."""

import json
import os
from pathlib import Path
from typing import Any

import docker
import docker.errors
from docker.models.containers import Container
from loguru import logger

from eval_recipes.benchmarking.docker_manager import DockerManager, collect_eval_recipes_package
from eval_recipes.benchmarking.job_framework.base import Job, JobContext, JobResult, JobStatus
from eval_recipes.benchmarking.jobs.execute_agent_job import ExecuteAgentJob
from eval_recipes.benchmarking.jobs.score.execute_evaluations_job import ExecuteEvaluationsJob
from eval_recipes.benchmarking.schemas import TaskAnalysisJobInput, TaskAnalysisJobOutput


class TaskAnalysisJob(Job[TaskAnalysisJobOutput]):
    """Job that analyzes completed trials, extracts project files, and performs cleanup.

    The failure analysis runs INSIDE the container using Claude Agent SDK, giving the
    analysis agent full access to the container's environment and ability to run commands.
    """

    output_model = TaskAnalysisJobOutput

    def __init__(
        self,
        job_input: TaskAnalysisJobInput,
        execute_agent_job: ExecuteAgentJob,
        execute_evaluations_job: ExecuteEvaluationsJob,
    ) -> None:
        self._input = job_input
        self._execute_agent_job = execute_agent_job
        self._execute_evaluations_job = execute_evaluations_job

    @property
    def job_id(self) -> str:
        agent_id = self._execute_agent_job._input.agent.id
        return f"analyze:{agent_id}:{self._input.task.name}:{self._input.trial_number}"

    @property
    def dependencies(self) -> list[Job[Any]]:
        return [self._execute_agent_job, self._execute_evaluations_job]

    async def run(self, context: JobContext) -> JobResult[TaskAnalysisJobOutput]:
        output_dir: Path = context.config.get("output_dir", Path.cwd() / ".benchmark_results_v2")

        task = self._input.task
        trial_number = self._input.trial_number

        # Get dependency outputs
        execute_output = context.get_output(self._execute_agent_job)
        eval_output = context.get_output(self._execute_evaluations_job)

        container_id = execute_output.container_id
        image_tag = execute_output.image_tag
        score = eval_output.score

        # Derive trial_dir from execute job's agent info
        agent_id = self._execute_agent_job._input.agent.id
        trial_dir = output_dir / agent_id / task.name / f"trial_{trial_number}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting job: {self.job_id}")

        docker_client: docker.DockerClient | None = None
        container: Container | None = None
        docker_manager: DockerManager | None = None

        # Analysis results (populated if analysis runs)
        report_content: str | None = None
        failure_category: str | None = None
        valid_trial = True
        analysis_skipped = False
        analysis_error: str | None = None

        try:
            # Connect to existing container
            docker_client = docker.from_env()
            container = docker_client.containers.get(container_id)

            # Create a DockerManager instance for utility methods
            docker_manager = DockerManager(
                log_dir=trial_dir,
                dockerfile="",  # Not used - container already exists
                image_tag="",  # Not used - container already exists
            )
            docker_manager.client = docker_client

            # Check score threshold - skip analysis if score is high
            if score >= self._input.analysis_score_threshold:
                logger.info(
                    f"Skipping analysis for trial {trial_number} "
                    f"(score {score:.1f} >= {self._input.analysis_score_threshold})"
                )
                analysis_skipped = True
            else:
                # Run failure analysis inside the container
                logger.info(f"Generating failure analysis for trial {trial_number} (score: {score:.1f})")
                try:
                    report_content, failure_category, valid_trial = self._run_failure_analysis(
                        docker_manager=docker_manager,
                        container=container,
                        task_name=task.name,
                        task_instructions=task.instructions or "",
                        trial_dir=trial_dir,
                    )
                except Exception as e:
                    logger.exception(f"Failure analysis failed: {e}")
                    analysis_error = str(e)

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

        finally:
            # Always extract project directory if we have a valid container
            if docker_manager and container:
                self._extract_project_directory(docker_manager, container, trial_dir)

            # Always cleanup docker resources
            if docker_client:
                self._cleanup_docker_resources(docker_client, container_id, image_tag)
                docker_client.close()

        # Return appropriate result
        if analysis_error:
            return JobResult(
                status=JobStatus.COMPLETED,
                output=TaskAnalysisJobOutput(
                    valid_trial=False,
                    analysis_skipped=False,
                    failure_report=f"Analysis failed: {analysis_error}",
                    failure_category="infrastructure_error",
                ),
            )

        return JobResult(
            status=JobStatus.COMPLETED,
            output=TaskAnalysisJobOutput(
                valid_trial=valid_trial,
                analysis_skipped=analysis_skipped,
                failure_report=report_content,
                failure_category=failure_category,
            ),
        )

    def _extract_project_directory(
        self,
        docker_manager: DockerManager,
        container: Container,
        trial_dir: Path,
    ) -> None:
        """Extract /project directory from container to trial directory."""
        project_dest = trial_dir / "project"
        logger.info(f"Extracting /project directory to {project_dest}")
        try:
            docker_manager.extract_directory_from_container(
                container=container,
                src_path="/project",
                dest_path=project_dest,
                exclude_dotfiles=False,  # Include dotfiles for debugging
                exclude_paths=["_eval_recipes"],  # Exclude runtime package
            )
            logger.info(f"Extracted project directory to {project_dest}")
        except docker.errors.NotFound:
            logger.warning("Container /project directory not found - may be empty")
        except Exception as e:
            logger.warning(f"Failed to extract project directory: {e}")

    def _cleanup_docker_resources(self, docker_client: docker.DockerClient, container_id: str, image_tag: str) -> None:
        """Clean up Docker container and image."""
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

    def _run_failure_analysis(
        self,
        docker_manager: DockerManager,
        container: Container,
        task_name: str,
        task_instructions: str,
        trial_dir: Path,
    ) -> tuple[str | None, str | None, bool]:
        """Run failure analysis inside the container using Claude Agent SDK.

        Returns:
            Tuple of (report_content, failure_category, valid_trial)
        """
        # Get the analysis_runner.py script
        analysis_runner_path = Path(__file__).parents[2] / "evaluation" / "analysis_runner.py"
        if not analysis_runner_path.exists():
            logger.error(f"Analysis runner script not found: {analysis_runner_path}")
            return None, None, True

        # Clean up any existing injected files/dirs to ensure fresh state
        container.exec_run(
            ["rm", "-rf", "/project/_eval_recipes", "/project/analysis_runner.py", "/project/instructions.txt"]
        )

        # Copy eval_recipes package to container subdirectory (enables imports without GitHub)
        # Create the target directory first since put_archive requires it to exist
        container.exec_run(["mkdir", "-p", "/project/_eval_recipes"])
        runtime_files = collect_eval_recipes_package()
        docker_manager.copy_files_to_container(
            container=container,
            files=runtime_files,
            dest_path="/project/_eval_recipes",
        )
        logger.info(f"Copied {len(runtime_files)} eval_recipes files to /project/_eval_recipes")

        # Prepare analysis script and instructions
        files_to_copy: dict[str, bytes] = {
            "analysis_runner.py": analysis_runner_path.read_bytes(),
        }

        # Add instructions if available
        if task_instructions:
            files_to_copy["instructions.txt"] = task_instructions.encode("utf-8")

        # Add log files from trial directory
        log_files = ["agent_output.log", "test_output.log", "test_results.json"]
        for log_file in log_files:
            log_path = trial_dir / log_file
            if log_path.exists():
                files_to_copy[log_file] = log_path.read_bytes()

        # Copy files to container
        docker_manager.copy_files_to_container(
            container=container,
            files=files_to_copy,
            dest_path="/project",
        )
        logger.info("Copied analysis script and instructions to container")

        # Build the analysis command - uv reads dependencies from pyproject.toml in _eval_recipes
        analysis_command = [
            "uv",
            "run",
            "--project",
            "/project/_eval_recipes",
            "/project/analysis_runner.py",
            "--task-name",
            task_name,
            "--output-path",
            "/project/.analysis_result.json",
        ]

        logger.info(f"Running analysis inside container: {' '.join(analysis_command)}")
        _exec_result, _analysis_output = docker_manager.exec_command(
            container=container,
            command=analysis_command,
            log_filename="analysis_agent.log",
            timeout=900,
            environment={
                "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
                "PYTHONPATH": "/project/_eval_recipes",  # So Python finds the copied eval_recipes package
            },
            workdir="/project",
        )
        logger.info("Analysis execution completed")

        # Read results from container
        result_bytes = docker_manager.read_file_from_container(container, "/project/.analysis_result.json")
        if result_bytes:
            try:
                result_data = json.loads(result_bytes.decode("utf-8"))
                failure_category = result_data.get("failure_category")
                valid_trial = result_data.get("valid_trial", True)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse analysis result: {e}")
                failure_category = None
                valid_trial = True
        else:
            logger.warning("Analysis result file not found in container")
            failure_category = None
            valid_trial = True

        # Read the failure report from container
        report_bytes = docker_manager.read_file_from_container(container, "/project/FAILURE_REPORT.md")
        if report_bytes:
            report_content = report_bytes.decode("utf-8")
            # Also save the report to trial directory for redundancy
            report_output_path = trial_dir / "FAILURE_REPORT.md"
            report_output_path.write_text(report_content, encoding="utf-8")
            logger.info(f"Failure report saved to: {report_output_path}")
        else:
            logger.warning("Failure report not found in container")
            report_content = None

        return report_content, failure_category, valid_trial
