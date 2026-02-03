# Copyright (c) Microsoft. All rights reserved.

"""Job for executing an agent on a task."""

from pathlib import Path
import time
from typing import Literal

import docker
import docker.errors
from liquid import Template
from loguru import logger
import pathspec

from eval_recipes.benchmarking.docker_manager import DockerManager
from eval_recipes.benchmarking.evaluation.agent_interacter import interact_with_agent
from eval_recipes.benchmarking.job_framework.base import Job, JobContext, JobResult, JobStatus
from eval_recipes.benchmarking.schemas import (
    AgentDefinition,
    ExecuteAgentJobInput,
    ExecuteAgentJobOutput,
    TaskDefinition,
)

BASE_DOCKERFILE_TEMPLATE = """\
FROM ubuntu:24.04

RUN apt-get update && \\
    apt-get install -y --no-install-recommends \\
    ca-certificates \\
    curl \\
    git

# Install uv: https://docs.astral.sh/uv/getting-started/installation/
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /project

# Most tests currently require the Claude Agent SDK
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
RUN apt-get install -y nodejs

RUN npm install -g @anthropic-ai/claude-code

ENV BASH_DEFAULT_TIMEOUT_MS=300000
ENV BASH_MAX_TIMEOUT_MS=600000

RUN claude --version

{{agent_installation}}

{{task_installation}}
"""


class ExecuteAgentJob(Job[ExecuteAgentJobOutput]):
    """Job that creates container and runs agent on task."""

    output_model = ExecuteAgentJobOutput

    def __init__(self, job_input: ExecuteAgentJobInput) -> None:
        self._input = job_input

    @property
    def job_id(self) -> str:
        return f"execute:{self._input.agent.id}:{self._input.task.name}:{self._input.trial_number}"

    async def run(self, context: JobContext) -> JobResult[ExecuteAgentJobOutput]:
        output_dir: Path = context.config.get("output_dir", Path.cwd() / ".benchmark_results_v2")
        environment: dict[str, str] = context.config.get("environment", {})

        agent = self._input.agent
        task = self._input.task
        trial_number = self._input.trial_number

        trial_dir = output_dir / agent.id / task.name / f"trial_{trial_number}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting job: {self.job_id}")

        # Build context files from installation_files
        build_context_files: dict[str, bytes] = {}
        for mapping in agent.installation_files:
            if mapping.source.exists():
                source_files = collect_directory_files(mapping.source)
                dest_prefix = mapping.dest.lstrip("/")
                for file_path, content in source_files.items():
                    build_context_files[f"{dest_prefix}/{file_path}"] = content
                logger.info(f"Collected {len(source_files)} file(s) from {mapping.source}")

        # Build dockerfile
        dockerfile_content = _build_dockerfile(agent, task)
        image_tag = f"benchmark-{agent.id}-{task.name}-trial{trial_number}".lower()

        docker_manager = DockerManager(
            log_dir=trial_dir,
            dockerfile=dockerfile_content,
            image_tag=image_tag,
            container_env=environment,
            build_context_files=build_context_files,
        )

        try:
            # Manually initialize Docker client and build/start container
            docker_manager.client = docker.from_env()
            _image, _build_logs, docker_manager.actual_image_tag = docker_manager._build_image(
                dockerfile=docker_manager.dockerfile,
                image_tag=docker_manager.image_tag,
            )
            docker_manager.container, docker_manager.container_id = docker_manager._run_container(
                image_tag=docker_manager.actual_image_tag,
                container_env=docker_manager.container_env,
            )

            if docker_manager.container is None or docker_manager.container_id is None:
                return JobResult(status=JobStatus.FAILED, error="Failed to start container")

            logger.info(f"Built image: {docker_manager.actual_image_tag}")
            logger.info(f"Container {docker_manager.container_id} started")

            # Copy runtime files to container
            for mapping in agent.runtime_files:
                if mapping.source.exists():
                    files = collect_directory_files(mapping.source)
                    if files:
                        docker_manager.copy_files_to_container(
                            container=docker_manager.container,
                            files=files,
                            dest_path=mapping.dest,
                        )
                        logger.info(f"Copied {len(files)} runtime file(s) to {mapping.dest}")

            # Copy task-time files to container
            for mapping in task.task_time_files:
                if mapping.source.exists():
                    files = collect_directory_files(mapping.source)
                    if files:
                        docker_manager.copy_files_to_container(
                            container=docker_manager.container,
                            files=files,
                            dest_path=mapping.dest,
                        )
                        logger.info(f"Copied {len(files)} task-time data file(s) to {mapping.dest}")

            # Build and execute command
            task_instructions = task.instructions or ""
            escaped_instructions = _escape_bash_string(task_instructions)
            command_template = Template(agent.command_template)
            command = command_template.render(task_instructions=escaped_instructions)

            logger.info(f"Executing command: {command}")
            agent_start_time = time.perf_counter()
            _, exec_logs = docker_manager.exec_command(
                container=docker_manager.container,
                command=["bash", "-c", command],
                log_filename="agent_output.log",
                timeout=task.timeout,
            )
            agent_duration = time.perf_counter() - agent_start_time
            logger.info(f"Command execution completed in {agent_duration:.1f}s")

            # Handle agent continuation
            continuation_occurred = False
            continuation_prompt: str | None = None
            continuation_error: str | None = None

            continuation_provider = self._input.continuation_provider
            continuation_model = self._input.continuation_model

            if continuation_provider != "none" and agent.command_template_continue is not None:
                try:
                    provider: Literal["openai", "azure_openai"] = continuation_provider  # type: ignore[assignment]
                    continuation_response = await interact_with_agent(
                        agent_log=exec_logs,
                        task_instructions=task_instructions,
                        provider=provider,
                        model=continuation_model,
                    )

                    if continuation_response:
                        continuation_occurred = True
                        continuation_prompt = continuation_response
                        logger.info("Continuation needed - executing continuation command")

                        # Escape and strip leading dashes to avoid CLI option interpretation
                        escaped_response = _escape_bash_string(continuation_response.lstrip("- "))
                        continuation_template = Template(agent.command_template_continue)
                        continuation_command = continuation_template.render(task_instructions=escaped_response)

                        logger.info(f"Executing continuation command: {continuation_command}")
                        continuation_start = time.perf_counter()
                        _, continuation_logs = docker_manager.exec_command(
                            container=docker_manager.container,
                            command=["bash", "-c", continuation_command],
                            log_filename="agent_continuation.log",
                            timeout=task.timeout,
                        )
                        continuation_duration = time.perf_counter() - continuation_start
                        logger.info(f"Continuation completed in {continuation_duration:.1f}s")

                        # Append continuation logs to main logs with delimiter
                        exec_logs += (
                            f"\n\n========== CONTINUATION ==========\n"
                            f"Continuation prompt: {continuation_response}\n\n"
                            f"{continuation_logs}"
                        )
                        agent_duration += continuation_duration
                    else:
                        logger.info("No continuation needed - agent task appears complete")
                except Exception as e:
                    logger.error(f"Failed during agent continuation: {e}")
                    continuation_error = str(e)
            else:
                if continuation_provider == "none":
                    logger.info("Agent continuation is disabled (continuation_provider='none')")
                else:
                    logger.info("Agent does not have a continuation template - skipping continuation check")

            output = ExecuteAgentJobOutput(
                container_id=docker_manager.container_id,
                image_tag=docker_manager.actual_image_tag,
                agent_console_log=exec_logs,
                agent_duration_seconds=agent_duration,
                continuation_occurred=continuation_occurred,
                continuation_prompt=continuation_prompt,
                continuation_error=continuation_error,
            )
            return JobResult(status=JobStatus.COMPLETED, output=output)

        except docker.errors.BuildError as e:
            error_msg = f"Docker build failed: {e}"
            logger.error(error_msg)
            return JobResult(status=JobStatus.FAILED, error=error_msg)

        except docker.errors.ContainerError as e:
            error_msg = f"Container error: {e}"
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


def _build_dockerfile(agent: AgentDefinition, task: TaskDefinition) -> str:
    """Build the complete Dockerfile from base template using liquid."""
    template = Template(BASE_DOCKERFILE_TEMPLATE)

    agent_installation = agent.dockerfile_portion
    if agent.installation_files:
        copy_commands = []
        for mapping in agent.installation_files:
            dest_prefix = mapping.dest.lstrip("/")
            copy_commands.append(f"COPY {dest_prefix} {mapping.dest}")
        agent_installation = "\n".join(copy_commands) + "\n" + agent_installation

    return template.render(
        agent_installation=agent_installation,
        task_installation=task.dockerfile_portion,
    )


def collect_directory_files(directory: Path) -> dict[str, bytes]:
    """
    Recursively collect all files from a directory.

    Respects .gitignore patterns if a .gitignore file exists in the directory root.
    Falls back to hardcoded skip patterns if no .gitignore is present.
    """
    skip_dirs = {".git", ".venv", "__pycache__", "node_modules", ".pytest_cache", ".mypy_cache", ".ruff_cache"}

    gitignore_spec: pathspec.PathSpec | None = None
    gitignore_path = directory / ".gitignore"
    if gitignore_path.exists():
        try:
            with gitignore_path.open(encoding="utf-8") as f:
                gitignore_spec = pathspec.PathSpec.from_lines("gitwildmatch", f)
        except Exception as e:
            logger.warning(f"Failed to parse .gitignore at {gitignore_path}: {e}")

    files: dict[str, bytes] = {}
    for file_path in directory.rglob("*"):
        if not file_path.is_file():
            continue

        relative_path = file_path.relative_to(directory)

        if gitignore_spec and gitignore_spec.match_file(str(relative_path)):
            continue

        if any(part in skip_dirs for part in file_path.parts):
            continue

        if file_path.suffix in {".pyc", ".pyo"}:
            continue

        files[str(relative_path)] = file_path.read_bytes()

    return files


def _escape_bash_string(text: str) -> str:
    """Escape special characters in a string for safe use in bash commands."""
    text = text.replace("\\", "\\\\")
    text = text.replace('"', '\\"')
    text = text.replace("`", "\\`")
    text = text.replace("$", "\\$")
    return text
