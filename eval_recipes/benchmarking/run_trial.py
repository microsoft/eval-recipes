# Copyright (c) Microsoft. All rights reserved


from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Literal
import uuid

from liquid import Template
from loguru import logger
import pathspec

from eval_recipes.benchmarking.agent_interacter import interact_with_agent
from eval_recipes.benchmarking.docker_manager import DockerManager
from eval_recipes.benchmarking.schemas import AgentConfig, TaskConfig, TrialResult

DEFAULT_EVAL_RECIPES_VERSION = "0.0.28"


@dataclass
class TrialConfig:
    environment: dict[str, str] = field(default_factory=dict)
    continuation_provider: Literal["openai", "azure_openai", "none"] = "none"
    continuation_model: Literal["gpt-5", "gpt-5.1"] = "gpt-5"
    eval_recipes_version: str = DEFAULT_EVAL_RECIPES_VERSION


async def run_trial(
    agent: AgentConfig,
    task: TaskConfig,
    trial_num: int,
    base_run_dir: Path,
    config: TrialConfig,
) -> TrialResult:
    """
    Run a single trial for an agent-task pair.

    Args:
        agent: Agent configuration
        task: Task configuration
        trial_num: Trial number (1-indexed)
        base_run_dir: Base directory for this agent-task pair
        config: Trial configuration

    Returns:
        TrialResult with score and metadata
    """
    task_name = f"{agent.name} on {task.name}"
    logger.info(f"Starting trial {trial_num} for {task_name}")
    trial_dir = base_run_dir / f"trial_{trial_num}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    required_vars = set(agent.required_env_vars + task.required_env_vars)
    container_env = {var: config.environment[var] for var in required_vars if var in config.environment}
    dockerfile_content = _build_dockerfile(agent, task)
    image_tag = f"benchmark-{agent.name}-{task.name}-trial{trial_num}".lower()

    build_context_files: dict[str, bytes] = {}
    if agent.local_source_path:
        logger.info(f"Collecting local source files from {agent.local_source_path}")
        local_source_files = _collect_directory_files(agent.local_source_path)
        for file_path, content in local_source_files.items():
            build_context_files[f"agent_source/{file_path}"] = content
        logger.info(f"Collected {len(local_source_files)} file(s) from local source")

    with DockerManager(
        log_dir=trial_dir,
        dockerfile=dockerfile_content,
        image_tag=image_tag,
        container_env=container_env,
        build_context_files=build_context_files,
    ) as docker_manager:
        assert docker_manager.container is not None
        logger.info(f"Built image: {docker_manager.actual_image_tag}")
        logger.info(f"Container {docker_manager.container_id} started for trial {trial_num}")

        if agent.data_dir and agent.data_dir.exists():
            logger.info(f"Copying agent data directory from {agent.data_dir} to container")
            agent_data_files = _collect_directory_files(agent.data_dir)
            if agent_data_files:
                docker_manager.copy_files_to_container(
                    container=docker_manager.container,
                    files=agent_data_files,
                    dest_path="/project",
                )

        if task.task_time_data_dir and task.task_time_data_dir.exists():
            logger.info(f"Copying task-time data directory from {task.task_time_data_dir} to container")
            task_time_data_files = _collect_directory_files(task.task_time_data_dir)
            if task_time_data_files:
                docker_manager.copy_files_to_container(
                    container=docker_manager.container,
                    files=task_time_data_files,
                    dest_path="/project",
                )

        escaped_instructions = _escape_bash_string(task.instructions)
        command_template = Template(agent.command_template)
        command = command_template.render(task_instructions=escaped_instructions)
        logger.info(f"Executing command for trial {trial_num}: {command}")

        agent_start_time = time.perf_counter()
        _, _exec_logs = docker_manager.exec_command(
            container=docker_manager.container,
            command=["bash", "-c", command],
            log_filename="agent_output.log",
            timeout=task.timeout,
        )
        agent_duration = time.perf_counter() - agent_start_time
        logger.info(f"Trial {trial_num} command execution completed. Output saved to: {trial_dir / 'agent_output.log'}")

        continuation_metadata: dict[str, Any] = {}
        if config.continuation_provider != "none" and agent.command_template_continue is not None:
            try:
                provider: Literal["openai", "azure_openai"] = config.continuation_provider  # type: ignore[assignment]
                model: Literal["gpt-5", "gpt-5.1"] = config.continuation_model  # type: ignore[assignment]
                continuation_response = await interact_with_agent(
                    agent_log=_exec_logs,
                    task_instructions=task.instructions,
                    provider=provider,
                    model=model,
                )

                if continuation_response:
                    continuation_metadata["continuation_occurred"] = True
                    continuation_metadata["continuation_prompt"] = continuation_response

                    # Strip leading dashes to prevent CLI option interpretation
                    escaped_response = _escape_bash_string(continuation_response.lstrip("- "))
                    continuation_template = Template(agent.command_template_continue)
                    continuation_command = continuation_template.render(task_instructions=escaped_response)

                    logger.info(f"Executing continuation command: {continuation_command}")

                    try:
                        continuation_start = time.perf_counter()
                        _, continuation_logs = docker_manager.exec_command(
                            container=docker_manager.container,
                            command=["bash", "-c", continuation_command],
                            log_filename="agent_continuation.log",
                            timeout=task.timeout,
                        )
                        continuation_duration = time.perf_counter() - continuation_start

                        agent_log_path = trial_dir / "agent_output.log"
                        with agent_log_path.open("a", encoding="utf-8") as f:
                            f.write("\n\n--- CONTINUATION ---\n\n")
                            f.write(f"Continuation prompt:\n{continuation_response}\n\n")
                            f.write("--- CONTINUATION OUTPUT ---\n\n")
                            f.write(continuation_logs)

                        agent_duration += continuation_duration

                    except Exception as e:
                        logger.error(f"Failed to execute continuation command: {e}")
                        continuation_metadata["continuation_error"] = str(e)
                        agent_log_path = trial_dir / "agent_output.log"
                        with agent_log_path.open("a", encoding="utf-8") as f:
                            f.write(f"\n\n--- CONTINUATION FAILED ---\n{e}\n")
                else:
                    logger.info("No continuation needed - agent task appears complete")
                    continuation_metadata["continuation_occurred"] = False

            except Exception as e:
                logger.error(f"Failed to check if agent needs continuation: {e}")
                continuation_metadata["continuation_check_error"] = str(e)
        else:
            if config.continuation_provider == "none":
                logger.info("Agent continuation is disabled (continuation_provider='none')")
            else:
                logger.info("Agent does not have a continuation template - skipping continuation check")
            continuation_metadata["continuation_disabled"] = True

        trial_result = _run_tests(
            docker_manager.container,
            task,
            trial_dir,
            docker_manager,
            trial_num,
            agent_duration,
            config.eval_recipes_version,
            continuation_metadata,
        )

        if trial_result:
            logger.info(f"Trial {trial_num} completed with score: {trial_result.score}")
            # Persist full trial result for HTML report generation
            trial_result_file = trial_dir / "trial_result.json"
            trial_result_file.write_text(trial_result.model_dump_json(indent=2), encoding="utf-8")
            return trial_result
        else:
            logger.warning(f"Trial {trial_num} failed to produce results")
            failed_trial_result = TrialResult(
                trial_number=trial_num,
                score=0.0,
                metadata={"error": "Test execution failed"},
                test_output="",
                agent_duration_seconds=agent_duration,
                test_duration_seconds=None,
            )
            # Persist full trial result for HTML report generation
            trial_result_file = trial_dir / "trial_result.json"
            trial_result_file.write_text(failed_trial_result.model_dump_json(indent=2), encoding="utf-8")
            return failed_trial_result


async def run_trial_for_comparison(
    agent: AgentConfig,
    task: TaskConfig,
    trial_dir: Path,
    config: TrialConfig,
) -> Path:
    """Run a trial for comparison evaluation (no test execution).

    This is similar to run_trial() but designed for comparison tasks:
    - Skips the test execution step (no score_eval)
    - Extracts /project folder to local directory
    - Returns path to extracted project directory

    Args:
        agent: Agent configuration
        task: Task configuration
        trial_dir: Directory for this trial's outputs
        config: Trial configuration

    Returns:
        Path to the extracted project directory
    """
    task_name = f"{agent.name} on {task.name}"
    logger.info(f"Starting comparison trial for {task_name}")
    trial_dir.mkdir(parents=True, exist_ok=True)

    required_vars = set(agent.required_env_vars + task.required_env_vars)
    container_env = {var: config.environment[var] for var in required_vars if var in config.environment}
    dockerfile_content = _build_dockerfile(agent, task)
    image_tag = f"comparison-{agent.name}-{task.name}".lower()

    build_context_files: dict[str, bytes] = {}
    if agent.local_source_path:
        logger.info(f"Collecting local source files from {agent.local_source_path}")
        local_source_files = _collect_directory_files(agent.local_source_path)
        for file_path, content in local_source_files.items():
            build_context_files[f"agent_source/{file_path}"] = content
        logger.info(f"Collected {len(local_source_files)} file(s) from local source")

    with DockerManager(
        log_dir=trial_dir,
        dockerfile=dockerfile_content,
        image_tag=image_tag,
        container_env=container_env,
        build_context_files=build_context_files,
    ) as docker_manager:
        assert docker_manager.container is not None
        logger.info(f"Built image: {docker_manager.actual_image_tag}")
        logger.info(f"Container {docker_manager.container_id} started for comparison trial")

        if agent.data_dir and agent.data_dir.exists():
            logger.info(f"Copying agent data directory from {agent.data_dir} to container")
            agent_data_files = _collect_directory_files(agent.data_dir)
            if agent_data_files:
                docker_manager.copy_files_to_container(
                    container=docker_manager.container,
                    files=agent_data_files,
                    dest_path="/project",
                )

        if task.task_time_data_dir and task.task_time_data_dir.exists():
            logger.info(f"Copying task-time data directory from {task.task_time_data_dir} to container")
            task_time_data_files = _collect_directory_files(task.task_time_data_dir)
            if task_time_data_files:
                docker_manager.copy_files_to_container(
                    container=docker_manager.container,
                    files=task_time_data_files,
                    dest_path="/project",
                )

        escaped_instructions = _escape_bash_string(task.instructions)
        command_template = Template(agent.command_template)
        command = command_template.render(task_instructions=escaped_instructions)
        logger.info(f"Executing comparison trial command: {command}")

        agent_start_time = time.perf_counter()
        _, _exec_logs = docker_manager.exec_command(
            container=docker_manager.container,
            command=["bash", "-c", command],
            log_filename="agent_output.log",
            timeout=task.timeout,
        )
        agent_duration = time.perf_counter() - agent_start_time
        logger.info(
            f"Comparison trial command execution completed in {agent_duration:.1f}s. "
            f"Output saved to: {trial_dir / 'agent_output.log'}"
        )

        # Handle continuation if configured
        if config.continuation_provider != "none" and agent.command_template_continue is not None:
            try:
                provider: Literal["openai", "azure_openai"] = config.continuation_provider
                model: Literal["gpt-5", "gpt-5.1"] = config.continuation_model
                continuation_response = await interact_with_agent(
                    agent_log=_exec_logs,
                    task_instructions=task.instructions,
                    provider=provider,
                    model=model,
                )

                if continuation_response:
                    escaped_response = _escape_bash_string(continuation_response.lstrip("- "))
                    continuation_template = Template(agent.command_template_continue)
                    continuation_command = continuation_template.render(task_instructions=escaped_response)

                    logger.info(f"Executing continuation command: {continuation_command}")

                    try:
                        _, continuation_logs = docker_manager.exec_command(
                            container=docker_manager.container,
                            command=["bash", "-c", continuation_command],
                            log_filename="agent_continuation.log",
                            timeout=task.timeout,
                        )

                        agent_log_path = trial_dir / "agent_output.log"
                        with agent_log_path.open("a", encoding="utf-8") as f:
                            f.write("\n\n--- CONTINUATION ---\n\n")
                            f.write(f"Continuation prompt:\n{continuation_response}\n\n")
                            f.write("--- CONTINUATION OUTPUT ---\n\n")
                            f.write(continuation_logs)

                    except Exception as e:
                        logger.error(f"Failed to execute continuation command: {e}")
                        agent_log_path = trial_dir / "agent_output.log"
                        with agent_log_path.open("a", encoding="utf-8") as f:
                            f.write(f"\n\n--- CONTINUATION FAILED ---\n{e}\n")
                else:
                    logger.info("No continuation needed - agent task appears complete")

            except Exception as e:
                logger.error(f"Failed to check if agent needs continuation: {e}")
        else:
            if config.continuation_provider == "none":
                logger.info("Agent continuation is disabled (continuation_provider='none')")
            else:
                logger.info("Agent does not have a continuation template - skipping continuation check")

        # Extract /project folder to local directory (excluding dotfiles)
        project_dir = trial_dir / "project"
        logger.info(f"Extracting /project folder to {project_dir}")
        docker_manager.extract_directory_from_container(
            container=docker_manager.container,
            src_path="/project",
            dest_path=project_dir,
            exclude_dotfiles=True,
        )
        logger.info(f"Comparison trial completed. Project extracted to: {project_dir}")

    return project_dir


def _run_tests(
    container: Any,
    task: TaskConfig,
    run_dir: Path,
    docker_manager: DockerManager,
    trial_number: int,
    agent_duration: float,
    eval_recipes_version: str,
    continuation_metadata: dict[str, Any] | None = None,
) -> TrialResult | None:
    """Run test script in container and return results."""
    if continuation_metadata is None:
        continuation_metadata = {}
    try:
        test_id = str(uuid.uuid4())
        logger.info(f"Running tests (trial {trial_number}) with ID: {test_id}")

        if task.test_time_data_dir and task.test_time_data_dir.exists():
            logger.info(f"Copying test-time data directory from {task.test_time_data_dir} to container")
            test_time_data_files = _collect_directory_files(task.test_time_data_dir)
            if test_time_data_files:
                docker_manager.copy_files_to_container(
                    container=container,
                    files=test_time_data_files,
                    dest_path="/project",
                )

        if task.score_eval is None:
            raise ValueError(f"Task '{task.name}' does not have score_eval configured")
        files = {
            "test.py": task.score_eval.test_script.read_bytes(),
            "instructions.txt": task.instructions.encode("utf-8"),
        }
        docker_manager.copy_files_to_container(container=container, files=files, dest_path="/project")

        # Build test command with eval-recipes as a dependency via --with
        # This installs eval-recipes into uv's isolated cache, not into /project
        git_url = f"git+https://github.com/microsoft/eval-recipes@v{eval_recipes_version}"
        test_command_parts = ["uv", "run", "--with", git_url, "--no-project", "/project/test.py"]
        logger.info(f"Running test: {' '.join(test_command_parts)}")
        test_start_time = time.perf_counter()
        _exec_result, full_output = docker_manager.exec_command(
            container=container,
            command=test_command_parts,
            log_filename="test_output.log",
            timeout=task.timeout,
            environment={
                "EVAL_RECIPES_TEST_ID": test_id,
            },
            workdir="/project",
        )
        test_duration = time.perf_counter() - test_start_time
        logger.info(f"Test output saved to: {run_dir / 'test_output.log'}")

        result_file_path = f"/project/.eval_recipes_test_results_{test_id}.json"
        result_output = docker_manager.read_file_from_container(container, result_file_path)
        if result_output:
            result_data = json.loads(result_output)
            trial_result = TrialResult(
                trial_number=trial_number,
                score=result_data["score"],
                metadata={**result_data.get("metadata", {}), **continuation_metadata},
                test_output=full_output,
                agent_duration_seconds=agent_duration,
                test_duration_seconds=test_duration,
            )
            logger.info(f"Test score: {trial_result.score}, metadata: {trial_result.metadata}")
            return trial_result
        else:
            logger.warning(f"Could not read results file: {result_file_path}")
            result_data = {"score": 0, "metadata": {"error": "No results file found"}}
            trial_result = TrialResult(
                trial_number=trial_number,
                score=0,
                metadata={**result_data["metadata"], **continuation_metadata},
                test_output=full_output,
                agent_duration_seconds=agent_duration,
                test_duration_seconds=test_duration,
            )
            return trial_result
    except Exception as e:
        logger.error(f"Failed to run tests: {e}")
        return None


def _build_dockerfile(agent: AgentConfig, task: TaskConfig, base_template_path: Path | None = None) -> str:
    """Build the complete Dockerfile from base template using liquid."""
    if base_template_path is None:
        base_template_path = Path(__file__).parent / "base.dockerfile"

    base_template = base_template_path.read_text()
    template = Template(base_template)

    agent_installation = agent.agent_installation
    if agent.local_source_path:
        copy_command = "COPY agent_source /tmp/agent_source\n"
        agent_installation = copy_command + agent_installation

    return template.render(
        agent_installation=agent_installation,
        task_installation=task.task_installation,
    )


def _collect_directory_files(directory: Path) -> dict[str, bytes]:
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
            logger.debug(f"Loaded .gitignore from {gitignore_path}")
        except Exception as e:
            logger.warning(f"Failed to parse .gitignore at {gitignore_path}: {e}")

    files = {}
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
