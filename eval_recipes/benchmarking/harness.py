# Copyright (c) Microsoft. All rights reserved.

import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
import statistics
import time
from typing import Any
import uuid

from liquid import Template
from loguru import logger
import pathspec
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn
import yaml

from eval_recipes.benchmarking.agent_interacter import interact_with_agent
from eval_recipes.benchmarking.create_html_report import create_html_report
from eval_recipes.benchmarking.docker_manager import DockerManager
from eval_recipes.benchmarking.filters import apply_filters
from eval_recipes.benchmarking.reporting import generate_summary_report, generate_task_report
from eval_recipes.benchmarking.schemas import AgentConfig, AggregatedTaskResult, TaskConfig, TaskInfo, TrialResult


class Harness:
    def __init__(
        self,
        agents_dir: Path | None = None,
        tasks_dir: Path | None = None,
        runs_dir: Path | None = None,
        environment: dict[str, str] | None = None,
        agent_filters: list[str] | None = None,
        task_filters: list[str] | None = None,
        max_parallel_tasks: int = 5,
        num_trials: int = 1,
        enable_agent_continuation: bool = True,
        report_score_threshold: float = 85.0,
        eval_recipes_version: str = "0.0.17",
    ) -> None:
        """
        Initialize the benchmark harness.

        Args:
            agents_dir: Path to agents directory (default: data/agents/)
            tasks_dir: Path to tasks directory (default: data/tasks/)
            runs_dir: Path to base runs directory (default: data/benchmarking/runs/).
                     A timestamped subdirectory will be created under this path.
            environment: Environment variables to pass to containers
            agent_filters: Optional list of filter strings for agents (e.g., ['name=claude_code'])
            task_filters: Optional list of filter strings for tasks (e.g., ['difficulty=medium'])
            max_parallel_tasks: Maximum number of tasks to run in parallel
            num_trials: Number of times to run each task
            enable_agent_continuation: Enable agent continuation checks (default: True)
            report_score_threshold: Minimum score threshold to skip report generation (default: 85.0)
            eval_recipes_version: Version of eval_recipes to install from GitHub for testing
        """
        repo_root = Path(__file__).parents[2]
        self.agents_dir = agents_dir or repo_root / "data" / "agents"
        self.tasks_dir = tasks_dir or repo_root / "data" / "tasks"

        # Always create a timestamped directory under runs_dir
        base_runs_dir = runs_dir or repo_root / "data" / "benchmarking" / "runs"
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
        self.runs_dir = base_runs_dir / timestamp
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        self.base_template = Path(__file__).parents[0] / "base.dockerfile"
        self.environment = environment or {}
        self.agent_filters = agent_filters
        self.task_filters = task_filters
        self.max_parallel_tasks = max_parallel_tasks
        self.num_trials = num_trials
        self.enable_agent_continuation = enable_agent_continuation
        self.report_score_threshold = report_score_threshold

        self.eval_recipes_version = eval_recipes_version

    def _load_agents(self) -> list[AgentConfig]:
        """
        Loads agent configurations from the agents directory.
        """
        agents = []
        if not self.agents_dir.exists():
            logger.warning(f"Agents directory {self.agents_dir} does not exist.")
            return agents

        for agent_dir in self.agents_dir.iterdir():
            if not agent_dir.is_dir():
                continue

            install_file = agent_dir / "install.dockerfile"
            command_template_file = agent_dir / "command_template.txt"
            command_template_continue_file = agent_dir / "command_template_continue.txt"
            agent_yaml_file = agent_dir / "agent.yaml"
            data_dir = agent_dir / "data"

            if not install_file.exists() or not command_template_file.exists() or not agent_yaml_file.exists():
                continue

            with agent_yaml_file.open() as f:
                agent_yaml = yaml.safe_load(f) or {}

            # Handle local_source_path if specified
            local_source_path = None
            if "local_source_path" in agent_yaml:
                local_source_path_str = agent_yaml["local_source_path"]
                local_source_path = Path(local_source_path_str).resolve()
                if not local_source_path.exists():
                    logger.warning(
                        f"Agent '{agent_dir.name}' specifies local_source_path='{local_source_path}' "
                        f"but path does not exist. Skipping this agent."
                    )
                    continue
                if not local_source_path.is_dir():
                    logger.warning(
                        f"Agent '{agent_dir.name}' specifies local_source_path='{local_source_path}' "
                        f"but path is not a directory. Skipping this agent."
                    )
                    continue
                logger.info(f"Agent '{agent_dir.name}' will use local source from: {local_source_path}")

            agents.append(
                AgentConfig(
                    name=agent_dir.name,
                    required_env_vars=agent_yaml.get("required_env_vars", []),
                    agent_installation=install_file.read_text(),
                    command_template=command_template_file.read_text(),
                    command_template_continue=(
                        command_template_continue_file.read_text() if command_template_continue_file.exists() else None
                    ),
                    data_dir=data_dir if data_dir.exists() and data_dir.is_dir() else None,
                    local_source_path=local_source_path,
                )
            )

        if self.agent_filters:
            agents = apply_filters(agents, self.agent_filters)
        logger.info(f"Loaded {len(agents)} agent(s)")
        return agents

    def _load_tasks(self) -> list[TaskConfig]:
        """
        Loads task configurations from the tasks directory.
        """
        tasks = []
        if not self.tasks_dir.exists():
            logger.warning(f"Tasks directory {self.tasks_dir} does not exist.")
            return tasks

        for task_dir in self.tasks_dir.iterdir():
            if not task_dir.is_dir():
                continue

            setup_file = task_dir / "setup.dockerfile"
            instructions_file = task_dir / "instructions.txt"
            test_script = task_dir / "test.py"
            task_yaml_file = task_dir / "task.yaml"
            data_dir = task_dir / "data"

            if not instructions_file.exists() or not test_script.exists() or not task_yaml_file.exists():
                continue

            with task_yaml_file.open() as f:
                task_yaml = yaml.safe_load(f) or {}

            task_info_data = task_yaml.get("task_info")
            if not task_info_data:
                logger.warning(f"Skipping task '{task_dir.name}', missing required 'task_info' field in task.yaml")
                continue

            task_info = TaskInfo(
                difficulty=task_info_data["difficulty"],
                non_deterministic_evals=task_info_data["non_deterministic_evals"],
            )

            tasks.append(
                TaskConfig(
                    name=task_dir.name,
                    required_env_vars=task_yaml.get("required_env_vars", []),
                    task_installation=setup_file.read_text() if setup_file.exists() else "",
                    instructions=instructions_file.read_text(),
                    test_script=test_script,
                    test_command=task_yaml.get("test_command", "uv run --no-project /project/test.py"),
                    data_dir=data_dir if data_dir.exists() and data_dir.is_dir() else None,
                    task_info=task_info,
                )
            )

        if self.task_filters:
            tasks = apply_filters(tasks, self.task_filters)
        logger.info(f"Loaded {len(tasks)} task(s)")
        return tasks

    def _validate_required_env_vars(self, agent: AgentConfig, task: TaskConfig) -> tuple[bool, list[str]]:
        """
        Validate that all required environment variables are provided.

        Returns:
            Tuple of (success, missing_vars) where success is True if all required vars are present
        """
        required_vars = set(agent.required_env_vars + task.required_env_vars)
        missing_vars = [var for var in required_vars if var not in self.environment]
        return len(missing_vars) == 0, missing_vars

    def _get_container_env_vars(self, agent: AgentConfig, task: TaskConfig) -> dict[str, str]:
        """
        Get the environment variables to pass to the container.

        Returns only the environment variables that are required by the agent or task.
        """
        required_vars = set(agent.required_env_vars + task.required_env_vars)
        return {var: self.environment[var] for var in required_vars if var in self.environment}

    def _build_dockerfile(self, agent: AgentConfig, task: TaskConfig) -> str:
        """Build the complete Dockerfile from base template using liquid."""
        base_template = self.base_template.read_text()
        template = Template(base_template)

        # If agent has local source, prepend COPY command to agent_installation
        agent_installation = agent.agent_installation
        if agent.local_source_path:
            copy_command = "COPY agent_source /tmp/agent_source\n"
            agent_installation = copy_command + agent_installation

        return template.render(
            agent_installation=agent_installation,
            task_installation=task.task_installation,
        )

    def _collect_directory_files(self, directory: Path) -> dict[str, bytes]:
        """
        Recursively collect all files from a directory.

        Respects .gitignore patterns if a .gitignore file exists in the directory root.
        Falls back to hardcoded skip patterns if no .gitignore is present.

        Args:
            directory: Path to directory to collect files from

        Returns:
            Dictionary mapping relative file paths to file contents
        """
        # Directories to skip when collecting files (fallback if no .gitignore)
        skip_dirs = {".git", ".venv", "__pycache__", "node_modules", ".pytest_cache", ".mypy_cache", ".ruff_cache"}

        # Try to load .gitignore from the root of the directory
        gitignore_spec: pathspec.PathSpec | None = None
        gitignore_path = directory / ".gitignore"
        if gitignore_path.exists():
            try:
                with gitignore_path.open() as f:
                    gitignore_spec = pathspec.PathSpec.from_lines("gitwildmatch", f)
                logger.debug(f"Loaded .gitignore from {gitignore_path}")
            except Exception as e:
                logger.warning(f"Failed to parse .gitignore at {gitignore_path}: {e}")

        files = {}
        for file_path in directory.rglob("*"):
            if not file_path.is_file():
                continue

            relative_path = file_path.relative_to(directory)

            # Check gitignore patterns first if available
            if gitignore_spec and gitignore_spec.match_file(str(relative_path)):
                continue

            # Fallback to hardcoded skip_dirs
            if any(part in skip_dirs for part in file_path.parts):
                continue

            # Skip compiled Python files
            if file_path.suffix in {".pyc", ".pyo"}:
                continue

            files[str(relative_path)] = file_path.read_bytes()

        return files

    def _aggregate_trial_results(
        self, trials: list[TrialResult], task_name: str, agent_name: str
    ) -> AggregatedTaskResult:
        """
        Aggregate results from multiple trials.

        Args:
            trials: List of TrialResult objects
            task_name: Name of the task
            agent_name: Name of the agent

        Returns:
            AggregatedTaskResult with statistics
        """
        scores = [trial.score for trial in trials]
        num_trials = len(scores)

        mean_score = statistics.mean(scores)
        median_score = statistics.median(scores)
        std_dev = statistics.stdev(scores) if num_trials > 1 else 0.0
        min_score = min(scores)
        max_score = max(scores)
        num_perfect_trials = sum(1 for s in scores if s == 100.0)

        # Calculate duration statistics
        agent_durations = [trial.agent_duration_seconds for trial in trials if trial.agent_duration_seconds is not None]
        test_durations = [trial.test_duration_seconds for trial in trials if trial.test_duration_seconds is not None]

        mean_agent_duration = statistics.mean(agent_durations) if agent_durations else None
        median_agent_duration = statistics.median(agent_durations) if agent_durations else None
        mean_test_duration = statistics.mean(test_durations) if test_durations else None
        median_test_duration = statistics.median(test_durations) if test_durations else None

        return AggregatedTaskResult(
            task_name=task_name,
            agent_name=agent_name,
            num_trials=num_trials,
            trials=trials,
            mean_score=mean_score,
            median_score=median_score,
            std_dev=std_dev,
            min_score=min_score,
            max_score=max_score,
            num_perfect_trials=num_perfect_trials,
            mean_agent_duration_seconds=mean_agent_duration,
            median_agent_duration_seconds=median_agent_duration,
            mean_test_duration_seconds=mean_test_duration,
            median_test_duration_seconds=median_test_duration,
        )

    def _run_tests(
        self,
        container: Any,
        task: TaskConfig,
        run_dir: Path,
        docker_manager: DockerManager,
        trial_number: int,
        agent_duration: float,
        continuation_metadata: dict[str, Any] | None = None,
    ) -> TrialResult | None:
        """Run test script in container and return results."""
        if continuation_metadata is None:
            continuation_metadata = {}
        try:
            # Generate unique test ID - this is to make sure we can identify result files uniquely in the container
            test_id = str(uuid.uuid4())
            logger.info(f"Running tests (trial {trial_number}) with ID: {test_id}")

            # Initialize /project as a uv project
            logger.info("Initializing /project as a uv project")
            _exec_result, _init_output = docker_manager.exec_command(
                container=container,
                command=["uv", "init", "--no-readme", "--no-pin-python", "--name", "test_project"],
                log_filename="uv_init_output.log",
                workdir="/project",
            )

            # Add eval_recipes from GitHub as a Git dependency
            git_url = f"git+https://github.com/microsoft/eval-recipes@v{self.eval_recipes_version}"
            docker_manager.exec_command(
                container=container,
                command=["uv", "add", git_url],
                log_filename="uv_add_eval_recipes_output.log",
                workdir="/project",
            )

            # Copy task data directory if it exists
            if task.data_dir and task.data_dir.exists():
                logger.info(f"Copying data directory from {task.data_dir} to container")
                data_files = self._collect_directory_files(task.data_dir)
                if data_files:
                    docker_manager.copy_files_to_container(
                        container=container,
                        files=data_files,
                        dest_path="/project",
                    )

            # Copy the test script and instructions file
            files = {
                "test.py": task.test_script.read_bytes(),
                "instructions.txt": task.instructions.encode("utf-8"),
            }
            docker_manager.copy_files_to_container(container=container, files=files, dest_path="/project")

            # Execute test script using uv run from /project
            # This uses /project's venv which has eval_recipes installed as an editable dependency
            logger.info(f"Running test: {task.test_command}")
            test_start_time = time.perf_counter()
            _exec_result, full_output = docker_manager.exec_command(
                container=container,
                command=["uv", "run", "test.py"],
                log_filename="test_output.log",
                timeout=1800,
                environment={
                    "EVAL_RECIPES_TEST_ID": test_id,
                },
                workdir="/project",
            )
            test_duration = time.perf_counter() - test_start_time
            logger.info(f"Test output saved to: {run_dir / 'test_output.log'}")

            # Read result file from container
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
                results_file = run_dir / "test_results.json"
                results_file.write_text(json.dumps(result_data, indent=2))
                logger.info(f"Test score: {trial_result.score}, metadata: {trial_result.metadata}")
                return trial_result
            else:
                logger.warning(f"Could not read results file: {result_file_path}")
                # Fallback: return score 0 with error metadata
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

    def _run_single_task_sync(
        self, agent: AgentConfig, task: TaskConfig
    ) -> tuple[Path, Path, str] | tuple[None, None, str]:
        """
        Synchronous implementation of running a single task with multiple trials.

        Args:
            agent: Agent configuration
            task: Task configuration

        Returns:
            Tuple of (base_run_dir, task_dir_path, task_name) if successful, (None, None, task_name) otherwise
        """
        task_name = f"{agent.name} on {task.name}"
        logger.info(f"Running agent '{agent.name}' on task '{task.name}' with {self.num_trials} trial(s)")

        valid, missing_vars = self._validate_required_env_vars(agent, task)
        if not valid:
            logger.error(
                f"Missing required environment variables for agent '{agent.name}' "
                f"and task '{task.name}': {missing_vars}"
            )
            return None, None, task_name

        # Create base directory for this agent-task pair
        base_run_dir = self.runs_dir / f"{agent.name}_{task.name}"
        base_run_dir.mkdir(parents=True, exist_ok=True)
        task_dir_path = self.tasks_dir / task.name

        trial_results: list[TrialResult] = []
        for trial_num in range(1, self.num_trials + 1):
            logger.info(f"Starting trial {trial_num}/{self.num_trials} for {task_name}")
            trial_dir = base_run_dir / f"trial_{trial_num}"
            trial_dir.mkdir(parents=True, exist_ok=True)

            container_env = self._get_container_env_vars(agent, task)
            dockerfile_content = self._build_dockerfile(agent, task)
            image_tag = f"benchmark-{agent.name}-{task.name}-trial{trial_num}".lower()

            # Collect local source files if agent uses local source
            build_context_files: dict[str, bytes] = {}
            if agent.local_source_path:
                logger.info(f"Collecting local source files from {agent.local_source_path}")
                local_source_files = self._collect_directory_files(agent.local_source_path)
                # Add files with agent_source/ prefix so they match COPY command in Dockerfile
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

                # Copy agent data directory if it exists
                if agent.data_dir and agent.data_dir.exists():
                    logger.info(f"Copying agent data directory from {agent.data_dir} to container")
                    agent_data_files = self._collect_directory_files(agent.data_dir)
                    if agent_data_files:
                        docker_manager.copy_files_to_container(
                            container=docker_manager.container,
                            files=agent_data_files,
                            dest_path="/project",
                        )

                # Create command to run agent
                escaped_instructions = escape_bash_string(task.instructions)
                command_template = Template(agent.command_template)
                command = command_template.render(task_instructions=escaped_instructions)
                logger.info(f"Executing command for trial {trial_num}: {command}")

                # Track agent execution time
                agent_start_time = time.perf_counter()
                _exec_result, _exec_logs = docker_manager.exec_command(
                    container=docker_manager.container,
                    command=["bash", "-c", command],
                    log_filename="agent_output.log",
                    timeout=1800,
                )
                agent_duration = time.perf_counter() - agent_start_time
                logger.info(
                    f"Trial {trial_num} command execution completed. Output saved to: {trial_dir / 'agent_output.log'}"
                )

                # Check if agent needs continuation
                continuation_metadata: dict[str, Any] = {}
                if self.enable_agent_continuation and agent.command_template_continue is not None:
                    try:
                        continuation_response = asyncio.run(
                            interact_with_agent(agent_log=_exec_logs, task_instructions=task.instructions)
                        )

                        if continuation_response:
                            continuation_metadata["continuation_occurred"] = True

                            # Render continuation command template
                            escaped_response = escape_bash_string(continuation_response)
                            continuation_template = Template(agent.command_template_continue)
                            continuation_command = continuation_template.render(task_instructions=escaped_response)

                            logger.info(f"Executing continuation command: {continuation_command}")

                            try:
                                # Execute continuation (use different log name temporarily)
                                continuation_start = time.perf_counter()
                                _continuation_result, continuation_logs = docker_manager.exec_command(
                                    container=docker_manager.container,
                                    command=["bash", "-c", continuation_command],
                                    log_filename="agent_continuation.log",
                                    timeout=1800,
                                )
                                continuation_duration = time.perf_counter() - continuation_start

                                # Append continuation logs to main agent_output.log
                                agent_log_path = trial_dir / "agent_output.log"
                                with agent_log_path.open("a") as f:
                                    f.write("\n\n--- CONTINUATION ---\n\n")
                                    f.write(continuation_logs)

                                agent_duration += continuation_duration

                            except Exception as e:
                                logger.error(f"Failed to execute continuation command: {e}")
                                continuation_metadata["continuation_error"] = str(e)
                                # Still append error to log
                                agent_log_path = trial_dir / "agent_output.log"
                                with agent_log_path.open("a") as f:
                                    f.write(f"\n\n--- CONTINUATION FAILED ---\n{e}\n")
                        else:
                            logger.info("No continuation needed - agent task appears complete")
                            continuation_metadata["continuation_occurred"] = False

                    except Exception as e:
                        logger.error(f"Failed to check if agent needs continuation: {e}")
                        continuation_metadata["continuation_check_error"] = str(e)
                else:
                    if not self.enable_agent_continuation:
                        logger.info("Agent continuation is disabled")
                    else:
                        logger.info("Agent does not have a continuation template - skipping continuation check")
                    continuation_metadata["continuation_disabled"] = True

                # Run tests for this trial
                trial_result = self._run_tests(
                    docker_manager.container,
                    task,
                    trial_dir,
                    docker_manager,
                    trial_num,
                    agent_duration,
                    continuation_metadata,
                )

                # Add trial result to list
                if trial_result:
                    trial_results.append(trial_result)
                    logger.info(f"Trial {trial_num} completed with score: {trial_result.score}")
                else:
                    logger.warning(f"Trial {trial_num} failed to produce results")
                    # Create a failed trial result
                    failed_trial_result = TrialResult(
                        trial_number=trial_num,
                        score=0.0,
                        metadata={"error": "Test execution failed"},
                        test_output="",
                        agent_duration_seconds=agent_duration,
                        test_duration_seconds=None,
                    )
                    trial_results.append(failed_trial_result)

        if trial_results:
            aggregated_result = self._aggregate_trial_results(trial_results, task.name, agent.name)

            # Write aggregated results to base directory
            aggregated_file = base_run_dir / "aggregated_results.json"
            aggregated_file.write_text(aggregated_result.model_dump_json(indent=2))
            logger.info(
                f"Aggregated results for {task_name}: mean={aggregated_result.mean_score:.1f}%, "
                f"std_dev={aggregated_result.std_dev:.1f}%, range=[{aggregated_result.min_score:.1f}%, {aggregated_result.max_score:.1f}%]"
            )
        else:
            logger.error(f"No trial results collected for {task_name}")
            return None, None, task_name

        return (base_run_dir, task_dir_path, task_name)

    async def _run_single_task(
        self, agent: AgentConfig, task: TaskConfig, progress: Progress | None = None, task_id: TaskID | None = None
    ) -> tuple[Path, Path] | None:
        """
        Run a single agent on a single task (async wrapper).

        Args:
            agent: Agent configuration
            task: Task configuration
            progress: Optional Rich progress instance for tracking
            task_id: Optional progress task ID for updating

        Returns:
            Tuple of (run_dir, task_dir_path) if successful, None otherwise
        """
        run_dir, task_dir_path, task_name = await asyncio.to_thread(self._run_single_task_sync, agent, task)
        if progress and task_id is not None:
            progress.update(task_id, advance=1, description=f"[green]✓[/green] Last completed: {task_name}")
        if run_dir is None or task_dir_path is None:
            return None
        return (run_dir, task_dir_path)

    async def _generate_single_task_report(
        self,
        run_dir: Path,
        task_dir_path: Path,
        semaphore: asyncio.Semaphore,
        progress: Progress | None = None,
        task_id: TaskID | None = None,
        report_score_threshold: float = 85.0,
    ) -> None:
        """
        Generate a failure report for a single task.

        Args:
            run_dir: Directory containing task execution outputs (base directory for all trials)
            task_dir_path: Directory containing task definition
            semaphore: Semaphore to limit concurrent report generation
            progress: Optional Rich progress instance for tracking
            task_id: Optional progress task ID for updating
            report_score_threshold: Minimum score threshold to skip report generation (default: 85.0)
        """
        logger.info(f"Generating failure report for {run_dir.name}...")
        async with semaphore:
            try:
                await generate_task_report(run_dir, task_dir_path, report_score_threshold)
            except Exception as e:
                logger.error(f"Failed to generate report for {run_dir.name}: {e}")

        # Update progress if provided
        if progress and task_id is not None:
            progress.update(task_id, advance=1, description=f"[green]✓[/green] Last completed: {run_dir.name}")

    async def run(self, generate_reports: bool = False) -> None:
        agents = self._load_agents()
        tasks = self._load_tasks()
        if not agents or not tasks:
            logger.error("No agents or tasks to run. Exiting.")
            return

        # Create progress display
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        )

        with progress:
            task_coroutines = []
            for agent in agents:
                for task in tasks:
                    task_coroutines.append((agent, task))

            task_progress_id = progress.add_task(
                f"Running {len(task_coroutines)} task(s) with max parallelism of {self.max_parallel_tasks}",
                total=len(task_coroutines),
            )
            semaphore = asyncio.Semaphore(self.max_parallel_tasks)

            async def run_with_semaphore(agent: AgentConfig, task: TaskConfig) -> tuple[Path, Path] | None:
                async with semaphore:
                    return await self._run_single_task(agent, task, progress, task_progress_id)

            logger.info(f"Running {len(task_coroutines)} task(s) with max parallelism of {self.max_parallel_tasks}")
            coroutines = [run_with_semaphore(agent, task) for agent, task in task_coroutines]
            results = await asyncio.gather(*coroutines, return_exceptions=True)

            # Track task runs for report generation (filter out None and exceptions)
            task_runs: list[tuple[Path, Path]] = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Task execution failed with exception: {result}")
                elif result is not None and isinstance(result, tuple):
                    task_runs.append(result)

            if generate_reports:
                logger.info("Generating reports...")
                report_progress_id = progress.add_task(
                    f"Generating reports with max parallelism of {self.max_parallel_tasks}",
                    total=len(task_runs),
                )
                # Generate individual task reports for failed tasks in parallel
                report_semaphore = asyncio.Semaphore(self.max_parallel_tasks)
                report_coroutines = [
                    self._generate_single_task_report(
                        run_dir,
                        task_dir_path,
                        report_semaphore,
                        progress,
                        report_progress_id,
                        self.report_score_threshold,
                    )
                    for run_dir, task_dir_path in task_runs
                ]
                if report_coroutines:
                    logger.info(f"Generating task failure reports with max parallelism of {self.max_parallel_tasks}")
                    await asyncio.gather(*report_coroutines)

                # Generate consolidated summary report
                progress.update(report_progress_id, description="Generating consolidated summary report...")
                logger.info("Generating consolidated summary report...")
                try:
                    await generate_summary_report(self.runs_dir)
                except Exception as e:
                    logger.error(f"Failed to generate consolidated report: {e}")

                # Generate HTML report
                progress.update(report_progress_id, description="Generating HTML report...")
                logger.info("Generating HTML report...")
                try:
                    create_html_report(self.runs_dir, self.tasks_dir)
                    logger.info(f"HTML report saved to: {self.runs_dir / 'benchmark_report.html'}")
                except Exception as e:
                    logger.error(f"Failed to generate HTML report: {e}")

                progress.update(report_progress_id, description="[green]✓[/green] All reports completed")


def escape_bash_string(text: str) -> str:
    """
    Escape special characters in a string for safe use in bash commands.
    """
    text = text.replace("\\", "\\\\")
    text = text.replace('"', '\\"')
    text = text.replace("`", "\\`")
    text = text.replace("$", "\\$")
    return text
