# Copyright (c) Microsoft. All rights reserved.

import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
import statistics
from typing import Any
import uuid

from liquid import Template
from loguru import logger
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn
import yaml

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
        eval_recipes_version: str = "0.0.12",
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
            agent_yaml_file = agent_dir / "agent.yaml"

            if not install_file.exists() or not command_template_file.exists() or not agent_yaml_file.exists():
                continue

            with agent_yaml_file.open() as f:
                agent_yaml = yaml.safe_load(f) or {}

            agents.append(
                AgentConfig(
                    name=agent_dir.name,
                    required_env_vars=agent_yaml.get("required_env_vars", []),
                    agent_installation=install_file.read_text(),
                    command_template=command_template_file.read_text(),
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
        return template.render(
            agent_installation=agent.agent_installation,
            task_installation=task.task_installation,
        )

    def _collect_directory_files(self, directory: Path) -> dict[str, bytes]:
        """
        Recursively collect all files from a directory.

        Args:
            directory: Path to directory to collect files from

        Returns:
            Dictionary mapping relative file paths to file contents
        """
        files = {}
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(directory)
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
        )

    def _run_tests(
        self,
        container: Any,
        task: TaskConfig,
        run_dir: Path,
        docker_manager: DockerManager,
        trial_number: int,
    ) -> TrialResult | None:
        """Run test script in container and return results."""
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
            logger.info(f"Test output saved to: {run_dir / 'test_output.log'}")

            # Read result file from container
            result_file_path = f"/project/.eval_recipes_test_results_{test_id}.json"
            result_output = docker_manager.read_file_from_container(container, result_file_path)
            if result_output:
                result_data = json.loads(result_output)
                trial_result = TrialResult(
                    trial_number=trial_number,
                    score=result_data["score"],
                    metadata=result_data.get("metadata", {}),
                    test_output=full_output,
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
                    metadata=result_data["metadata"],
                    test_output=full_output,
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

            with DockerManager(
                log_dir=trial_dir, dockerfile=dockerfile_content, image_tag=image_tag, container_env=container_env
            ) as docker_manager:
                assert docker_manager.container is not None
                logger.info(f"Built image: {docker_manager.actual_image_tag}")
                logger.info(f"Container {docker_manager.container_id} started for trial {trial_num}")

                # Create command to run agent
                escaped_instructions = escape_bash_string(task.instructions)
                command_template = Template(agent.command_template)
                command = command_template.render(task_instructions=escaped_instructions)
                logger.info(f"Executing command for trial {trial_num}: {command}")

                _exec_result, _exec_logs = docker_manager.exec_command(
                    container=docker_manager.container,
                    command=["bash", "-c", command],
                    log_filename="agent_output.log",
                    timeout=1800,
                )
                logger.info(
                    f"Trial {trial_num} command execution completed. Output saved to: {trial_dir / 'agent_output.log'}"
                )

                # Run tests for this trial
                trial_result = self._run_tests(docker_manager.container, task, trial_dir, docker_manager, trial_num)

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
    ) -> None:
        """
        Generate a failure report for a single task.

        Args:
            run_dir: Directory containing task execution outputs (base directory for all trials)
            task_dir_path: Directory containing task definition
            semaphore: Semaphore to limit concurrent report generation
            progress: Optional Rich progress instance for tracking
            task_id: Optional progress task ID for updating
        """
        logger.info(f"Generating failure report for {run_dir.name}...")
        async with semaphore:
            try:
                await generate_task_report(run_dir, task_dir_path)
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
                        run_dir, task_dir_path, report_semaphore, progress, report_progress_id
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
