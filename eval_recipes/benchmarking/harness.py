# Copyright (c) Microsoft. All rights reserved

from pathlib import Path
from typing import Literal

from loguru import logger
import yaml

from eval_recipes.benchmarking.create_html_report import create_html_report
from eval_recipes.benchmarking.jobs.base import Job, JobContext, JobResult, JobStatus
from eval_recipes.benchmarking.jobs.runner import JobRunner
from eval_recipes.benchmarking.reporting import generate_agent_consolidated_report, generate_trial_report
from eval_recipes.benchmarking.run_trial import DEFAULT_EVAL_RECIPES_VERSION, TrialConfig, run_trial
from eval_recipes.benchmarking.schemas import AgentConfig, ScoreEvalConfig, ScoreRunSpec, TaskConfig, TaskInfo


class TrialJob(Job):
    def __init__(
        self,
        agent: AgentConfig,
        task: TaskConfig,
        trial_num: int,
        base_run_dir: Path,
        config: TrialConfig,
    ):
        self._agent = agent
        self._task = task
        self._trial_num = trial_num
        self._base_run_dir = base_run_dir
        self._config = config

    @property
    def job_id(self) -> str:
        return f"trial:{self._agent.name}:{self._task.name}:{self._trial_num}"

    @property
    def max_retries(self) -> int:
        return 2

    async def run(self, context: JobContext) -> JobResult:
        try:
            result = await run_trial(
                self._agent,
                self._task,
                self._trial_num,
                self._base_run_dir,
                self._config,
            )
            return JobResult(
                status=JobStatus.COMPLETED,
                outputs={"trial_result": result.model_dump()},
            )
        except Exception as e:
            logger.error(f"Trial job failed: {e}")
            return JobResult(status=JobStatus.SKIPPED, error=str(e))


class TrialReportJob(Job):
    """Job that generates a failure report for a single trial."""

    def __init__(
        self,
        agent: AgentConfig,
        task: TaskConfig,
        trial_num: int,
        base_run_dir: Path,
        tasks_dir: Path,
        report_score_threshold: float = 85.0,
    ):
        self._agent = agent
        self._task = task
        self._trial_num = trial_num
        self._base_run_dir = base_run_dir
        self._tasks_dir = tasks_dir
        self._report_score_threshold = report_score_threshold

    @property
    def job_id(self) -> str:
        return f"report:{self._agent.name}:{self._task.name}:{self._trial_num}"

    @property
    def dependencies(self) -> list[str]:
        return [f"trial:{self._agent.name}:{self._task.name}:{self._trial_num}"]

    async def run(self, context: JobContext) -> JobResult:
        try:
            # Get trial result from dependency
            trial_job_id = self.dependencies[0]
            trial_data = context.get_output(trial_job_id, "trial_result")
            if not trial_data:
                return JobResult(status=JobStatus.SKIPPED, error="No trial result found")

            trial_score = trial_data.get("score", 0)
            trial_dir = self._base_run_dir / f"trial_{self._trial_num}"
            task_dir = self._tasks_dir / self._task.name

            report_generated = await generate_trial_report(
                trial_dir=trial_dir,
                task_directory=task_dir,
                trial_score=trial_score,
                trial_number=self._trial_num,
                report_score_threshold=self._report_score_threshold,
            )

            return JobResult(
                status=JobStatus.COMPLETED,
                outputs={"report_generated": report_generated, "trial_score": trial_score},
            )
        except Exception as e:
            logger.error(f"Trial report job failed: {e}")
            return JobResult(status=JobStatus.FAILED, error=str(e))


class ConsolidatedReportJob(Job):
    """Job that generates a consolidated report for all tasks run by an agent."""

    def __init__(
        self,
        agent: AgentConfig,
        task_trials: dict[str, int],
        runs_dir: Path,
    ):
        self._agent = agent
        self._task_trials = task_trials  # Maps task_name -> num_trials
        self._runs_dir = runs_dir

    @property
    def job_id(self) -> str:
        return f"consolidated:{self._agent.name}"

    @property
    def dependencies(self) -> list[str]:
        # Depend on ALL report jobs for this agent across all tasks
        deps = []
        for task_name, num_trials in self._task_trials.items():
            for trial_num in range(1, num_trials + 1):
                deps.append(f"report:{self._agent.name}:{task_name}:{trial_num}")
        return deps

    async def run(self, context: JobContext) -> JobResult:
        try:
            report_generated = await generate_agent_consolidated_report(
                agent_name=self._agent.name,
                runs_dir=self._runs_dir,
                task_trials=self._task_trials,
            )
            return JobResult(
                status=JobStatus.COMPLETED,
                outputs={"report_generated": report_generated},
            )
        except Exception as e:
            logger.error(f"Consolidated report job failed: {e}")
            return JobResult(status=JobStatus.FAILED, error=str(e))


class HtmlReportJob(Job):
    """Job that generates the final HTML benchmark report."""

    def __init__(
        self,
        agent_names: list[str],
        task_names: list[str],
        max_trials: int,
        runs_dir: Path,
        tasks_dir: Path,
    ):
        self._agent_names = agent_names
        self._task_names = task_names
        self._max_trials = max_trials
        self._runs_dir = runs_dir
        self._tasks_dir = tasks_dir

    @property
    def job_id(self) -> str:
        return "html_report"

    @property
    def dependencies(self) -> list[str]:
        # Depend on ALL consolidated report jobs
        return [f"consolidated:{agent_name}" for agent_name in self._agent_names]

    async def run(self, context: JobContext) -> JobResult:
        try:
            create_html_report(
                benchmarks_output_dir=self._runs_dir,
                tasks_directory=self._tasks_dir,
                agent_names=self._agent_names,
                task_names=self._task_names,
                num_trials=self._max_trials,
            )
            return JobResult(
                status=JobStatus.COMPLETED,
                outputs={"report_path": str(self._runs_dir / "benchmark_report.html")},
            )
        except Exception as e:
            logger.error(f"HTML report job failed: {e}")
            return JobResult(status=JobStatus.FAILED, error=str(e))


class Harness:
    def __init__(
        self,
        agents_dir: Path,
        tasks_dir: Path,
        run_definition: ScoreRunSpec,
        runs_dir: Path | None = None,
        environment: dict[str, str] | None = None,
        max_parallel_trials: int = 5,
        continuation_provider: Literal["openai", "azure_openai", "none"] = "none",
        continuation_model: Literal["gpt-5", "gpt-5.1"] = "gpt-5",
        eval_recipes_version: str = DEFAULT_EVAL_RECIPES_VERSION,
        report_score_threshold: float = 85.0,
    ) -> None:
        self.agents_dir = agents_dir
        self.tasks_dir = tasks_dir
        self.run_definition = run_definition
        self.runs_dir = runs_dir or Path.cwd() / ".benchmark_results"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        self.environment = environment or {}
        self.max_parallel_trials = max_parallel_trials
        self.continuation_provider: Literal["openai", "azure_openai", "none"] = continuation_provider
        self.continuation_model: Literal["gpt-5", "gpt-5.1"] = continuation_model
        self.eval_recipes_version = eval_recipes_version
        self.report_score_threshold = report_score_threshold

    async def run(self) -> None:
        # Load all agents and tasks
        agents_by_name = self._load_agents_by_name()
        tasks_by_name = self._load_tasks_by_name()

        # Resolve run definition to (agent, task, num_trials) tuples
        agent_task_trials: list[tuple[AgentConfig, TaskConfig, int]] = []
        for agent_spec in self.run_definition.definitions:
            agent = agents_by_name.get(agent_spec.agent)
            if agent is None:
                raise ValueError(f"Agent '{agent_spec.agent}' not found in {self.agents_dir}")

            for task_spec in agent_spec.tasks:
                task = tasks_by_name.get(task_spec.task)
                if task is None:
                    raise ValueError(f"Task '{task_spec.task}' not found in {self.tasks_dir}")

                # Determine trial count: task-level > agent-level > 1
                num_trials = task_spec.trials or agent_spec.trials or 1
                agent_task_trials.append((agent, task, num_trials))

        if not agent_task_trials:
            logger.error("No agent-task combinations to run. Exiting.")
            return

        trial_config = TrialConfig(
            environment=self.environment,
            continuation_provider=self.continuation_provider,
            continuation_model=self.continuation_model,
            eval_recipes_version=self.eval_recipes_version,
        )

        # Build jobs from agent_task_trials
        jobs: list[Job] = []

        # Track task_trials per agent for consolidated reports
        agent_task_trials_map: dict[str, dict[str, int]] = {}
        all_task_names: set[str] = set()
        all_agent_names: set[str] = set()
        max_trials = 0

        for agent, task, num_trials in agent_task_trials:
            all_agent_names.add(agent.name)
            all_task_names.add(task.name)
            max_trials = max(max_trials, num_trials)

            if agent.name not in agent_task_trials_map:
                agent_task_trials_map[agent.name] = {}
            agent_task_trials_map[agent.name][task.name] = num_trials

            base_run_dir = self.runs_dir / f"{agent.name}_{task.name}"
            base_run_dir.mkdir(parents=True, exist_ok=True)

            for trial_num in range(1, num_trials + 1):
                # Trial job
                jobs.append(
                    TrialJob(
                        agent=agent,
                        task=task,
                        trial_num=trial_num,
                        base_run_dir=base_run_dir,
                        config=trial_config,
                    )
                )

                # Report job (depends on trial)
                jobs.append(
                    TrialReportJob(
                        agent=agent,
                        task=task,
                        trial_num=trial_num,
                        base_run_dir=base_run_dir,
                        tasks_dir=self.tasks_dir,
                        report_score_threshold=self.report_score_threshold,
                    )
                )

        # Create consolidated report jobs (one per agent)
        for agent_name in all_agent_names:
            agent = agents_by_name[agent_name]
            jobs.append(
                ConsolidatedReportJob(
                    agent=agent,
                    task_trials=agent_task_trials_map[agent_name],
                    runs_dir=self.runs_dir,
                )
            )

        # Create HTML report job (depends on all consolidated reports)
        jobs.append(
            HtmlReportJob(
                agent_names=list(all_agent_names),
                task_names=list(all_task_names),
                max_trials=max_trials,
                runs_dir=self.runs_dir,
                tasks_dir=self.tasks_dir,
            )
        )

        logger.info(f"Created {len(jobs)} job(s)")

        runner = JobRunner(
            state_path=self.runs_dir / "jobs.db",
            max_parallel=self.max_parallel_trials,
        )
        runner.add_jobs(jobs)

        await runner.run()
        logger.info("Job execution complete")

    def _load_agents_by_name(self) -> dict[str, AgentConfig]:
        """Load agent configurations from the agents directory."""
        agents: dict[str, AgentConfig] = {}
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

            with agent_yaml_file.open(encoding="utf-8") as f:
                agent_yaml = yaml.safe_load(f) or {}

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

            agents[agent_dir.name] = AgentConfig(
                name=agent_dir.name,
                required_env_vars=agent_yaml.get("required_env_vars", []),
                agent_installation=install_file.read_text(encoding="utf-8"),
                command_template=command_template_file.read_text(encoding="utf-8"),
                command_template_continue=(
                    command_template_continue_file.read_text(encoding="utf-8")
                    if command_template_continue_file.exists()
                    else None
                ),
                data_dir=data_dir if data_dir.exists() and data_dir.is_dir() else None,
                local_source_path=local_source_path,
                agent_log_hint=agent_yaml.get("agent_log_hint"),
            )

        logger.info(f"Loaded {len(agents)} agent(s)")
        return agents

    def _load_tasks_by_name(self) -> dict[str, TaskConfig]:
        """Load task configurations from the tasks directory."""
        tasks: dict[str, TaskConfig] = {}
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
            task_time_data_dir = task_dir / "task_time_data"
            test_time_data_dir = task_dir / "test_time_data"

            if not instructions_file.exists() or not test_script.exists() or not task_yaml_file.exists():
                continue

            with task_yaml_file.open(encoding="utf-8") as f:
                task_yaml = yaml.safe_load(f) or {}

            task_info_data = task_yaml.get("task_info")
            if not task_info_data:
                logger.warning(f"Skipping task '{task_dir.name}', missing required 'task_info' field in task.yaml")
                continue

            task_info = TaskInfo(
                difficulty=task_info_data["difficulty"],
                non_deterministic_evals=task_info_data.get("non_deterministic_evals", False),
            )

            tasks[task_dir.name] = TaskConfig(
                name=task_dir.name,
                eval_type="score",
                required_env_vars=task_yaml.get("required_env_vars", []),
                task_installation=setup_file.read_text(encoding="utf-8") if setup_file.exists() else "",
                instructions=instructions_file.read_text(encoding="utf-8"),
                score_eval=ScoreEvalConfig(
                    test_script=test_script,
                    test_command=task_yaml.get("test_command", "uv run --no-project /project/test.py"),
                ),
                task_time_data_dir=task_time_data_dir
                if task_time_data_dir.exists() and task_time_data_dir.is_dir()
                else None,
                test_time_data_dir=test_time_data_dir
                if test_time_data_dir.exists() and test_time_data_dir.is_dir()
                else None,
                timeout=task_yaml.get("timeout", 1800),
                task_info=task_info,
            )

        logger.info(f"Loaded {len(tasks)} task(s)")
        return tasks
