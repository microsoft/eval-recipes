# Copyright (c) Microsoft. All rights reserved.

from collections import defaultdict
import json
from pathlib import Path
from typing import Literal

from loguru import logger
import yaml

from eval_recipes.benchmarking.aggregate_comparison_report import (
    generate_aggregate_report,
    generate_final_aggregate_report,
)
from eval_recipes.benchmarking.jobs.base import Job, JobContext, JobResult, JobStatus
from eval_recipes.benchmarking.jobs.runner import JobRunner
from eval_recipes.benchmarking.run_trial import DEFAULT_EVAL_RECIPES_VERSION, TrialConfig, run_trial_for_comparison
from eval_recipes.benchmarking.schemas import (
    AgentConfig,
    AggregateReport,
    ComparisonBenchmarkResults,
    ComparisonEvalConfig,
    ComparisonResult,
    ComparisonRunResult,
    ComparisonTaskConfig,
    ComparisonTaskSpec,
    FinalAggregateReport,
    TaskConfig,
    TaskInfo,
)
from eval_recipes.benchmarking.semantic_test_comparison import semantic_test_comparison


def _get_comparison_folder_name(task_name: str, agent_names: list[str]) -> str:
    """Generate a folder name for a comparison.

    Args:
        task_name: Name of the task
        agent_names: List of agent names in the comparison

    Returns:
        Folder name in format: {task}-{agent1}_vs_{agent2}_vs_...
    """
    return f"{task_name}-{'_vs_'.join(agent_names)}"


class ComparisonTrialJob(Job):
    """Job that runs a single agent trial for comparison evaluation."""

    def __init__(
        self,
        agent: AgentConfig,
        task: TaskConfig,
        runs_dir: Path,
        config: TrialConfig,
    ):
        self._agent = agent
        self._task = task
        self._runs_dir = runs_dir
        self._config = config

    @property
    def job_id(self) -> str:
        return f"comparison_trial:{self._agent.name}:{self._task.name}"

    @property
    def max_retries(self) -> int:
        return 2

    async def run(self, context: JobContext) -> JobResult:
        try:
            trial_dir = self._runs_dir / "agent_outputs" / f"{self._task.name}_{self._agent.name}"
            project_dir = await run_trial_for_comparison(
                agent=self._agent,
                task=self._task,
                trial_dir=trial_dir,
                config=self._config,
            )
            return JobResult(
                status=JobStatus.COMPLETED,
                outputs={"project_dir": str(project_dir)},
            )
        except Exception as e:
            logger.error(f"Comparison trial job failed: {e}")
            return JobResult(status=JobStatus.FAILED, error=str(e))


class SemanticComparisonJob(Job):
    """Job that runs semantic_test_comparison on trial outputs."""

    def __init__(
        self,
        task: TaskConfig,
        comparison_folder_name: str,
        comparison_run_num: int,
        agent_names: list[str],
        runs_dir: Path,
        guidelines: str | None = None,
    ):
        self._task = task
        self._comparison_folder_name = comparison_folder_name
        self._comparison_run_num = comparison_run_num
        self._agent_names = agent_names
        self._runs_dir = runs_dir
        self._guidelines = guidelines

    @property
    def job_id(self) -> str:
        return f"semantic_comparison:{self._comparison_folder_name}:{self._comparison_run_num}"

    @property
    def dependencies(self) -> list[str]:
        """Depends on all trial jobs for this task."""
        return [f"comparison_trial:{agent_name}:{self._task.name}" for agent_name in self._agent_names]

    async def run(self, context: JobContext) -> JobResult:
        try:
            # Collect project directories from all trial jobs
            directories: list[Path] = []
            for dep_job_id in self.dependencies:
                project_dir_str = context.get_output(dep_job_id, "project_dir")
                if not project_dir_str:
                    return JobResult(
                        status=JobStatus.FAILED,
                        error=f"Missing project_dir from dependency {dep_job_id}",
                    )
                directories.append(Path(project_dir_str))

            logger.info(f"Running semantic comparison {self._comparison_run_num} for task '{self._task.name}'")

            # Create output directory and log file path
            comparison_results_dir = self._runs_dir / self._comparison_folder_name / f"comp_{self._comparison_run_num}"
            comparison_results_dir.mkdir(parents=True, exist_ok=True)
            log_file = comparison_results_dir / "semantic_comparison.log"

            comparison_result: ComparisonResult = await semantic_test_comparison(
                original_task=self._task.instructions,
                directories=directories,
                guidelines=self._guidelines,
                log_file=log_file,
            )

            # Save individual comparison result
            result_file = comparison_results_dir / "result.json"

            result = ComparisonRunResult(
                task_name=self._task.name,
                comparison_run_num=self._comparison_run_num,
                result=comparison_result,
                agent_names=self._agent_names,
            )
            result_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")

            return JobResult(
                status=JobStatus.COMPLETED,
                outputs={"comparison_result": result.model_dump()},
            )
        except Exception as e:
            logger.error(f"Semantic comparison job failed: {e}")
            return JobResult(status=JobStatus.FAILED, error=str(e))


class AggregateReportJob(Job):
    """Job that generates an aggregate report from comparison results."""

    def __init__(
        self,
        task: TaskConfig,
        comparison_folder_name: str,
        agent_names: list[str],
        runs_dir: Path,
        comparison_runs: int,
        provider: Literal["openai", "azure_openai"] = "openai",
        model: str = "gpt-5",
    ):
        self._task = task
        self._comparison_folder_name = comparison_folder_name
        self._agent_names = agent_names
        self._runs_dir = runs_dir
        self._comparison_runs = comparison_runs
        self._provider: Literal["openai", "azure_openai"] = provider
        self._model = model

    @property
    def job_id(self) -> str:
        return f"aggregate_report:{self._comparison_folder_name}"

    @property
    def dependencies(self) -> list[str]:
        """Depends on all semantic comparison jobs for this task."""
        return [f"semantic_comparison:{self._comparison_folder_name}:{i}" for i in range(1, self._comparison_runs + 1)]

    async def run(self, context: JobContext) -> JobResult:
        try:
            # Collect comparison results from dependencies
            comparison_results: list[ComparisonRunResult] = []
            for dep_job_id in self.dependencies:
                result_data = context.get_output(dep_job_id, "comparison_result")
                if result_data:
                    comparison_results.append(ComparisonRunResult.model_validate(result_data))

            if not comparison_results:
                return JobResult(
                    status=JobStatus.FAILED,
                    error="No comparison results available",
                )

            logger.info(f"Generating aggregate report for task '{self._task.name}'")

            # Generate aggregate report
            report: AggregateReport = await generate_aggregate_report(
                task_name=self._task.name,
                task_instructions=self._task.instructions,
                comparison_results=comparison_results,
                provider=self._provider,
                model=self._model,
            )

            # Save report
            report_file = self._runs_dir / self._comparison_folder_name / "aggregate_report.json"
            report_file.write_text(report.model_dump_json(indent=2), encoding="utf-8")

            logger.info(f"Aggregate report saved to {report_file}")

            return JobResult(
                status=JobStatus.COMPLETED,
                outputs={"aggregate_report": report.model_dump()},
            )
        except Exception as e:
            logger.error(f"Aggregate report job failed: {e}")
            return JobResult(status=JobStatus.FAILED, error=str(e))


class FinalAggregateReportJob(Job):
    """Job that generates a final aggregate report synthesizing all task reports."""

    def __init__(
        self,
        agent_names: list[str],
        aggregate_report_job_ids: list[str],
        runs_dir: Path,
        tasks_dir: Path,
        provider: Literal["openai", "azure_openai"] = "openai",
        model: str = "gpt-5.2",
    ):
        self._agent_names = agent_names
        self._aggregate_report_job_ids = aggregate_report_job_ids
        self._runs_dir = runs_dir
        self._tasks_dir = tasks_dir
        self._provider: Literal["openai", "azure_openai"] = provider
        self._model = model

    @property
    def job_id(self) -> str:
        return "final_aggregate_report"

    @property
    def dependencies(self) -> list[str]:
        """Depends on all aggregate report jobs."""
        return self._aggregate_report_job_ids

    async def run(self, context: JobContext) -> JobResult:
        try:
            # Collect all aggregate reports from dependencies
            aggregate_reports: list[AggregateReport] = []
            task_instructions: dict[str, str] = {}

            for dep_job_id in self.dependencies:
                report_data = context.get_output(dep_job_id, "aggregate_report")
                if report_data:
                    report = AggregateReport.model_validate(report_data)
                    aggregate_reports.append(report)

                    # Load task instructions
                    instructions_path = self._tasks_dir / report.task_name / "instructions.txt"
                    if instructions_path.exists():
                        task_instructions[report.task_name] = instructions_path.read_text(encoding="utf-8")

            if not aggregate_reports:
                return JobResult(
                    status=JobStatus.FAILED,
                    error="No aggregate reports available",
                )

            logger.info(f"Generating final aggregate report from {len(aggregate_reports)} task reports")

            # Compute overall metrics from raw results
            raw_results_path = self._runs_dir / "raw_results.json"
            if not raw_results_path.exists():
                return JobResult(
                    status=JobStatus.FAILED,
                    error="raw_results.json not found",
                )

            raw_data = json.loads(raw_results_path.read_text(encoding="utf-8"))
            benchmark_results = ComparisonBenchmarkResults.model_validate(raw_data)

            overall_avg_rank, overall_win_rate, task_rankings = self._compute_metrics(benchmark_results)

            # Build task summaries
            task_summaries = []
            for report in sorted(aggregate_reports, key=lambda r: r.task_name):
                instructions = task_instructions.get(report.task_name, "Instructions not available")
                # Truncate long instructions
                if len(instructions) > 500:
                    instructions = instructions[:500] + "..."

                # Get rankings for this task
                rankings_str = task_rankings.get(report.task_name, "Rankings not available")

                task_summaries.append(
                    {
                        "name": report.task_name,
                        "instructions": instructions,
                        "rankings": rankings_str,
                        "analysis": report.analysis,
                    }
                )

            # Generate final report
            final_report: FinalAggregateReport = await generate_final_aggregate_report(
                agent_names=self._agent_names,
                overall_avg_rank=overall_avg_rank,
                overall_win_rate=overall_win_rate,
                task_summaries=task_summaries,
                provider=self._provider,
                model=self._model,
            )

            # Save report
            report_file = self._runs_dir / "final_aggregate_report.json"
            report_file.write_text(final_report.model_dump_json(indent=2), encoding="utf-8")

            logger.info(f"Final aggregate report saved to {report_file}")

            return JobResult(
                status=JobStatus.COMPLETED,
                outputs={"final_aggregate_report": final_report.model_dump()},
            )
        except Exception as e:
            logger.error(f"Final aggregate report job failed: {e}")
            return JobResult(status=JobStatus.FAILED, error=str(e))

    def _compute_metrics(
        self,
        benchmark_results: ComparisonBenchmarkResults,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
        """Compute overall metrics from benchmark results.

        Returns:
            Tuple of (overall_avg_rank, overall_win_rate, task_rankings).
        """
        # Collect all ranks per agent across all tasks
        total_ranks: dict[str, list[int]] = defaultdict(list)
        task_rankings: dict[str, str] = {}

        # Group by task first
        task_runs: dict[str, list[ComparisonRunResult]] = defaultdict(list)
        for run in benchmark_results.comparison_runs:
            task_runs[run.task_name].append(run)

        for task_name, runs in task_runs.items():
            # Compute average rank per agent for this task
            agent_ranks_for_task: dict[str, list[int]] = defaultdict(list)
            for run in runs:
                for rank_position, agent_idx in enumerate(run.result.rankings, start=1):
                    agent_name = run.agent_names[agent_idx]
                    agent_ranks_for_task[agent_name].append(rank_position)
                    total_ranks[agent_name].append(rank_position)

            # Build rankings string for this task
            avg_ranks_for_task = {agent: sum(ranks) / len(ranks) for agent, ranks in agent_ranks_for_task.items()}
            sorted_agents = sorted(avg_ranks_for_task.items(), key=lambda x: x[1])
            rankings_parts = [f"{agent}: {avg:.2f}" for agent, avg in sorted_agents]
            task_rankings[task_name] = ", ".join(rankings_parts)

        # Overall Average Rank
        overall_avg_rank = {agent: sum(ranks) / len(ranks) for agent, ranks in total_ranks.items() if ranks}

        # Overall Win Rate (percentage ranked #1)
        total_wins: dict[str, int] = defaultdict(int)
        total_comparisons = len(benchmark_results.comparison_runs)

        for run in benchmark_results.comparison_runs:
            winner_idx = run.result.rankings[0]  # First in rankings is the winner
            winner_name = run.agent_names[winner_idx]
            total_wins[winner_name] += 1

        overall_win_rate = {
            agent: (wins / total_comparisons * 100) if total_comparisons > 0 else 0
            for agent, wins in total_wins.items()
        }

        # Ensure all agents appear in win rate even if they have 0 wins
        for agent in self._agent_names:
            if agent not in overall_win_rate:
                overall_win_rate[agent] = 0

        return overall_avg_rank, overall_win_rate, task_rankings


class ComparisonHarness:
    """Harness for running comparison-based benchmarks.

    This harness uses explicit task-to-agents associations for head-to-head
    comparisons evaluated via semantic_test_comparison.
    """

    def __init__(
        self,
        agents_dir: Path,
        tasks_dir: Path,
        runs_dir: Path | None = None,
        environment: dict[str, str] | None = None,
        max_parallel: int = 5,
        comparison_runs: int = 3,
        continuation_provider: Literal["openai", "azure_openai", "none"] = "none",
        continuation_model: Literal["gpt-5", "gpt-5.1"] = "gpt-5",
        eval_recipes_version: str = DEFAULT_EVAL_RECIPES_VERSION,
        report_score_threshold: float = 85.0,
    ) -> None:
        """Initialize comparison harness.

        Args:
            agents_dir: Directory containing agent configs.
            tasks_dir: Directory containing task configs.
            runs_dir: Directory to store run outputs (default: .comparison_results/).
            environment: Environment variables to pass to containers.
            max_parallel: Maximum parallel job executions.
            comparison_runs: Number of semantic_test_comparison runs.
            continuation_provider: Provider for agent continuation.
            continuation_model: Model for continuation.
            eval_recipes_version: Version of eval-recipes to use.
            report_score_threshold: Score threshold for reports.
        """
        self.runs_dir = runs_dir or Path.cwd() / ".comparison_results"
        self.agents_dir = agents_dir
        self.tasks_dir = tasks_dir
        self.environment = environment or {}
        self.max_parallel = max_parallel
        self.comparison_runs = comparison_runs
        self.continuation_provider: Literal["openai", "azure_openai", "none"] = continuation_provider
        self.continuation_model: Literal["gpt-5", "gpt-5.1"] = continuation_model
        self.eval_recipes_version = eval_recipes_version
        self.report_score_threshold = report_score_threshold

        self.runs_dir.mkdir(parents=True, exist_ok=True)

    async def run(
        self,
        comparison_specs: list[ComparisonTaskSpec],
    ) -> Path:
        """Run comparison benchmark.

        Args:
            comparison_specs: List of comparison task specifications.

        Returns:
            Path to the results JSON file.
        """

        # Phase 1: Resolve specs to configs
        logger.info(f"Resolving {len(comparison_specs)} comparison spec(s)")
        configs = self._resolve_comparison_configs(comparison_specs, self.tasks_dir)

        # Phase 2: Build job DAG
        jobs = self._build_jobs(configs, self.tasks_dir)
        logger.info(f"Created {len(jobs)} jobs")

        # Phase 3: Run jobs with JobRunner
        runner = JobRunner(
            state_path=self.runs_dir / "jobs.db",
            max_parallel=self.max_parallel,
        )
        runner.add_jobs(jobs)
        await runner.run()
        logger.info("Job execution complete")

        # Phase 4: Collect results and write output
        return self._collect_and_write_results(configs, runner)

    def _build_jobs(self, configs: list[ComparisonTaskConfig], tasks_dir: Path) -> list[Job]:
        """Build the job DAG for all comparison configs."""
        jobs: list[Job] = []
        created_trial_job_ids: set[str] = set()
        aggregate_report_job_ids: list[str] = []
        all_agent_names: set[str] = set()

        trial_config = TrialConfig(
            environment=self.environment,
            continuation_provider=self.continuation_provider,
            continuation_model=self.continuation_model,
            eval_recipes_version=self.eval_recipes_version,
        )

        for config in configs:
            agent_names = [agent.name for agent in config.agents]
            comparison_folder_name = _get_comparison_folder_name(config.task.name, agent_names)
            all_agent_names.update(agent_names)

            # Create trial jobs for each agent (deduplicated across comparisons)
            for agent in config.agents:
                job_id = f"comparison_trial:{agent.name}:{config.task.name}"
                if job_id not in created_trial_job_ids:
                    created_trial_job_ids.add(job_id)
                    jobs.append(
                        ComparisonTrialJob(
                            agent=agent,
                            task=config.task,
                            runs_dir=self.runs_dir,
                            config=trial_config,
                        )
                    )

            # Create semantic comparison jobs for each comparison run
            for comparison_run_num in range(1, self.comparison_runs + 1):
                jobs.append(
                    SemanticComparisonJob(
                        task=config.task,
                        comparison_folder_name=comparison_folder_name,
                        comparison_run_num=comparison_run_num,
                        agent_names=agent_names,
                        runs_dir=self.runs_dir,
                        guidelines=config.guidelines,
                    )
                )

            # Create aggregate report job (depends on all semantic comparison jobs)
            aggregate_report_job_id = f"aggregate_report:{comparison_folder_name}"
            aggregate_report_job_ids.append(aggregate_report_job_id)
            jobs.append(
                AggregateReportJob(
                    task=config.task,
                    comparison_folder_name=comparison_folder_name,
                    agent_names=agent_names,
                    runs_dir=self.runs_dir,
                    comparison_runs=self.comparison_runs,
                )
            )

        # Create final aggregate report job (depends on all aggregate report jobs)
        if aggregate_report_job_ids:
            jobs.append(
                FinalAggregateReportJob(
                    agent_names=list(all_agent_names),
                    aggregate_report_job_ids=aggregate_report_job_ids,
                    runs_dir=self.runs_dir,
                    tasks_dir=tasks_dir,
                )
            )

        return jobs

    def _collect_and_write_results(
        self,
        configs: list[ComparisonTaskConfig],
        runner: JobRunner,
    ) -> Path:
        """Collect results from completed jobs and write to file."""
        all_results: list[ComparisonRunResult] = []

        for config in configs:
            agent_names = [agent.name for agent in config.agents]
            comparison_folder_name = _get_comparison_folder_name(config.task.name, agent_names)

            for comparison_run_num in range(1, self.comparison_runs + 1):
                job_id = f"semantic_comparison:{comparison_folder_name}:{comparison_run_num}"
                state = runner.get_state(job_id)

                if state and state.status == JobStatus.COMPLETED:
                    result_data = state.outputs.get("comparison_result")
                    if result_data:
                        all_results.append(ComparisonRunResult.model_validate(result_data))

        benchmark_results = ComparisonBenchmarkResults(comparison_runs=all_results)
        return self._write_results(benchmark_results)

    def _resolve_comparison_configs(
        self,
        comparison_specs: list[ComparisonTaskSpec],
        tasks_dir: Path,
    ) -> list[ComparisonTaskConfig]:
        """Resolve specs to full configs by loading agent and task configs.

        Args:
            comparison_specs: Specifications with names.
            tasks_dir: Directory containing task configs.

        Returns:
            Fully resolved comparison configs.
        """
        # Load all agents
        agents_by_name = self._load_agents_by_name()

        configs = []
        for spec in comparison_specs:
            # Load task
            task = self._load_task(tasks_dir, spec.task_name)
            if task is None:
                raise ValueError(f"Task '{spec.task_name}' not found in {tasks_dir}")

            # Resolve agents
            agents = []
            for agent_name in spec.agent_names:
                if agent_name not in agents_by_name:
                    raise ValueError(f"Agent '{agent_name}' not found in {self.agents_dir}")
                agents.append(agents_by_name[agent_name])

            if len(agents) < 2:
                raise ValueError(f"Task '{spec.task_name}' requires at least 2 agents, got {len(agents)}")

            # Get guidelines from task config
            guidelines = task.comparison_eval.guidelines if task.comparison_eval is not None else None

            configs.append(
                ComparisonTaskConfig(
                    task=task,
                    agents=agents,
                    guidelines=guidelines,
                )
            )

        return configs

    def _load_agents_by_name(self) -> dict[str, AgentConfig]:
        """Load all agent configurations from the agents directory."""
        agents = {}
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
                if not local_source_path.exists() or not local_source_path.is_dir():
                    logger.warning(
                        f"Agent '{agent_dir.name}' specifies local_source_path='{local_source_path}' "
                        f"but path does not exist or is not a directory. Skipping."
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
            )

        logger.info(f"Loaded {len(agents)} agent(s)")
        return agents

    def _load_task(self, tasks_dir: Path, task_name: str) -> TaskConfig | None:
        """Load a single task configuration by name."""
        task_dir = tasks_dir / task_name
        if not task_dir.exists() or not task_dir.is_dir():
            return None

        setup_file = task_dir / "setup.dockerfile"
        instructions_file = task_dir / "instructions.txt"
        task_yaml_file = task_dir / "task.yaml"
        task_time_data_dir = task_dir / "task_time_data"
        test_time_data_dir = task_dir / "test_time_data"

        if not instructions_file.exists() or not task_yaml_file.exists():
            return None

        with task_yaml_file.open(encoding="utf-8") as f:
            task_yaml = yaml.safe_load(f) or {}

        task_info_data = task_yaml.get("task_info")
        if not task_info_data:
            logger.warning(f"Task '{task_name}' missing required 'task_info' field in task.yaml")
            return None

        task_info = TaskInfo(
            difficulty=task_info_data["difficulty"],
            non_deterministic_evals=task_info_data.get("non_deterministic_evals", False),
            categories=task_info_data.get("categories", []),
        )

        # Determine eval_type from yaml, default to "comparison" for comparison harness
        eval_type = task_yaml.get("eval_type", "comparison")

        # Build comparison_eval config if applicable
        comparison_eval = None
        if eval_type in ("comparison", "both"):
            comparison_eval_data = task_yaml.get("comparison_eval", {})
            comparison_eval = ComparisonEvalConfig(
                guidelines=comparison_eval_data.get("guidelines"),
            )

        return TaskConfig(
            name=task_name,
            eval_type=eval_type,
            required_env_vars=task_yaml.get("required_env_vars", []),
            task_installation=setup_file.read_text(encoding="utf-8") if setup_file.exists() else "",
            instructions=instructions_file.read_text(encoding="utf-8"),
            task_time_data_dir=task_time_data_dir
            if task_time_data_dir.exists() and task_time_data_dir.is_dir()
            else None,
            test_time_data_dir=test_time_data_dir
            if test_time_data_dir.exists() and test_time_data_dir.is_dir()
            else None,
            timeout=task_yaml.get("timeout", 1800),
            task_info=task_info,
            comparison_eval=comparison_eval,
        )

    def _write_results(
        self,
        results: ComparisonBenchmarkResults,
    ) -> Path:
        """Write benchmark results to JSON file.

        Args:
            results: All comparison results.

        Returns:
            Path to the results file.
        """
        results_file = self.runs_dir / "raw_results.json"
        results_file.write_text(results.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"Results written to {results_file}")
        return results_file
