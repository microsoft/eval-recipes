# Copyright (c) Microsoft. All rights reserved.

from pathlib import Path
from typing import Any

from loguru import logger

from eval_recipes.benchmarking.job_framework.base import Job, JobState
from eval_recipes.benchmarking.job_framework.runner import JobRunner
from eval_recipes.benchmarking.jobs.comparison.comparison_aggregation_job import ComparisonAggregationJob
from eval_recipes.benchmarking.jobs.comparison.comparison_final_analysis_job import ComparisonFinalAnalysisJob
from eval_recipes.benchmarking.jobs.comparison.comparison_results_aggregation_job import ComparisonResultsAggregationJob
from eval_recipes.benchmarking.jobs.comparison.comparison_trial_job import ComparisonTrialJob
from eval_recipes.benchmarking.jobs.comparison.semantic_comparison_job import SemanticComparisonJob
from eval_recipes.benchmarking.schemas import (
    AgentDefinition,
    ComparisonAggregationJobInput,
    ComparisonBenchmarkDefinition,
    ComparisonEvalConfig,
    ComparisonFinalAnalysisJobInput,
    ComparisonResultsAggregationJobInput,
    ComparisonTrialJobInput,
    SemanticComparisonJobInput,
    TaskDefinition,
)


class ComparisonPipeline:
    """Creates and runs jobs for comparison-based benchmarks."""

    def __init__(
        self,
        benchmark: ComparisonBenchmarkDefinition,
        agents: dict[str, AgentDefinition],
        tasks: dict[str, TaskDefinition],
        output_dir: Path,
        max_parallel: int = 5,
        environment: dict[str, str] | None = None,
    ) -> None:
        self.benchmark = benchmark
        self.agents = agents
        self.tasks = tasks
        self.output_dir = output_dir
        self.max_parallel = max_parallel
        self.environment = environment or {}

        self._task_trial_jobs: dict[str, list[ComparisonTrialJob]] = {}
        self._task_comparison_jobs: dict[str, list[SemanticComparisonJob]] = {}
        self._aggregation_jobs: list[ComparisonAggregationJob] = []
        self._all_trial_jobs: list[ComparisonTrialJob] = []
        self._all_comparison_jobs: list[SemanticComparisonJob] = []
        self._final_analysis_job: ComparisonFinalAnalysisJob | None = None

    def create_jobs(self) -> list[Job[Any]]:
        """Create all jobs for comparison benchmarks.

        For each comparison benchmark (task + agent_ids):
            For each agent_id:
                Create ComparisonTrialJob (wraps ExecuteAgentJob and ExtractProjectJob)
            For each comparison run (1 to comparison_runs):
                Create SemanticComparisonJob (depends on all ComparisonTrialJobs for this task)
        """
        jobs: list[Job[Any]] = []
        self._task_trial_jobs = {}
        self._task_comparison_jobs = {}
        self._aggregation_jobs = []
        self._all_trial_jobs = []
        self._all_comparison_jobs = []
        self._final_analysis_job = None

        for comparison_def in self.benchmark.comparison_benchmarks:
            task = self.tasks.get(comparison_def.task_name)
            if task is None:
                logger.warning(f"Task '{comparison_def.task_name}' not found, skipping")
                continue

            self._task_trial_jobs[task.name] = []

            for agent_id in comparison_def.agent_ids:
                agent = self.agents.get(agent_id)
                if agent is None:
                    logger.warning(f"Agent '{agent_id}' not found, skipping")
                    continue

                # Create ComparisonTrialJob (wraps agent execution and project extraction)
                trial_input = ComparisonTrialJobInput(
                    agent=agent,
                    task=task,
                    trial_number=1,
                    continuation_provider=self.benchmark.continuation_provider,
                    continuation_model=self.benchmark.continuation_model,
                )
                trial_job = ComparisonTrialJob(trial_input)
                jobs.append(trial_job)
                self._task_trial_jobs[task.name].append(trial_job)
                self._all_trial_jobs.append(trial_job)

            # Create SemanticComparisonJobs for this task (only if we have at least 2 trial jobs)
            task_trial_jobs = self._task_trial_jobs[task.name]
            if len(task_trial_jobs) >= 2:
                # Get guidelines from task's ComparisonEvalConfig
                guidelines = self._get_comparison_guidelines(task)

                # Initialize comparison jobs list for this task
                self._task_comparison_jobs[task.name] = []

                # Create a comparison job for each run
                for run_number in range(1, self.benchmark.comparison_runs + 1):
                    comparison_input = SemanticComparisonJobInput(
                        task_name=task.name,
                        task_instructions=task.instructions or "",
                        comparison_run_number=run_number,
                        guidelines=guidelines,
                    )
                    comparison_job = SemanticComparisonJob(comparison_input, task_trial_jobs)
                    jobs.append(comparison_job)
                    self._task_comparison_jobs[task.name].append(comparison_job)
                    self._all_comparison_jobs.append(comparison_job)

                # Create ComparisonAggregationJob to aggregate all comparison runs for this task
                task_comparison_jobs = self._task_comparison_jobs[task.name]
                if len(task_comparison_jobs) >= 1:
                    aggregation_input = ComparisonAggregationJobInput(
                        task_name=task.name,
                        task_instructions=task.instructions or "",
                        provider=self.benchmark.continuation_provider
                        if self.benchmark.continuation_provider != "none"
                        else "openai",
                        model=self.benchmark.continuation_model,
                    )
                    aggregation_job = ComparisonAggregationJob(aggregation_input, task_comparison_jobs)
                    jobs.append(aggregation_job)
                    self._aggregation_jobs.append(aggregation_job)

        # Create ComparisonFinalAnalysisJob to aggregate all per-task reports
        if len(self._aggregation_jobs) >= 1:
            final_input = ComparisonFinalAnalysisJobInput(
                provider=self.benchmark.continuation_provider
                if self.benchmark.continuation_provider != "none"
                else "openai",
                model=self.benchmark.continuation_model,
            )
            self._final_analysis_job = ComparisonFinalAnalysisJob(final_input, self._aggregation_jobs)
            jobs.append(self._final_analysis_job)

        # Create ComparisonResultsAggregationJob to produce structured JSON output
        results_input = ComparisonResultsAggregationJobInput(include_project_zips=True)
        results_job = ComparisonResultsAggregationJob(
            job_input=results_input,
            comparison_trial_jobs=self._all_trial_jobs,
            semantic_comparison_jobs=self._all_comparison_jobs,
            aggregation_jobs=self._aggregation_jobs,
            final_analysis_job=self._final_analysis_job,
            tasks=self.tasks,
        )
        jobs.append(results_job)

        logger.info(f"Created {len(jobs)} comparison job(s)")
        return jobs

    def _get_comparison_guidelines(self, task: TaskDefinition) -> str | None:
        """Get comparison guidelines from task's evaluation configs."""
        for eval_config in task.evaluation_configs:
            if isinstance(eval_config, ComparisonEvalConfig):
                return eval_config.guidelines
        return None

    async def run(self) -> dict[str, JobState]:
        """Run the comparison pipeline."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        runner = JobRunner(
            state_path=self.output_dir / "comparison_jobs.db",
            max_parallel=self.max_parallel,
            config={
                "output_dir": self.output_dir,
                "environment": self.environment,
            },
        )
        runner.add_jobs(self.create_jobs())
        return await runner.run()
