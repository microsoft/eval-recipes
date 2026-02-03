# Copyright (c) Microsoft. All rights reserved.

from pathlib import Path
from typing import Any

from loguru import logger

from eval_recipes.benchmarking.job_framework.base import Job, JobState
from eval_recipes.benchmarking.job_framework.runner import JobRunner
from eval_recipes.benchmarking.jobs.score.agent_comparison_job import AgentComparisonJob
from eval_recipes.benchmarking.jobs.score.final_analysis_job import FinalAnalysisJob
from eval_recipes.benchmarking.jobs.score.results_aggregation_job import ResultsAggregationJob
from eval_recipes.benchmarking.jobs.score.trial_execution_job import TrialExecutionJob
from eval_recipes.benchmarking.schemas import (
    AgentComparisonJobInput,
    AgentDefinition,
    FinalAnalysisJobInput,
    ResultsAggregationJobInput,
    ScoreBenchmarkDefinition,
    TaskDefinition,
    TrialExecutionJobInput,
)


class ScorePipeline:
    """Creates and runs jobs for score-based benchmarks."""

    def __init__(
        self,
        benchmark: ScoreBenchmarkDefinition,
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

    def create_jobs(self) -> list[Job[Any]]:
        jobs: list[Job[Any]] = []
        all_trial_jobs: list[TrialExecutionJob] = []
        agent_trial_jobs: dict[str, list[TrialExecutionJob]] = {}

        # For each agent in the benchmark definition:
        for agent_def in self.benchmark.score_benchmarks:
            agent = self.agents.get(agent_def.agent_id)
            if agent is None:
                logger.warning(f"Agent '{agent_def.agent_id}' not found, skipping")
                continue

            agent_trial_jobs[agent.id] = []

            # Create a TrialExecutionJob for each task and trial
            for task_name in agent_def.task_names:
                task = self.tasks.get(task_name)
                if task is None:
                    logger.warning(f"Task '{task_name}' not found, skipping")
                    continue

                for trial in range(1, agent_def.trials + 1):
                    # Create combined trial execution job
                    trial_input = TrialExecutionJobInput(
                        agent=agent,
                        task=task,
                        trial_number=trial,
                        continuation_provider=self.benchmark.continuation_provider,
                        continuation_model=self.benchmark.continuation_model,
                        agent_log_hint=agent.agent_logs_paths[0] if agent.agent_logs_paths else None,
                        analysis_score_threshold=self.benchmark.analysis_score_threshold,
                    )
                    trial_job = TrialExecutionJob(trial_input)
                    jobs.append(trial_job)
                    all_trial_jobs.append(trial_job)
                    agent_trial_jobs[agent.id].append(trial_job)

        # Create FinalAnalysisJob for each agent (depends on all TrialExecutionJobs for that agent)
        final_analysis_jobs: list[FinalAnalysisJob] = []
        for agent_id, trial_jobs in agent_trial_jobs.items():
            if trial_jobs:
                final_analysis_input = FinalAnalysisJobInput(
                    agent_id=agent_id,
                    provider=self.benchmark.continuation_provider
                    if self.benchmark.continuation_provider != "none"
                    else "openai",
                    model=self.benchmark.continuation_model,
                )
                final_analysis_job = FinalAnalysisJob(final_analysis_input, trial_jobs)
                jobs.append(final_analysis_job)
                final_analysis_jobs.append(final_analysis_job)

        # Create AgentComparisonJob if 2+ agents (depends on all FinalAnalysisJobs)
        comparison_job: AgentComparisonJob | None = None
        if len(final_analysis_jobs) >= 2:
            comparison_input = AgentComparisonJobInput(
                provider=self.benchmark.continuation_provider
                if self.benchmark.continuation_provider != "none"
                else "openai",
                model=self.benchmark.continuation_model,
            )
            comparison_job = AgentComparisonJob(comparison_input, final_analysis_jobs)
            jobs.append(comparison_job)

        # Create ResultsAggregationJob (depends on all trial and final jobs)
        results_input = ResultsAggregationJobInput(
            include_project_zips=True,
            include_logs=True,
        )
        results_job = ResultsAggregationJob(
            job_input=results_input,
            trial_execution_jobs=all_trial_jobs,
            final_analysis_jobs=final_analysis_jobs,
            agent_comparison_job=comparison_job,
            tasks=self.tasks,
            agents=self.agents,
        )
        jobs.append(results_job)

        logger.info(f"Created {len(jobs)} job(s)")
        return jobs

    async def run(self) -> dict[str, JobState]:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        runner = JobRunner(
            state_path=self.output_dir / "jobs.db",
            max_parallel=self.max_parallel,
            config={
                "output_dir": self.output_dir,
                "environment": self.environment,
            },
        )
        runner.add_jobs(self.create_jobs())
        return await runner.run()
