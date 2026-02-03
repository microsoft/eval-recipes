# Copyright (c) Microsoft. All rights reserved.

"""Job for aggregating all comparison benchmark results into structured JSON."""

from collections import defaultdict
import contextlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import zipfile

from loguru import logger

from eval_recipes.benchmarking.job_framework.base import Job, JobContext, JobResult, JobStatus
from eval_recipes.benchmarking.jobs.comparison.comparison_aggregation_job import ComparisonAggregationJob
from eval_recipes.benchmarking.jobs.comparison.comparison_final_analysis_job import ComparisonFinalAnalysisJob
from eval_recipes.benchmarking.jobs.comparison.comparison_trial_job import ComparisonTrialJob
from eval_recipes.benchmarking.jobs.comparison.semantic_comparison_job import SemanticComparisonJob
from eval_recipes.benchmarking.reporting.create_comparison_html_report import create_comparison_html_report
from eval_recipes.benchmarking.schemas import (
    ComparisonBenchmarkManifest,
    ComparisonBenchmarkSummary,
    ComparisonOverviewMetrics,
    ComparisonResultsAggregationJobInput,
    ComparisonResultsAggregationJobOutput,
    ComparisonTaskMetrics,
    ComparisonTrialData,
    TaskDefinition,
)


class ComparisonResultsAggregationJob(Job[ComparisonResultsAggregationJobOutput]):
    """Job that aggregates all comparison benchmark results into structured JSON.

    Creates:
    - results/comparison_manifest.json: Full benchmark manifest with all data
    - results/comparison_summary.json: Lightweight summary for dashboard display
    - results/agents/{agent_id}/tasks/{task_name}/project.zip: Project zips (optional)
    """

    output_model = ComparisonResultsAggregationJobOutput

    def __init__(
        self,
        job_input: ComparisonResultsAggregationJobInput,
        comparison_trial_jobs: list[ComparisonTrialJob],
        semantic_comparison_jobs: list[SemanticComparisonJob],
        aggregation_jobs: list[ComparisonAggregationJob],
        final_analysis_job: ComparisonFinalAnalysisJob | None,
        tasks: dict[str, TaskDefinition],
    ) -> None:
        self._input = job_input
        self._comparison_trial_jobs = comparison_trial_jobs
        self._semantic_comparison_jobs = semantic_comparison_jobs
        self._aggregation_jobs = aggregation_jobs
        self._final_analysis_job = final_analysis_job
        self._tasks = tasks

    @property
    def job_id(self) -> str:
        return "comparison_results_aggregation"

    @property
    def soft_dependencies(self) -> list[Job[Any]]:
        deps: list[Job[Any]] = []
        deps.extend(self._comparison_trial_jobs)
        deps.extend(self._semantic_comparison_jobs)
        deps.extend(self._aggregation_jobs)
        if self._final_analysis_job:
            deps.append(self._final_analysis_job)
        return deps

    async def run(self, context: JobContext) -> JobResult[ComparisonResultsAggregationJobOutput]:
        output_dir: Path = context.config.get("output_dir", Path.cwd() / ".benchmark_results_v2")

        logger.info(f"Starting job: {self.job_id}")

        # Create results directory
        results_dir = output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        try:
            benchmark_timestamp = datetime.now(UTC).isoformat()

            # Collect semantic comparison outputs by task
            task_comparisons: dict[str, list[tuple[int, dict[str, int], str]]] = defaultdict(list)
            for job in self._semantic_comparison_jobs:
                output = context.try_get_output(job)
                if output:
                    task_comparisons[output.task_name].append(
                        (output.comparison_run_number, output.rankings, output.reasoning)
                    )

            # Collect comparison trial job outputs by task+agent
            project_dirs: dict[str, dict[str, str]] = {}  # task_name -> agent_id -> project_dir
            for job in self._comparison_trial_jobs:
                output = context.try_get_output(job)
                if output:
                    project_dirs.setdefault(output.task_name, {})[output.agent_id] = output.project_dir

            # Collect aggregation job outputs by task
            aggregation_outputs: dict[str, tuple[str, str]] = {}  # task_name -> (analysis, path)
            for job in self._aggregation_jobs:
                output = context.try_get_output(job)
                if output:
                    aggregation_outputs[output.task_name] = (output.analysis_report, output.report_path)

            # Collect final analysis
            final_analysis_report: str | None = None
            final_analysis_report_path: str | None = None
            if self._final_analysis_job:
                final_output = context.try_get_output(self._final_analysis_job)
                if final_output:
                    final_analysis_report = final_output.analysis_report
                    final_analysis_report_path = final_output.report_path

            # Collect all unique agent IDs
            all_agent_ids: set[str] = set()
            for agent_dict in project_dirs.values():
                all_agent_ids.update(agent_dict.keys())

            # Build per-task metrics
            all_task_metrics: list[ComparisonTaskMetrics] = []

            for task_name in sorted(task_comparisons.keys()):
                comparisons = task_comparisons[task_name]
                task_def = self._tasks.get(task_name)
                if not task_def:
                    logger.warning(f"Task definition not found for '{task_name}', skipping")
                    continue

                # Build agent_ranks from comparison outputs
                agent_ranks: dict[str, list[int]] = defaultdict(list)
                trials: list[ComparisonTrialData] = []

                # Sort by comparison_run_number for consistent ordering
                for run_num, rankings, reasoning in sorted(comparisons, key=lambda x: x[0]):
                    for agent_id, rank in rankings.items():
                        agent_ranks[agent_id].append(rank)
                    trials.append(
                        ComparisonTrialData(comparison_run_number=run_num, rankings=rankings, reasoning=reasoning)
                    )

                # Compute metrics
                agent_avg_rank = {a: sum(r) / len(r) for a, r in agent_ranks.items() if r}
                agent_win_rate = {
                    a: (sum(1 for r in ranks if r == 1) / len(ranks) * 100) if ranks else 0.0
                    for a, ranks in agent_ranks.items()
                }
                kendalls_w = _calculate_kendalls_w(dict(agent_ranks))

                # Get aggregate analysis
                aggregate_analysis = ""
                aggregate_analysis_path: str | None = None
                if task_name in aggregation_outputs:
                    aggregate_analysis, aggregate_analysis_path = aggregation_outputs[task_name]
                    # Convert to relative path
                    if aggregate_analysis_path:
                        with contextlib.suppress(ValueError):
                            aggregate_analysis_path = str(Path(aggregate_analysis_path).relative_to(output_dir))

                # Create project zips if requested
                project_zip_paths: dict[str, str] = {}
                if self._input.include_project_zips:
                    for agent_id, proj_dir in project_dirs.get(task_name, {}).items():
                        zip_rel_path = Path("agents") / agent_id / "tasks" / task_name / "project.zip"
                        zip_path = results_dir / zip_rel_path
                        zip_path.parent.mkdir(parents=True, exist_ok=True)
                        _create_zip_with_safe_timestamps(Path(proj_dir), zip_path)
                        project_zip_paths[agent_id] = str(zip_rel_path)

                task_metrics = ComparisonTaskMetrics(
                    task_name=task_name,
                    task_instructions=task_def.instructions or "",
                    task_info=task_def.task_info,
                    agent_ranks=dict(agent_ranks),
                    agent_avg_rank=agent_avg_rank,
                    agent_win_rate=agent_win_rate,
                    agreement_kendalls_w=kendalls_w,
                    aggregate_analysis=aggregate_analysis,
                    aggregate_analysis_path=aggregate_analysis_path,
                    trials=trials,
                    project_zip_paths=project_zip_paths,
                )
                all_task_metrics.append(task_metrics)

            # Compute overview metrics
            overview = _compute_overview_metrics(all_task_metrics)

            # Build manifest
            manifest = ComparisonBenchmarkManifest(
                benchmark_timestamp=benchmark_timestamp,
                benchmark_log_path="benchmark.log" if (output_dir / "benchmark.log").exists() else None,
                agent_ids=sorted(all_agent_ids),
                overview=overview,
                final_analysis_report=final_analysis_report,
                final_analysis_report_path=final_analysis_report_path,
                tasks=all_task_metrics,
            )

            # Build summary
            num_comparison_runs = max((len(t.trials) for t in all_task_metrics), default=0)
            summary = ComparisonBenchmarkSummary(
                benchmark_timestamp=benchmark_timestamp,
                num_tasks=len(all_task_metrics),
                num_comparison_runs_per_task=num_comparison_runs,
                agent_ids=sorted(all_agent_ids),
                agent_avg_rank=overview.agent_avg_rank,
                agent_win_rate=overview.agent_win_rate,
                agent_task_wins=overview.agent_task_wins,
                mean_kendalls_w=overview.mean_kendalls_w,
            )

            # Write manifest and summary
            manifest_path = results_dir / "comparison_manifest.json"
            manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
            logger.info(f"Comparison manifest saved to: {manifest_path}")

            summary_path = results_dir / "comparison_summary.json"
            summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
            logger.info(f"Comparison summary saved to: {summary_path}")

            # Generate HTML report
            html_report_path = create_comparison_html_report(
                manifest=manifest,
                results_dir=results_dir,
            )
            logger.info(f"Comparison HTML report saved to: {html_report_path}")

            return JobResult(
                status=JobStatus.COMPLETED,
                output=ComparisonResultsAggregationJobOutput(
                    manifest_path=str(manifest_path),
                    summary_path=str(summary_path),
                    results_dir=str(results_dir),
                    html_report_path=str(html_report_path),
                ),
            )

        except Exception as e:
            error_msg = f"Failed to aggregate comparison results: {e}"
            logger.exception(error_msg)
            return JobResult(status=JobStatus.FAILED, error=error_msg)


def _calculate_kendalls_w(agent_ranks: dict[str, list[int]]) -> float | None:
    """Calculate Kendall's W (coefficient of concordance).

    Measures agreement among m comparison runs ranking n agents.
    W ranges from 0 (no agreement) to 1 (perfect agreement).

    Formula: W = 12 * S / (m^2 * (n^3 - n))
    Where S = sum of squared deviations of rank sums from mean.
    """
    if not agent_ranks:
        return None

    agents = list(agent_ranks.keys())
    n = len(agents)  # number of agents (items)
    m = len(agent_ranks[agents[0]]) if agents else 0  # number of comparisons (judges)

    if m < 2 or n < 2:
        return None  # Need at least 2 comparisons and 2 agents

    # Calculate rank sums for each agent
    rank_sums = [sum(agent_ranks[agent]) for agent in agents]

    # Calculate mean rank sum
    mean_rank_sum = sum(rank_sums) / n

    # Calculate S (sum of squared deviations)
    s = sum((r - mean_rank_sum) ** 2 for r in rank_sums)

    # Calculate W
    w = 12 * s / (m**2 * (n**3 - n))

    return w


def _compute_overview_metrics(tasks_data: list[ComparisonTaskMetrics]) -> ComparisonOverviewMetrics:
    """Compute cross-task aggregation metrics for a comparison benchmark."""
    if not tasks_data:
        return ComparisonOverviewMetrics(
            agent_avg_rank={},
            agent_win_rate={},
            agent_task_wins={},
            task_ties=0,
            mean_kendalls_w=None,
        )

    # Collect all ranks per agent across all tasks
    total_ranks: dict[str, list[int]] = defaultdict(list)
    for task_data in tasks_data:
        for agent, ranks in task_data.agent_ranks.items():
            total_ranks[agent].extend(ranks)

    # Overall Average Rank
    overall_avg_rank = {agent: sum(ranks) / len(ranks) for agent, ranks in total_ranks.items() if ranks}

    # Overall Win Rate (percentage ranked #1)
    total_wins: dict[str, int] = defaultdict(int)
    total_comparisons = 0
    for task_data in tasks_data:
        for agent, ranks in task_data.agent_ranks.items():
            total_wins[agent] += sum(1 for r in ranks if r == 1)
        # Count total comparison runs from first agent
        if task_data.agent_ranks:
            first_agent = next(iter(task_data.agent_ranks.keys()))
            total_comparisons += len(task_data.agent_ranks[first_agent])

    overall_win_rate = {
        agent: (wins / total_comparisons * 100) if total_comparisons > 0 else 0.0 for agent, wins in total_wins.items()
    }

    # Task Wins (which agent won each task based on avg rank)
    task_wins: dict[str, int] = defaultdict(int)
    task_ties = 0
    for task_data in tasks_data:
        if not task_data.agent_avg_rank:
            continue
        best_rank = min(task_data.agent_avg_rank.values())
        winners = [agent for agent, rank in task_data.agent_avg_rank.items() if rank == best_rank]
        if len(winners) == 1:
            task_wins[winners[0]] += 1
        else:
            task_ties += 1

    # Mean Kendall's W across all tasks
    w_values = [
        task_data.agreement_kendalls_w for task_data in tasks_data if task_data.agreement_kendalls_w is not None
    ]
    mean_kendalls_w = sum(w_values) / len(w_values) if w_values else None

    return ComparisonOverviewMetrics(
        agent_avg_rank=overall_avg_rank,
        agent_win_rate=overall_win_rate,
        agent_task_wins=dict(task_wins),
        task_ties=task_ties,
        mean_kendalls_w=mean_kendalls_w,
    )


# Minimum valid timestamp for ZIP files (1980-01-01 00:00:00)
_MIN_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _create_zip_with_safe_timestamps(source_dir: Path, zip_path: Path) -> None:
    """Create a zip archive, fixing file timestamps that are before 1980.

    ZIP format doesn't support timestamps before 1980. Files from Docker containers
    or with reset timestamps may have dates like 1970-01-01. This function creates
    a zip archive while ensuring all timestamps are valid.
    """
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in source_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(source_dir)
                # Create ZipInfo manually to control the timestamp
                info = zipfile.ZipInfo(str(arcname))
                # Get the file's mtime
                mtime = file_path.stat().st_mtime
                # Convert to local time tuple
                time_tuple = datetime.fromtimestamp(mtime).timetuple()[:6]
                # Use minimum valid timestamp if the file's timestamp is too old
                if time_tuple < _MIN_ZIP_TIMESTAMP:
                    info.date_time = _MIN_ZIP_TIMESTAMP
                else:
                    info.date_time = time_tuple
                # Write the file with the safe timestamp
                info.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(info, file_path.read_bytes())
