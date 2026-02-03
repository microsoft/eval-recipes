# Copyright (c) Microsoft. All rights reserved.

"""Job for aggregating all benchmark results into a structured format."""

from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import statistics
from typing import Any
import zipfile

from loguru import logger

from eval_recipes.benchmarking.job_framework.base import Job, JobContext, JobResult, JobStatus
from eval_recipes.benchmarking.jobs.score.agent_comparison_job import AgentComparisonJob
from eval_recipes.benchmarking.jobs.score.final_analysis_job import FinalAnalysisJob
from eval_recipes.benchmarking.jobs.score.trial_execution_job import TrialExecutionJob
from eval_recipes.benchmarking.reporting.create_html_report import create_html_report
from eval_recipes.benchmarking.schemas import (
    AgentDefinition,
    AgentMetrics,
    AgentSummary,
    BenchmarkManifest,
    BenchmarkSummary,
    ResultsAggregationJobInput,
    ResultsAggregationJobOutput,
    TaskDefinition,
    TaskMetrics,
    TrialExecutionJobOutput,
    TrialMetrics,
)


class ResultsAggregationJob(Job[ResultsAggregationJobOutput]):
    """Job that aggregates all benchmark results into a structured format.

    Creates:
    - results/manifest.json: Full benchmark manifest with all data
    - results/summary.json: Lightweight summary for dashboard display
    - results/benchmark.log: Copy of the benchmark log
    - results/comparison/: Comparison reports (if 2+ agents)
    - results/agents/{agent_id}/: Per-agent reports and trial data
    """

    output_model = ResultsAggregationJobOutput

    def __init__(
        self,
        job_input: ResultsAggregationJobInput,
        trial_execution_jobs: list[TrialExecutionJob],
        final_analysis_jobs: list[FinalAnalysisJob],
        agent_comparison_job: AgentComparisonJob | None,
        tasks: dict[str, TaskDefinition],
        agents: dict[str, AgentDefinition],
    ) -> None:
        self._input = job_input
        self._trial_execution_jobs = trial_execution_jobs
        self._final_analysis_jobs = final_analysis_jobs
        self._agent_comparison_job = agent_comparison_job
        self._tasks = tasks
        self._agents = agents

    @property
    def job_id(self) -> str:
        return "results_aggregation"

    @property
    def soft_dependencies(self) -> list[Job[Any]]:
        deps: list[Job[Any]] = []
        deps.extend(self._trial_execution_jobs)
        deps.extend(self._final_analysis_jobs)
        if self._agent_comparison_job:
            deps.append(self._agent_comparison_job)
        return deps

    async def run(self, context: JobContext) -> JobResult[ResultsAggregationJobOutput]:
        output_dir: Path = context.config.get("output_dir", Path.cwd() / ".benchmark_results_v2")

        logger.info(f"Starting job: {self.job_id}")

        # Create results directory
        results_dir = output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Get timestamp from benchmark log or use current time
            benchmark_timestamp = datetime.now(UTC).isoformat()

            # Copy benchmark.log to results
            benchmark_log_src = output_dir / "benchmark.log"
            benchmark_log_dest = results_dir / "benchmark.log"
            if benchmark_log_src.exists():
                shutil.copy2(benchmark_log_src, benchmark_log_dest)
                logger.info(f"Copied benchmark.log to {benchmark_log_dest}")

            # Collect agent comparison data
            comparison_exec_summary: str | None = None
            comparison_full_report: str | None = None
            comparison_exec_summary_path: str | None = None
            comparison_full_report_path: str | None = None

            if self._agent_comparison_job:
                comparison_output = context.try_get_output(self._agent_comparison_job)
                if comparison_output and comparison_output.full_report:
                    # Create comparison directory
                    comparison_dir = results_dir / "comparison"
                    comparison_dir.mkdir(parents=True, exist_ok=True)

                    # Save comparison reports
                    exec_summary_dest = comparison_dir / "EXECUTIVE_SUMMARY.md"
                    exec_summary_dest.write_text(comparison_output.executive_summary, encoding="utf-8")
                    comparison_exec_summary_path = str(exec_summary_dest.relative_to(results_dir))

                    full_report_dest = comparison_dir / "FULL_REPORT.md"
                    full_report_dest.write_text(comparison_output.full_report, encoding="utf-8")
                    comparison_full_report_path = str(full_report_dest.relative_to(results_dir))

                    comparison_exec_summary = comparison_output.executive_summary
                    comparison_full_report = comparison_output.full_report

                    logger.info("Copied agent comparison reports")

            # Build job lookup map for final analysis jobs by agent_id
            final_job_map = {job._input.agent_id: job for job in self._final_analysis_jobs}

            # Organize jobs by agent_id -> task_name -> trial_number
            agent_task_trials: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}

            for trial_job in self._trial_execution_jobs:
                agent_id = trial_job._input.agent.id
                task_name = trial_job._input.task.name
                trial_number = trial_job._input.trial_number

                if agent_id not in agent_task_trials:
                    agent_task_trials[agent_id] = {}
                if task_name not in agent_task_trials[agent_id]:
                    agent_task_trials[agent_id][task_name] = {}

                # Get combined output from TrialExecutionJob
                raw_output = context.try_get_output(trial_job)
                trial_output: TrialExecutionJobOutput | None = None
                if raw_output is not None:
                    trial_output = TrialExecutionJobOutput.model_validate(raw_output.model_dump())

                agent_task_trials[agent_id][task_name][trial_number] = {
                    "trial_output": trial_output,
                    "task": trial_job._input.task,
                }

            # Build agent metrics
            all_agent_metrics: list[AgentMetrics] = []

            for agent_id in sorted(agent_task_trials.keys()):
                agent_def = self._agents.get(agent_id)
                if not agent_def:
                    logger.warning(f"Agent definition not found for '{agent_id}'")
                    continue

                agent_dir = results_dir / "agents" / agent_id
                agent_dir.mkdir(parents=True, exist_ok=True)

                # Copy agent reports
                exec_summary_path: str | None = None
                full_report_path: str | None = None

                final_job = final_job_map.get(agent_id)
                if final_job:
                    final_output = context.try_get_output(final_job)
                    if final_output and final_output.report_generated:
                        # Save reports
                        if final_output.executive_summary:
                            exec_dest = agent_dir / "EXECUTIVE_SUMMARY.md"
                            exec_dest.write_text(final_output.executive_summary, encoding="utf-8")
                            exec_summary_path = str(exec_dest.relative_to(results_dir))

                        if final_output.full_report:
                            full_dest = agent_dir / "FULL_REPORT.md"
                            full_dest.write_text(final_output.full_report, encoding="utf-8")
                            full_report_path = str(full_dest.relative_to(results_dir))

                # Build task metrics for this agent
                agent_task_metrics_list: list[TaskMetrics] = []

                for task_name in sorted(agent_task_trials[agent_id].keys()):
                    task_trials = agent_task_trials[agent_id][task_name]
                    trial_metrics_list: list[TrialMetrics] = []

                    # Get task definition
                    task_def = self._tasks.get(task_name)
                    if not task_def:
                        logger.warning(f"Task definition not found for '{task_name}'")
                        continue

                    # Create task directory
                    task_dir = agent_dir / "tasks" / task_name / "trials"
                    task_dir.mkdir(parents=True, exist_ok=True)

                    for trial_number in sorted(task_trials.keys()):
                        trial_data = task_trials[trial_number]
                        trial_output: TrialExecutionJobOutput | None = trial_data["trial_output"]

                        # Create trial directory
                        trial_dir = task_dir / f"trial_{trial_number}"
                        trial_dir.mkdir(parents=True, exist_ok=True)

                        # Source trial directory from output
                        src_trial_dir = output_dir / agent_id / task_name / f"trial_{trial_number}"

                        # Collect logs
                        logs: dict[str, str] = {}
                        log_files = ["build_image.log", "agent_output.log", "test_output.log"]
                        for log_file in log_files:
                            src_log = src_trial_dir / log_file
                            if src_log.exists() and self._input.include_logs:
                                dest_log = trial_dir / log_file
                                shutil.copy2(src_log, dest_log)
                                logs[log_file.replace(".log", "")] = str(dest_log.relative_to(results_dir))

                        # Copy test_results.json
                        test_results_src = src_trial_dir / "test_results.json"
                        rubric: dict[str, Any] = {}
                        if test_results_src.exists():
                            dest_results = trial_dir / "test_results.json"
                            shutil.copy2(test_results_src, dest_results)
                            try:
                                rubric = json.loads(test_results_src.read_text(encoding="utf-8")).get("metadata", {})
                            except (json.JSONDecodeError, KeyError):
                                logger.warning(f"Failed to parse test results: {test_results_src}")

                        # Copy failure report if exists
                        failure_report_path: str | None = None
                        failure_report_src = src_trial_dir / "FAILURE_REPORT.md"
                        if failure_report_src.exists():
                            dest_failure = trial_dir / "FAILURE_REPORT.md"
                            shutil.copy2(failure_report_src, dest_failure)
                            failure_report_path = str(dest_failure.relative_to(results_dir))

                        # Create project zip if requested
                        project_zip_path: str | None = None
                        if self._input.include_project_zips:
                            project_src = src_trial_dir / "project"
                            if project_src.exists():
                                zip_dest = trial_dir / "project.zip"
                                _create_zip_with_safe_timestamps(project_src, zip_dest)
                                project_zip_path = str(zip_dest.relative_to(results_dir))

                        # Build trial metrics from combined TrialExecutionJobOutput
                        score = trial_output.score if trial_output else 0.0
                        agent_duration = trial_output.agent_duration_seconds if trial_output else None
                        test_duration = trial_output.test_duration_seconds if trial_output else None
                        valid_trial = trial_output.valid_trial if trial_output else True
                        failure_category = trial_output.failure_category if trial_output else None

                        trial_metrics = TrialMetrics(
                            trial_number=trial_number,
                            score=score,
                            agent_duration_seconds=agent_duration,
                            test_duration_seconds=test_duration,
                            valid_trial=valid_trial,
                            failure_category=failure_category,
                            failure_report_path=failure_report_path,
                            rubric=rubric,
                            logs=logs,
                            project_zip_path=project_zip_path,
                        )
                        trial_metrics_list.append(trial_metrics)

                    # Compute task-level aggregated metrics
                    task_metrics = self._compute_task_metrics(
                        task_name=task_name,
                        task_def=task_def,
                        trial_metrics_list=trial_metrics_list,
                    )
                    agent_task_metrics_list.append(task_metrics)

                # Compute agent-level aggregated metrics
                agent_metrics = self._compute_agent_metrics(
                    agent_id=agent_id,
                    agent_name=agent_def.agent_name,
                    task_metrics_list=agent_task_metrics_list,
                    exec_summary_path=exec_summary_path,
                    full_report_path=full_report_path,
                )
                all_agent_metrics.append(agent_metrics)

            # Build manifest
            manifest = BenchmarkManifest(
                benchmark_timestamp=benchmark_timestamp,
                benchmark_log_path="benchmark.log",
                comparison_executive_summary=comparison_exec_summary,
                comparison_full_report=comparison_full_report,
                comparison_executive_summary_path=comparison_exec_summary_path,
                comparison_full_report_path=comparison_full_report_path,
                agents=all_agent_metrics,
            )

            # Build summary
            agent_summaries = [
                AgentSummary(
                    agent_id=a.agent_id,
                    agent_name=a.agent_name,
                    mean_score=a.mean_score,
                    variability=a.variability,
                    consistency_rate=a.consistency_rate,
                )
                for a in all_agent_metrics
            ]

            summary = BenchmarkSummary(
                benchmark_timestamp=benchmark_timestamp,
                total_agents=len(all_agent_metrics),
                total_tasks=sum(a.num_unique_tasks for a in all_agent_metrics),
                total_trials=sum(a.total_trials for a in all_agent_metrics),
                agent_summaries=agent_summaries,
            )

            # Write manifest and summary
            manifest_path = results_dir / "manifest.json"
            manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
            logger.info(f"Manifest saved to: {manifest_path}")

            summary_path = results_dir / "summary.json"
            summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
            logger.info(f"Summary saved to: {summary_path}")

            # Generate HTML report
            html_report_path = create_html_report(
                manifest=manifest,
                results_dir=results_dir,
            )
            logger.info(f"HTML report saved to: {html_report_path}")

            return JobResult(
                status=JobStatus.COMPLETED,
                output=ResultsAggregationJobOutput(
                    manifest_path=str(manifest_path),
                    summary_path=str(summary_path),
                    results_dir=str(results_dir),
                    html_report_path=str(html_report_path),
                ),
            )

        except Exception as e:
            error_msg = f"Failed to aggregate results: {e}"
            logger.exception(error_msg)
            return JobResult(status=JobStatus.FAILED, error=error_msg)

    def _compute_task_metrics(
        self,
        task_name: str,
        task_def: TaskDefinition,
        trial_metrics_list: list[TrialMetrics],
    ) -> TaskMetrics:
        """Compute aggregated task metrics from trial metrics."""
        # Filter to valid trials only for score aggregation
        valid_trials = [t for t in trial_metrics_list if t.valid_trial]
        scores = [t.score for t in valid_trials]

        if not scores:
            # All trials invalid - use zeros
            mean_score = 0.0
            std_dev = 0.0
            min_score = 0.0
            max_score = 0.0
            median_score = 0.0
            num_perfect = 0
        else:
            mean_score = statistics.mean(scores)
            std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0
            min_score = min(scores)
            max_score = max(scores)
            median_score = statistics.median(scores)
            num_perfect = sum(1 for s in scores if s == 100.0)

        # Timing aggregation
        agent_durations = [t.agent_duration_seconds for t in valid_trials if t.agent_duration_seconds is not None]
        test_durations = [t.test_duration_seconds for t in valid_trials if t.test_duration_seconds is not None]

        mean_agent_duration = statistics.mean(agent_durations) if agent_durations else None
        mean_test_duration = statistics.mean(test_durations) if test_durations else None

        return TaskMetrics(
            task_name=task_name,
            instructions=task_def.instructions or "",
            task_info=task_def.task_info,
            num_trials=len(trial_metrics_list),
            num_valid_trials=len(valid_trials),
            mean_score=mean_score,
            std_dev=std_dev,
            min_score=min_score,
            max_score=max_score,
            median_score=median_score,
            num_perfect_trials=num_perfect,
            mean_agent_duration_seconds=mean_agent_duration,
            mean_test_duration_seconds=mean_test_duration,
            trials=trial_metrics_list,
        )

    def _compute_agent_metrics(
        self,
        agent_id: str,
        agent_name: str,
        task_metrics_list: list[TaskMetrics],
        exec_summary_path: str | None,
        full_report_path: str | None,
    ) -> AgentMetrics:
        """Compute aggregated agent metrics from task metrics."""
        if not task_metrics_list:
            return AgentMetrics(
                agent_id=agent_id,
                agent_name=agent_name,
                num_unique_tasks=0,
                total_trials=0,
                total_valid_trials=0,
                mean_score=0.0,
                variability=0.0,
                consistency_rate=0.0,
                mean_agent_duration_seconds=None,
                executive_summary_path=exec_summary_path,
                full_report_path=full_report_path,
                tasks=[],
            )

        # Aggregate across tasks
        task_mean_scores = [t.mean_score for t in task_metrics_list]
        task_std_devs = [t.std_dev for t in task_metrics_list]

        mean_score = statistics.mean(task_mean_scores) if task_mean_scores else 0.0
        variability = statistics.mean(task_std_devs) if task_std_devs else 0.0

        # Consistency: % of tasks with std_dev < 10
        consistent_tasks = sum(1 for t in task_metrics_list if t.std_dev < 10.0)
        consistency_rate = (consistent_tasks / len(task_metrics_list)) * 100 if task_metrics_list else 0.0

        # Total counts
        total_trials = sum(t.num_trials for t in task_metrics_list)
        total_valid = sum(t.num_valid_trials for t in task_metrics_list)

        # Timing
        durations = [t.mean_agent_duration_seconds for t in task_metrics_list if t.mean_agent_duration_seconds]
        mean_duration = statistics.mean(durations) if durations else None

        return AgentMetrics(
            agent_id=agent_id,
            agent_name=agent_name,
            num_unique_tasks=len(task_metrics_list),
            total_trials=total_trials,
            total_valid_trials=total_valid,
            mean_score=mean_score,
            variability=variability,
            consistency_rate=consistency_rate,
            mean_agent_duration_seconds=mean_duration,
            executive_summary_path=exec_summary_path,
            full_report_path=full_report_path,
            tasks=task_metrics_list,
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
