# Copyright (c) Microsoft. All rights reserved.

"""Job for aggregating all per-task comparison reports into a final analysis."""

from pathlib import Path
from typing import Any

from liquid import render
from loguru import logger

from eval_recipes.benchmarking.job_framework.base import Job, JobContext, JobResult, JobStatus
from eval_recipes.benchmarking.jobs.comparison.comparison_aggregation_job import ComparisonAggregationJob
from eval_recipes.benchmarking.schemas import ComparisonFinalAnalysisJobInput, ComparisonFinalAnalysisJobOutput
from eval_recipes.utils.llm import create_client, truncate_reports_to_token_limit

COMPARISON_FINAL_ANALYSIS_SYSTEM_PROMPT = """\
You are an expert at synthesizing AI agent comparison results across multiple tasks.
Your goal is to provide a comprehensive summary of how agents performed relative to each other
across all evaluated tasks.

Be factual and grounded in the provided reports. Do not speculate beyond what is shown."""

COMPARISON_FINAL_ANALYSIS_USER_PROMPT = """\
## Per-Task Comparison Summaries

{{task_reports}}

## Your Analysis Task
Synthesize the above per-task summaries into a final report that answers:
- Which agent(s) performed best overall and why?
- Were there tasks where rankings differed significantly?
- What are the key differentiating factors between agents?

Provide a well-organized markdown report with clear sections."""


class ComparisonFinalAnalysisJob(Job[ComparisonFinalAnalysisJobOutput]):
    """Job that aggregates all per-task comparison reports into a final analysis.

    This is the Level 2 aggregation that synthesizes insights across all tasks.
    """

    output_model = ComparisonFinalAnalysisJobOutput

    def __init__(
        self,
        job_input: ComparisonFinalAnalysisJobInput,
        comparison_aggregation_jobs: list[ComparisonAggregationJob],
    ) -> None:
        self._input = job_input
        self._comparison_aggregation_jobs = comparison_aggregation_jobs

    @property
    def job_id(self) -> str:
        return "comparison_final_analysis"

    @property
    def soft_dependencies(self) -> list[Job[Any]]:
        return list(self._comparison_aggregation_jobs)

    async def run(self, context: JobContext) -> JobResult[ComparisonFinalAnalysisJobOutput]:
        output_dir: Path = context.config.get("output_dir", Path.cwd() / ".benchmark_results_v2")

        logger.info(f"Starting job: {self.job_id}")

        # Collect outputs from ComparisonAggregationJobs (skip failed)
        task_reports: list[tuple[str, str]] = []  # (task_name, analysis_report)
        for job in self._comparison_aggregation_jobs:
            output = context.try_get_output(job)
            if output is not None and output.num_comparisons_analyzed > 0:
                task_reports.append((output.task_name, output.analysis_report))

        # Skip if no task reports available
        if not task_reports:
            logger.info("No per-task comparison reports available, skipping final analysis")
            report_path = output_dir / "COMPARISON_FINAL_REPORT.md"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            empty_report = "No per-task comparison reports available for final analysis."
            report_path.write_text(empty_report, encoding="utf-8")

            return JobResult(
                status=JobStatus.COMPLETED,
                output=ComparisonFinalAnalysisJobOutput(
                    analysis_report=empty_report,
                    report_path=str(report_path),
                    num_tasks_analyzed=0,
                ),
            )

        logger.info(f"Synthesizing {len(task_reports)} per-task comparison reports")

        # Format reports with task name headers
        formatted_reports = []
        for task_name, report in task_reports:
            formatted_reports.append(f"## Task: {task_name}\n\n{report}\n\n{'=' * 80}\n")

        # Truncate if needed
        truncated_reports = truncate_reports_to_token_limit(formatted_reports)
        all_task_reports = "\n".join(truncated_reports)

        # Render user prompt
        user_prompt = render(COMPARISON_FINAL_ANALYSIS_USER_PROMPT, task_reports=all_task_reports)

        messages: list = [
            {"role": "system", "content": COMPARISON_FINAL_ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            async with create_client(provider=self._input.provider) as client:
                response = await client.responses.create(
                    model=self._input.model,
                    input=messages,
                    store=False,
                )

            analysis = response.output_text or "Failed to generate final analysis."

            # Save the report
            report_path = output_dir / "COMPARISON_FINAL_REPORT.md"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(analysis, encoding="utf-8")
            logger.info(f"Comparison final analysis report saved to: {report_path}")

            return JobResult(
                status=JobStatus.COMPLETED,
                output=ComparisonFinalAnalysisJobOutput(
                    analysis_report=analysis,
                    report_path=str(report_path),
                    num_tasks_analyzed=len(task_reports),
                ),
            )

        except Exception as e:
            error_msg = f"Failed to generate comparison final analysis report: {e}"
            logger.exception(error_msg)
            return JobResult(status=JobStatus.FAILED, error=error_msg)
