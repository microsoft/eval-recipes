# Copyright (c) Microsoft. All rights reserved.

"""Job for aggregating comparison results across multiple runs for a single task."""

from pathlib import Path
from typing import Any

from liquid import render
from loguru import logger

from eval_recipes.benchmarking.job_framework.base import Job, JobContext, JobResult, JobStatus
from eval_recipes.benchmarking.jobs.comparison.semantic_comparison_job import SemanticComparisonJob
from eval_recipes.benchmarking.schemas import ComparisonAggregationJobInput, ComparisonAggregationJobOutput
from eval_recipes.utils.llm import create_client, truncate_reports_to_token_limit

COMPARISON_AGGREGATION_SYSTEM_PROMPT = """\
You are an expert evaluator analyzing AI agent comparison results.
Your task is to synthesize multiple comparison judgments into a clear summary that
explains WHY agents were ranked the way they were.

Be factual and accurate. Only report patterns clearly supported by the comparison
reasoning provided. Do not speculate or add information not present in the comparisons."""

COMPARISON_AGGREGATION_USER_PROMPT = """\
## Original Task
{{task_instructions}}

## Comparison Results ({{comparison_count}} runs)
{{comparison_results}}

## Your Analysis Task
Based on the {{comparison_count}} comparison runs above, provide a concise summary \
explaining WHY the rankings were what they were.

Output a bulleted list of 5-10 items, each about one sentence. Focus on:
- Why one agent consistently won (if applicable)
- Key factors that separated the agents
- Areas of agreement or disagreement between comparison runs

Be specific and cite evidence from the reasoning provided. Do not fabricate details."""


class ComparisonAggregationJob(Job[ComparisonAggregationJobOutput]):
    """Job that aggregates multiple comparison runs into a single summary report for a task.

    Uses a single OpenAI API call to generate a markdown report.
    """

    output_model = ComparisonAggregationJobOutput

    def __init__(
        self,
        job_input: ComparisonAggregationJobInput,
        semantic_comparison_jobs: list[SemanticComparisonJob],
    ) -> None:
        self._input = job_input
        self._semantic_comparison_jobs = semantic_comparison_jobs

    @property
    def job_id(self) -> str:
        return f"comparison_aggregation:{self._input.task_name}"

    @property
    def soft_dependencies(self) -> list[Job[Any]]:
        return list(self._semantic_comparison_jobs)

    async def run(self, context: JobContext) -> JobResult[ComparisonAggregationJobOutput]:
        output_dir: Path = context.config.get("output_dir", Path.cwd() / ".benchmark_results_v2")

        logger.info(f"Starting job: {self.job_id}")

        # Collect comparison outputs (skip failed jobs)
        comparison_outputs = []
        for job in self._semantic_comparison_jobs:
            output = context.try_get_output(job)
            if output is not None:
                comparison_outputs.append(output)

        # Skip if no comparison results
        if not comparison_outputs:
            logger.info(f"No comparison results for task '{self._input.task_name}', skipping aggregation")
            report_path = output_dir / "comparisons" / self._input.task_name / "AGGREGATE_REPORT.md"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            empty_report = "No comparison results available for aggregation."
            report_path.write_text(empty_report, encoding="utf-8")

            return JobResult(
                status=JobStatus.COMPLETED,
                output=ComparisonAggregationJobOutput(
                    task_name=self._input.task_name,
                    analysis_report=empty_report,
                    report_path=str(report_path),
                    num_comparisons_analyzed=0,
                ),
            )

        logger.info(f"Aggregating {len(comparison_outputs)} comparison runs for task '{self._input.task_name}'")

        # Format each comparison run
        formatted_runs = []
        for output in comparison_outputs:
            # Sort rankings by rank (1=best first)
            sorted_rankings = sorted(output.rankings.items(), key=lambda x: x[1])
            ranked_agents = [f"{rank}. {agent_id}" for agent_id, rank in sorted_rankings]

            formatted = f"""### Comparison Run {output.comparison_run_number}
**Rankings (best to worst):**
{chr(10).join(ranked_agents)}

**Reasoning:**
{output.reasoning}

---"""
            formatted_runs.append(formatted)

        # Truncate if needed
        truncated_runs = truncate_reports_to_token_limit(formatted_runs)
        comparison_results = "\n".join(truncated_runs)

        # Render user prompt
        user_prompt = render(
            COMPARISON_AGGREGATION_USER_PROMPT,
            task_instructions=self._input.task_instructions,
            comparison_count=len(comparison_outputs),
            comparison_results=comparison_results,
        )

        messages: list = [
            {"role": "system", "content": COMPARISON_AGGREGATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            async with create_client(provider=self._input.provider) as client:
                response = await client.responses.create(
                    model=self._input.model,
                    input=messages,
                    store=False,
                )

            analysis = response.output_text or "Failed to generate analysis."

            # Save the report
            report_path = output_dir / "comparisons" / self._input.task_name / "AGGREGATE_REPORT.md"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(analysis, encoding="utf-8")
            logger.info(f"Comparison aggregation report saved to: {report_path}")

            return JobResult(
                status=JobStatus.COMPLETED,
                output=ComparisonAggregationJobOutput(
                    task_name=self._input.task_name,
                    analysis_report=analysis,
                    report_path=str(report_path),
                    num_comparisons_analyzed=len(comparison_outputs),
                ),
            )

        except Exception as e:
            error_msg = f"Failed to generate comparison aggregation report: {e}"
            logger.exception(error_msg)
            return JobResult(status=JobStatus.FAILED, error=error_msg)
