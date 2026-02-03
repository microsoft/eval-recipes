# Copyright (c) Microsoft. All rights reserved.

"""Job for comparing failure reports across multiple agents."""

from pathlib import Path
from typing import Any

from liquid import render
from loguru import logger
from openai.types.shared_params.reasoning import Reasoning
from pydantic import BaseModel, Field

from eval_recipes.benchmarking.job_framework.base import Job, JobContext, JobResult, JobStatus
from eval_recipes.benchmarking.jobs.score.final_analysis_job import FinalAnalysisJob
from eval_recipes.benchmarking.schemas import AgentComparisonJobInput, AgentComparisonJobOutput
from eval_recipes.utils.llm import create_client, truncate_reports_to_token_limit

AGENT_COMPARISON_SYSTEM_PROMPT = """\
You are an expert at comparing AI agent performance based on benchmark failure analysis reports.
Your task is to objectively and factually analyze why certain agents performed better or worse
than others based on the evidence in the provided failure reports.
Your goal is to answer the question of why each agent scored the way it did, relative to the others.

ABOUT THESE REPORTS:
Each report contains a consolidated analysis of an agent's failures across multiple benchmark tasks.

YOUR GOAL:
Compare the agents' performance and explain:
- Which agents performed better/worse and why
- What distinguished the agent's.
- Common patterns vs unique issues per agent

REQUIREMENTS:
- Be factual and grounded in the reports - cite specific examples
- Write 2-5 well-organized paragraphs
- Do NOT speculate beyond what the reports show
- Do NOT make recommendations for improvement"""

AGENT_COMPARISON_USER_PROMPT = """\
Compare the following agent failure reports and analyze why certain agents performed
better or worse than others.

You will provide two outputs:

1. **full_report**: A detailed markdown comparison with:
   - Which agents performed better/worse and why
   - What distinguished the agents
   - Common patterns vs unique issues per agent
   - Specific examples and citations from the reports

2. **executive_summary**: A 1-2 paragraph summary with:
   - Overall comparison assessment
   - Key differentiators between agents

---

# Agent Failure Reports

{{all_reports}}"""


class ComparisonReportResponse(BaseModel):
    """Structured output for the comparison report."""

    full_report: str = Field(
        description="Detailed markdown comparison with analysis of agent performance differences and specific examples."
    )
    executive_summary: str = Field(description="1-2 paragraph summary of the comparison with key differentiators.")


class AgentComparisonJob(Job[AgentComparisonJobOutput]):
    """Job that compares failure reports across multiple agents.

    Uses a single OpenAI API call to generate a comparison report.
    """

    output_model = AgentComparisonJobOutput

    def __init__(
        self,
        job_input: AgentComparisonJobInput,
        final_analysis_jobs: list[FinalAnalysisJob],
    ) -> None:
        self._input = job_input
        self._final_analysis_jobs = final_analysis_jobs

    @property
    def job_id(self) -> str:
        return "agent_comparison"

    @property
    def soft_dependencies(self) -> list[Job[Any]]:
        return list(self._final_analysis_jobs)

    async def run(self, context: JobContext) -> JobResult[AgentComparisonJobOutput]:
        output_dir: Path = context.config.get("output_dir", Path.cwd() / ".benchmark_results_v2")

        logger.info(f"Starting job: {self.job_id}")

        # Collect reports from dependencies (soft deps may have failed)
        agent_reports: list[tuple[str, str]] = []  # (agent_id, report_content)
        for job in self._final_analysis_jobs:
            output = context.try_get_output(job)
            if output is None:
                # Job didn't complete successfully, skip it
                continue
            if output.report_generated and output.full_report:
                agent_id = job._input.agent_id
                # Combine executive summary and full report for comparison context
                report_content = f"## Executive Summary\n\n{output.executive_summary}\n\n{output.full_report}"
                agent_reports.append((agent_id, report_content))

        # Skip if fewer than 2 agents have reports
        if len(agent_reports) < 2:
            logger.info(f"Fewer than 2 agents with reports ({len(agent_reports)}), skipping comparison")
            return JobResult(
                status=JobStatus.COMPLETED,
                output=AgentComparisonJobOutput(
                    executive_summary="",
                    full_report="",
                    executive_summary_path="",
                    full_report_path="",
                    num_agents_compared=len(agent_reports),
                ),
            )

        logger.info(f"Comparing {len(agent_reports)} agent reports")

        # Format reports with headers
        formatted_reports = []
        for agent_id, report in agent_reports:
            formatted_reports.append(f"## Agent: {agent_id}\n\n{report}\n\n{'=' * 80}\n")

        # Truncate if needed
        truncated_reports = truncate_reports_to_token_limit(formatted_reports)
        all_reports = "\n".join(truncated_reports)

        # Render user prompt
        user_prompt = render(AGENT_COMPARISON_USER_PROMPT, all_reports=all_reports)

        messages: list = [
            {"role": "system", "content": AGENT_COMPARISON_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            async with create_client(provider=self._input.provider) as client:
                response = await client.responses.parse(
                    model=self._input.model,
                    input=messages,
                    text_format=ComparisonReportResponse,
                    reasoning=Reasoning(effort="medium"),
                    store=False,
                )

            if not response.output_parsed:
                error_msg = "Failed to parse comparison report response"
                logger.error(error_msg)
                return JobResult(status=JobStatus.FAILED, error=error_msg)

            parsed_response: ComparisonReportResponse = response.output_parsed

            # Save executive summary and full report separately
            executive_summary_path = output_dir / "COMPARISON_SUMMARY.md"
            executive_summary_path.write_text(parsed_response.executive_summary, encoding="utf-8")
            logger.info(f"Comparison executive summary saved to: {executive_summary_path}")

            full_report_path = output_dir / "COMPARISON_FULL.md"
            full_report_path.write_text(parsed_response.full_report, encoding="utf-8")
            logger.info(f"Comparison full report saved to: {full_report_path}")

            return JobResult(
                status=JobStatus.COMPLETED,
                output=AgentComparisonJobOutput(
                    executive_summary=parsed_response.executive_summary,
                    full_report=parsed_response.full_report,
                    executive_summary_path=str(executive_summary_path),
                    full_report_path=str(full_report_path),
                    num_agents_compared=len(agent_reports),
                ),
            )

        except Exception as e:
            error_msg = f"Failed to generate agent comparison report: {e}"
            logger.exception(error_msg)
            return JobResult(status=JobStatus.FAILED, error=error_msg)
