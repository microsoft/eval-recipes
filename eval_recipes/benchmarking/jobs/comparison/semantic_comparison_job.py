# Copyright (c) Microsoft. All rights reserved.

"""Job for performing blind semantic comparisons of multiple agents' outputs on the same task."""

from pathlib import Path
from typing import Any

from loguru import logger

from eval_recipes.benchmarking.evaluation.semantic_test_comparison import semantic_test_comparison
from eval_recipes.benchmarking.job_framework.base import Job, JobContext, JobResult, JobStatus
from eval_recipes.benchmarking.jobs.comparison.comparison_trial_job import ComparisonTrialJob
from eval_recipes.benchmarking.schemas import SemanticComparisonJobInput, SemanticComparisonJobOutput


class SemanticComparisonJob(Job[SemanticComparisonJobOutput]):
    """Job that performs blind semantic comparison of multiple agents' outputs on the same task."""

    output_model = SemanticComparisonJobOutput

    def __init__(
        self,
        job_input: SemanticComparisonJobInput,
        comparison_trial_jobs: list[ComparisonTrialJob],
    ) -> None:
        """
        Initialize the semantic comparison job.

        Args:
            job_input: Configuration for the comparison
            comparison_trial_jobs: List of ComparisonTrialJob instances (one per agent)
        """
        self._input = job_input
        self._comparison_trial_jobs = comparison_trial_jobs

        # Build agent_id list in the same order as comparison_trial_jobs for later mapping
        self._agent_ids = [job._input.agent.id for job in comparison_trial_jobs]

    @property
    def job_id(self) -> str:
        agent_ids_str = "_".join(sorted(self._agent_ids))
        return f"semantic_comparison:{self._input.task_name}:{agent_ids_str}:run{self._input.comparison_run_number}"

    @property
    def dependencies(self) -> list[Job[Any]]:
        return list(self._comparison_trial_jobs)

    async def run(self, context: JobContext) -> JobResult[SemanticComparisonJobOutput]:
        output_dir: Path = context.config.get("output_dir", Path.cwd() / ".benchmark_results_v2")

        logger.info(f"Starting job: {self.job_id}")

        # Collect project directories from comparison trial job outputs
        # Maintain order to preserve index mapping
        directories: list[Path] = []
        index_to_agent_id: dict[int, str] = {}

        for i, trial_job in enumerate(self._comparison_trial_jobs):
            trial_output = context.get_output(trial_job)
            directories.append(Path(trial_output.project_dir))
            index_to_agent_id[i] = trial_output.agent_id

        # Create output directory for comparison results
        comparison_output_dir = output_dir / "comparisons" / self._input.task_name
        comparison_output_dir.mkdir(parents=True, exist_ok=True)

        log_file = comparison_output_dir / f"comparison_{self._input.comparison_run_number}.log"

        try:
            # Run the semantic comparison
            comparison_result = await semantic_test_comparison(
                original_task=self._input.task_instructions,
                directories=directories,
                guidelines=self._input.guidelines,
                log_file=log_file,
            )

            # Convert index-based rankings to agent_id-based rankings
            # rankings is a list of indices ordered best to worst
            rankings: dict[str, int] = {}
            for rank, index in enumerate(comparison_result.rankings, start=1):
                agent_id = index_to_agent_id[index]
                rankings[agent_id] = rank

            # Build anonymous_to_agent_id mapping
            anonymous_to_agent_id: dict[str, str] = {}
            for anon_name, index in comparison_result.anonymous_to_index.items():
                anonymous_to_agent_id[anon_name] = index_to_agent_id[index]

            output = SemanticComparisonJobOutput(
                task_name=self._input.task_name,
                comparison_run_number=self._input.comparison_run_number,
                reasoning=comparison_result.reasoning,
                rankings=rankings,
                anonymous_to_agent_id=anonymous_to_agent_id,
            )

            # Save result JSON
            result_file = comparison_output_dir / f"result_{self._input.comparison_run_number}.json"
            result_file.write_text(output.model_dump_json(indent=2), encoding="utf-8")

            logger.info(f"Semantic comparison completed: {self.job_id}")
            logger.info(f"Rankings: {rankings}")

            return JobResult(status=JobStatus.COMPLETED, output=output)

        except Exception as e:
            error_msg = f"Semantic comparison failed: {e}"
            logger.exception(error_msg)

            # Write error to a file for debugging
            error_file = comparison_output_dir / f"error_{self._input.comparison_run_number}.txt"
            error_file.write_text(error_msg, encoding="utf-8")

            return JobResult(status=JobStatus.FAILED, error=error_msg)
