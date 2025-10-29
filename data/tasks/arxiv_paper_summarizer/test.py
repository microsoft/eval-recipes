# Copyright (c) Microsoft. All rights reserved.

import asyncio
import sys
from pathlib import Path

import click
from loguru import logger

from eval_recipes.benchmarking.semantic_test import semantic_test
from eval_recipes.benchmarking.test_utils import (
    get_instructions_from_file_or_default,
    get_test_id_from_env_or_default,
    write_test_result,
)


# Define Semantic Test 1: Implementation Verification

STEPS_1_IMPLEMENTATION_VERIFICATION = """1. Explore the code in the project directory to understand the implementation.
2. Verify the following dependencies and integrations:
   - Check if an Anthropic LLM is being used for summarization
   - **PDF Extraction Libraries**: Look for at least 3 different PDF extraction libraries (pdfminer.six, pdfplumber, and one more)
     * The tool should have a CLI flag to enable multi-library extraction mode
     * The tool should intelligently select the best extraction result when this flag is used
   - **CRITICAL**: Verify integration with eval-recipes claim verification LOW-LEVEL API
     * The tool should import and use functions directly from eval_recipes.evaluations.claim_verification
     * Look for imports like: from eval_recipes.evaluations.claim_verification import ...
     * When constructing input data for the claim verification low level API, it should pass in the paper as source_context, the generated summary as the text, and "user_question" should be the user prompt sent to the LLM or something simple like "Summarize this paper"
     * Should NOT just call a high-level evaluate() function
   - **Parallelism Configuration**: Check if claim verification is configured with high parallelism (5+ threads/workers)
     * `ClaimVerificationEvaluator` is initialized with a `config: ClaimVerificationEvaluatorConfig`. `ClaimVerificationEvaluatorConfig.max_concurrency` should be set to at least 5
   - For context, this is the repo: https://github.com/microsoft/eval-recipes
3. Check for revision logic:
   - Code should run claim verification on the summary
   - If claims are unverified, code should revise the summary to remove them. This requires custom logic that sets up another prompt that takes in the generated summary, the claim verification results (which should at least be the list of OutputClaimVerificationEvaluator outputs), and then uses those to instruct the LLM to produce a revised summary that addresses unverified claims
4. Verify output file generation logic for both summary and verification results.
5. Check summary generation guidelines:
   - There should exist a prompt for the LLM that instructs it to aim for 1-2 paragraph summaries (abstract-sized)
   - Summary should be written for someone who casually follows the field, not an expert. However, it should emphasize the need to be factual and accurate to avoid fabrication."""

RUBRIC_1_IMPLEMENTATION_VERIFICATION = {
    "anthropic_model_usage": "str - (15 points) Does the code use an Anthropic LLM for summarization?",
    "multi_library_pdf_extraction": "str - (25 points) Does the code support at least 3 PDF extraction libraries with intelligent selection via a CLI flag?",
    "claim_verification_low_level_api": "str - (25 points) Does the code use eval-recipes claim verification LOW-LEVEL API (importing directly from `eval_recipes.evaluations.claim_verification.ClaimVerificationEvaluator.run()`)?",
    "parallelism_configured": "str - (10 points) Is claim verification configured with high parallelism (5+ threads/workers)?",
    "revision_logic": "str - (15 points) Is there logic to revise the summary based on claim verification results if claims are unverified?",
    "output_file_generation": "str - (5 points) Does the code generate both summary and verification output files?",
    "summary_guidelines": "str - (5 points) Does the code follow guidelines for summary length (1-2 paragraphs) and target audience?",
    "score": "float - Score between 0 and 100 based on the above criteria. Sum the points earned from each criterion.",
}


# Define Semantic Test 2: Functional Run + Output Validation

STEPS_2_FUNCTIONAL_RUN = """1. Find the README in the project directory that explains how to use the tool.
2. Based on the README, determine the correct command to run the tool.
   - Look for the CLI flag that enables multi-library PDF extraction mode
   - If such a flag exists, include it in your test command
3. Run the tool with this test arxiv link: https://arxiv.org/pdf/2411.10541
   - The tool can take up to 15 minutes to complete (PDF download with multiple libraries, summarization, claim verification)
   - If the tool takes longer than 15 minutes or fails with errors, stop and note this as a failure
4. After the tool completes (or fails), check for output files:
   - Look for a summary file
   - Look for a verification results file
5. If both files exist, examine their contents:
   - Summary file should contain a coherent 1-2 paragraph summary (abstract-sized)
   - Summary should be accessible to someone who casually follows the field
   - Verification file should show results from claim verification with clear formatting
6. Verify the README exists and has clear usage instructions including the multi-library extraction flag."""

RUBRIC_2_FUNCTIONAL_RUN = {
    "readme_exists": "str - (10 points) Does a README exist with clear usage instructions including multi-library extraction flag?",
    "tool_runs_successfully": "str - (30 points) Does the tool run without critical errors when given the arxiv link?",
    "completes_within_time": "str - (15 points) Does the tool complete within 15 minutes?",
    "summary_file_created": "str - (15 points) Is a summary markdown file created with the paper title?",
    "verification_file_created": "str - (15 points) Is a verification results markdown file created?",
    "summary_appropriate_length": "str - (15 points) Is the summary 1-2 paragraphs (abstract-sized) and accessible to casual readers?",
    "score": "float - Score between 0 and 100 based on the above criteria. Sum the points earned from each criterion.",
}


@click.command()
@click.option(
    "--test-id",
    default=lambda: get_test_id_from_env_or_default("dev"),
    help="Test ID for result file naming (defaults to EVAL_RECIPES_TEST_ID env var)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=lambda: Path(__file__).parents[0],
    help="Directory to write result file",
)
@click.option(
    "--instructions-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to instructions file (defaults to ./instructions.txt in working directory)",
)
def main(test_id: str, output_dir: Path, instructions_file: Path | None) -> int:
    """Test script for arxiv_paper_summarizer task."""
    return asyncio.run(run_test(test_id, output_dir, instructions_file))


async def run_test(test_id: str, output_dir: Path, instructions_file: Path | None) -> int:
    instructions = get_instructions_from_file_or_default(instructions_file=instructions_file)

    try:
        logger.info("Running semantic test 1: Verifying implementation and dependencies...")
        result_1 = await semantic_test(
            steps=STEPS_1_IMPLEMENTATION_VERIFICATION,
            rubric=RUBRIC_1_IMPLEMENTATION_VERIFICATION,
            context=instructions,
            working_dir=Path("/project"),
        )

        logger.info("Running semantic test 2: Running tool and validating outputs...")
        result_2 = await semantic_test(
            steps=STEPS_2_FUNCTIONAL_RUN,
            rubric=RUBRIC_2_FUNCTIONAL_RUN,
            context=instructions,
            working_dir=Path("/project"),
        )

        # Calculate final score with weighted average
        final_score = result_1.score * 0.40 + result_2.score * 0.60

        metadata = {
            "instructions": instructions,
            "semantic_test_1_implementation": {
                "score": result_1.score,
                "details": result_1.metadata,
            },
            "semantic_test_2_functional_run": {
                "score": result_2.score,
                "details": result_2.metadata,
            },
            "final_score": final_score,
            "scoring_weights": {
                "implementation_verification": "40%",
                "functional_run_and_outputs": "60%",
            },
        }

        write_test_result(output_dir, test_id, final_score, metadata)
        logger.info(f"Test completed with final score: {final_score:.1f}/100")
        return 0

    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        metadata = {
            "instructions": instructions,
            "error": str(e),
        }
        write_test_result(output_dir, test_id, 0, metadata)
        return 0


if __name__ == "__main__":
    sys.exit(main())


"""
Sample command to test this task locally:
export ANTHROPIC_API_KEY=your_key
export OPENAI_API_KEY=your_key
uv run scripts/run_benchmarks.py --task-filter name=arxiv_paper_summarizer --agent-filter name=claude_code
"""
