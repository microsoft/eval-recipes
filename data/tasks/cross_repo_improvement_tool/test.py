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


# Define Semantic Test 1: Check for repomix usage

STEPS_1_REPOMIX_USAGE = """1. Explore the code that the agent generated in this project directory to see if it uses repomix \
to analyze repositories (via `npx repomix@latest` or similar).
2. Look for where the tool invokes repomix commands to "roll up" source and target repositories.
3. Check if the tool properly handles both local and remote repositories as specified in the requirements.
4. Verify that the tool captures and uses the repomix output for analysis."""

RUBRIC_1_REPOMIX_USAGE = {
    "uses_repomix": "str - explanation of whether the agent's solution uses repomix (npx repomix@latest or similar)",
    "handles_source_and_target": "str - does the tool support analyzing both a source repo and a target repo",
    "handles_local_and_remote": "str - does the tool handle both local and remote repositories",
    "score": "float - Score 100 if the tool correctly uses repomix for both source and target repos. \
Score 75 if it uses repomix but with limitations. Score 50 if it partially uses repomix. Score 0 if it doesn't use repomix at all.",
}


# Define Semantic Test 2: Check for feedback loop architecture

STEPS_2_FEEDBACK_LOOPS = """1. Examine the code structure to identify if there are internal feedback loops as specified.
2. Look for multiple reviewer components (e.g., grounding reviewer, philosophy/patterns reviewer, and any additional reviewers).
3. Check if the tool has a mechanism where:
   - Content is drafted
   - Passed through isolated review processes
   - Feedback is collected and passed back to analysis stages
   - The loop continues until reviews pass
4. Verify that there is a human review step that allows for comments and triggers re-analysis."""

RUBRIC_2_FEEDBACK_LOOPS = {
    "has_internal_loops": "str - (30 points) does the tool implement internal feedback loops where analysis -> draft -> review -> back to analysis",
    "multiple_reviewers": "str - (25 points) are there multiple distinct reviewers (grounding, philosophy/patterns, etc.)",
    "isolated_review_processes": "str - (20 points) are reviews performed as isolated processes as specified",
    "human_review_integration": "str - (25 points) is there a human review step that allows comments and triggers re-analysis",
    "score": "float - Score between 0 and 100 based on the above criteria. \
Each criterion contributes points as indicated. Sum the points earned from each criterion.",
}


# Define Semantic Test 3: Check output quality and structure

STEPS_3_OUTPUT_STRUCTURE = """1. Look for where the tool generates its output files and directories.
2. Check if the output includes:
   - A comprehensive analysis covering high-level to detailed breakdown of opportunities
   - For each opportunity, separate detailed proposals with full context and implementation guidance
   - Output directory and files named based on the context of the input/ask
3. Verify that proposals are detailed enough to hand off to a team without full context or access to source repo.
4. Check if the tool produces structured, organized output that separates the analysis from individual proposals."""

RUBRIC_3_OUTPUT_STRUCTURE = {
    "comprehensive_analysis": "str - (25 points) does the output include a comprehensive analysis with opportunities and their value/rationale",
    "separate_proposals": "str - (25 points) are individual opportunities broken down into separate detailed proposals",
    "implementation_guidance": "str - (25 points) do proposals include full context and implementation guidance for handoff",
    "contextual_naming": "str - (15 points) are output directories and files named based on the input context",
    "organized_structure": "str - (10 points) is the output well-organized and structured",
    "score": "float - Score between 0 and 100 based on the above criteria. \
Each criterion contributes points as indicated. Sum the points earned from each criterion.",
}


# Define Semantic Test 4: Check for specific reviewer types

STEPS_4_REVIEWER_TYPES = """1. Examine the codebase to identify the different types of reviewers implemented.
2. Check specifically for:
   - A grounding reviewer that ensures analysis/drafts are grounded to source and target repo roll-ups and the input ask
   - A philosophy/patterns reviewer that analyzes how well drafts fit within the philosophies and patterns of the target repo
   - Additional reviewers beyond these two
3. Verify that each reviewer has a distinct purpose and performs isolated reviews."""

RUBRIC_4_REVIEWER_TYPES = {
    "grounding_reviewer": "str - (40 points) is there a grounding reviewer that ensures content is grounded to repos and ask",
    "philosophy_reviewer": "str - (40 points) is there a philosophy/patterns reviewer for ensuring adaptation to target patterns",
    "additional_reviewers": "str - (20 points) are there other thoughtful reviewers beyond the two core ones",
    "score": "float - Score between 0 and 100 based on the above criteria. \
Each criterion contributes points as indicated. Sum the points earned from each criterion.",
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
    """Test script for cross_repo_improvement_tool task."""
    return asyncio.run(run_test(test_id, output_dir, instructions_file))


async def run_test(test_id: str, output_dir: Path, instructions_file: Path | None) -> int:
    instructions = get_instructions_from_file_or_default(instructions_file=instructions_file)

    try:
        logger.info("Running semantic test 1: Checking for repomix usage...")
        result_1 = await semantic_test(
            steps=STEPS_1_REPOMIX_USAGE,
            rubric=RUBRIC_1_REPOMIX_USAGE,
            context=instructions,
            working_dir=Path("/project"),
        )

        logger.info("Running semantic test 2: Checking for feedback loop architecture...")
        result_2 = await semantic_test(
            steps=STEPS_2_FEEDBACK_LOOPS,
            rubric=RUBRIC_2_FEEDBACK_LOOPS,
            context=instructions,
            working_dir=Path("/project"),
        )

        logger.info("Running semantic test 3: Checking output quality and structure...")
        result_3 = await semantic_test(
            steps=STEPS_3_OUTPUT_STRUCTURE,
            rubric=RUBRIC_3_OUTPUT_STRUCTURE,
            context=instructions,
            working_dir=Path("/project"),
        )

        logger.info("Running semantic test 4: Checking for specific reviewer types...")
        result_4 = await semantic_test(
            steps=STEPS_4_REVIEWER_TYPES,
            rubric=RUBRIC_4_REVIEWER_TYPES,
            context=instructions,
            working_dir=Path("/project"),
        )

        # Calculate final score as weighted average
        # Weights: repomix (20%), feedback loops (30%), output structure (25%), reviewers (25%)
        final_score = (
            result_1.score * 0.20 + result_2.score * 0.30 + result_3.score * 0.25 + result_4.score * 0.25
        )

        metadata = {
            "instructions": instructions,
            "semantic_test_1_repomix_usage": {
                "score": result_1.score,
                "details": result_1.metadata,
            },
            "semantic_test_2_feedback_loops": {
                "score": result_2.score,
                "details": result_2.metadata,
            },
            "semantic_test_3_output_structure": {
                "score": result_3.score,
                "details": result_3.metadata,
            },
            "semantic_test_4_reviewer_types": {
                "score": result_4.score,
                "details": result_4.metadata,
            },
            "final_score": final_score,
            "scoring_weights": {
                "repomix_usage": "20%",
                "feedback_loops": "30%",
                "output_structure": "25%",
                "reviewer_types": "25%",
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
