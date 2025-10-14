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


# Test configuration - repos to analyze
SOURCE_REPO = "github/spec-kit"
TARGET_REPO = "github/copilot-cli"
IMPROVEMENT_REQUEST = """Analyze how spec-kit's structured 4-phase workflow (Specify → Plan → Tasks → Implementation), \
validation checkpoints, and template system could improve copilot-cli's handling of complex \
coding tasks. Focus on how copilot-cli could adopt more structured planning capabilities \
while maintaining its conversational terminal-native experience."""


# Define Semantic Test 1: Architecture & Implementation Review

STEPS_1_ARCHITECTURE_REVIEW = f"""1. Explore the code that the agent generated in this project directory to understand the tool's architecture.
2. Check if the tool uses repomix correctly:
   - Look for where it invokes repomix (e.g., `npx repomix@latest` or similar)
   - Verify it can handle remote GitHub repositories (not just local repos)
   - Check that it processes both a source repo and a target repo
3. Examine the feedback loop architecture:
   - Look for evidence of multiple distinct reviewers (e.g., grounding reviewer, philosophy/patterns reviewer, additional reviewers)
   - Check for analysis → draft → review → re-analysis loop patterns
   - Verify reviewers are implemented as isolated processes
   - Look for human review integration that allows comments and triggers re-analysis
4. Check the overall architecture supports iterative refinement until content passes all reviews."""

RUBRIC_1_ARCHITECTURE_REVIEW = {
    "repomix_integration": "str - (20 points) Does the tool use repomix (npx repomix or similar) for both source and target repos, with support for remote GitHub repos?",
    "feedback_loop_architecture": "str - (30 points) Is there clear evidence of analysis → draft → review → re-analysis loops with feedback passed back to analysis stages?",
    "multiple_reviewers": "str - (25 points) Are there multiple distinct reviewers (at least: grounding reviewer, philosophy/patterns reviewer, and one additional reviewer)?",
    "human_review_integration": "str - (15 points) Is there a human review step that allows comments and triggers re-analysis with feedback?",
    "isolated_processes": "str - (10 points) Are reviews performed as separate, isolated processes as specified in requirements?",
    "score": "float - Score between 0 and 100. Sum the points earned from each criterion above.",
}


# Define Semantic Test 2: Run Tool & Validate Output Structure

STEPS_2_RUN_AND_VALIDATE_STRUCTURE = f"""1. Find the README file that explains how to use the cross-repo improvement tool.
2. Based on the README, determine the correct command to run the tool with these parameters:
   - Source repo: {SOURCE_REPO}
   - Target repo: {TARGET_REPO}
   - Improvement request: "{IMPROVEMENT_REQUEST}"
3. Run the command. This will take 10-20 minutes as it:
   - Runs repomix on both GitHub repositories
   - Performs multi-phase analysis with feedback loops
   - Generates comprehensive outputs
4. Monitor the execution and check for any errors in stdout/stderr.
5. After completion, examine the output structure:
   - Check if a timestamped or contextually-named output directory was created
   - Look for a comprehensive analysis file that includes:
     * High-level opportunities overview
     * Individual opportunity breakdowns
     * Value/rationale for each opportunity
   - Look for separate detailed proposal files for each opportunity containing:
     * Full context from the source repo
     * Implementation guidance for the target repo
     * Sufficient detail for handoff to a team without source access
   - Check if file/directory names are based on the input context (repos and request)
   - Look for evidence of review iterations (feedback files, iteration logs, etc.)
6. Verify the tool completed successfully and produced organized, structured output."""

RUBRIC_2_RUN_AND_VALIDATE_STRUCTURE = {
    "readme_exists": "str - (5 points) Does a comprehensive README exist with clear usage instructions?",
    "tool_runs_successfully": "str - (20 points) Does the tool run without fatal errors?",
    "repomix_execution": "str - (10 points) Is there evidence the tool ran repomix on both repositories?",
    "comprehensive_analysis_file": "str - (15 points) Does output include comprehensive analysis with high-level to detailed opportunity breakdown and rationale?",
    "separate_proposal_files": "str - (15 points) Are individual opportunities documented in separate detailed proposal files?",
    "contextual_naming": "str - (10 points) Are output directories and files named based on the repos and request context?",
    "implementation_guidance": "str - (15 points) Do proposals include detailed implementation guidance suitable for handoff?",
    "review_iteration_evidence": "str - (10 points) Is there evidence of feedback loop iterations (feedback files, version history, iteration logs)?",
    "score": "float - Score between 0 and 100. Sum the points earned from each criterion above.",
}


# Define Semantic Test 3: Deep Quality Validation

STEPS_3_DEEP_QUALITY_VALIDATION = f"""1. Find the README file that explains how to use the cross-repo improvement tool.
2. Based on the README, determine the correct command to run the tool with these parameters:
   - Source repo: {SOURCE_REPO}
   - Target repo: {TARGET_REPO}
   - Improvement request: "{IMPROVEMENT_REQUEST}"
3. Run the command. This will take 10-20 minutes as it runs repomix and performs multi-phase analysis.
4. After completion, deeply examine the comprehensive analysis file:
   - Are the identified opportunities actually present in github/spec-kit's codebase/patterns?
   - Do the opportunities make sense for github/copilot-cli's purpose and architecture?
   - Is the analysis grounded in actual code, patterns, and design from both repos (not generic advice)?
   - Does it demonstrate deep understanding of both tools' purposes and design philosophies?
5. Select 2-3 individual proposal files and analyze them deeply:
   - Specificity: Do proposals reference actual code, patterns, or structures from spec-kit?
   - Applicability: Are suggestions realistic and appropriate for copilot-cli's architecture?
   - Implementation guidance: Is the guidance specific enough for a team to implement?
   - Completeness: Could a team implement these proposals without accessing spec-kit's source code?
   - Value: Would these improvements genuinely benefit copilot-cli users and align with its goals?
6. Check for review quality evidence:
   - Are proposals grounded to both repos (not generic "best practices")?
   - Do suggestions align with copilot-cli's existing patterns and philosophy?
   - Is there evidence that multiple reviewers influenced the final quality?
7. Overall assessment:
   - Would the GitHub team maintaining copilot-cli find this analysis valuable?
   - Are suggestions actionable, well-reasoned, and worth considering?
   - Does the analysis justify the tool's complex multi-reviewer feedback loop architecture?"""

RUBRIC_3_DEEP_QUALITY_VALIDATION = {
    "opportunities_grounded_in_source": "str - (20 points) Do opportunities reference actual patterns/code from spec-kit (not generic suggestions)?",
    "appropriate_for_target": "str - (15 points) Do suggestions make sense for copilot-cli's purpose, architecture, and user experience?",
    "deep_understanding": "str - (15 points) Does analysis demonstrate deep understanding of both tools' designs and philosophies?",
    "proposal_specificity": "str - (15 points) Do proposals include specific references to source patterns rather than generic advice?",
    "implementation_completeness": "str - (15 points) Is implementation guidance detailed enough for handoff without requiring source access?",
    "actionability": "str - (10 points) Are suggestions realistic, actionable, and well-reasoned?",
    "overall_value": "str - (10 points) Would the GitHub copilot-cli team find this analysis genuinely valuable?",
    "score": "float - Score between 0 and 100. Sum the points earned from each criterion above.",
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
        logger.info("Running semantic test 1: Architecture & Implementation Review...")
        result_1 = await semantic_test(
            steps=STEPS_1_ARCHITECTURE_REVIEW,
            rubric=RUBRIC_1_ARCHITECTURE_REVIEW,
            context=instructions,
            working_dir=Path("/project"),
        )

        logger.info("Running semantic test 2: Run Tool & Validate Output Structure...")
        result_2 = await semantic_test(
            steps=STEPS_2_RUN_AND_VALIDATE_STRUCTURE,
            rubric=RUBRIC_2_RUN_AND_VALIDATE_STRUCTURE,
            context=instructions,
            working_dir=Path("/project"),
        )

        logger.info("Running semantic test 3: Deep Quality Validation...")
        result_3 = await semantic_test(
            steps=STEPS_3_DEEP_QUALITY_VALIDATION,
            rubric=RUBRIC_3_DEEP_QUALITY_VALIDATION,
            context=instructions,
            working_dir=Path("/project"),
        )

        # Calculate final score as weighted average
        # Weights: architecture (20%), structure (35%), quality (45%)
        final_score = result_1.score * 0.20 + result_2.score * 0.35 + result_3.score * 0.45

        metadata = {
            "instructions": instructions,
            "test_repos": {
                "source": SOURCE_REPO,
                "target": TARGET_REPO,
                "improvement_request": IMPROVEMENT_REQUEST,
            },
            "semantic_test_1_architecture_review": {
                "score": result_1.score,
                "details": result_1.metadata,
            },
            "semantic_test_2_run_and_validate_structure": {
                "score": result_2.score,
                "details": result_2.metadata,
            },
            "semantic_test_3_deep_quality_validation": {
                "score": result_3.score,
                "details": result_3.metadata,
            },
            "final_score": final_score,
            "scoring_weights": {
                "architecture_review": "20%",
                "run_and_validate_structure": "35%",
                "deep_quality_validation": "45%",
            },
        }

        write_test_result(output_dir, test_id, final_score, metadata)
        logger.info(f"Test completed with final score: {final_score:.1f}/100")
        return 0

    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        metadata = {
            "instructions": instructions,
            "test_repos": {
                "source": SOURCE_REPO,
                "target": TARGET_REPO,
                "improvement_request": IMPROVEMENT_REQUEST,
            },
            "error": str(e),
        }
        write_test_result(output_dir, test_id, 0, metadata)
        return 0


if __name__ == "__main__":
    sys.exit(main())
