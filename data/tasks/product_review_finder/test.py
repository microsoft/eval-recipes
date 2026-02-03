# Copyright (c) Microsoft. All rights reserved.

import asyncio
from pathlib import Path
import sys

import click
from loguru import logger

from eval_recipes.benchmarking.evaluation.semantic_test import semantic_test
from eval_recipes.benchmarking.evaluation.test_utils import (
    get_instructions_from_file_or_default,
    get_test_id_from_env_or_default,
    write_test_result,
)

# Define Semantic Test 1: Check Multi-Stage Architecture

AGENT_SDK_DEFINITION = """The solution should use an Agent SDK, such as Claude Agent/Code SDK, Microsoft Agent Framework, Microsoft Amplifier (https://github.com/microsoft/amplifier/tree/next), OpenAI Codex CLI, or others that are similarly capable. These SDKs must have the following functionality:
- Automatic Context Management to ensure your agent doesn't run out of context.
- Rich tool ecosystem: File operations, code execution, web search, and MCP extensibility
- Excels at code generation and effectively gives the agent a "computer" where it can find appropriate files, write and edit files, lint the code, run it, debug, edit, and sometimes take these actions iteratively until it succeeds.
- APIs like OpenAI's chat completions or Responses API, Anthropic's Messages API, or Azure OpenAI alone are NOT sufficient and should not recieve any credit."""

STEPS_1_ARCHITECTURE = f"""{AGENT_SDK_DEFINITION}

1. Explore the code for the product review tool to understand the architecture.
2. Check for required dependencies:
   - Look for dependency files (pyproject.toml, requirements.txt, etc.)
   - Verify the code uses an Agent SDK (see definition above) for LLM interactions
   - Verify the code uses ddgs (Dux Distributed Global Search) for finding reviews
   - Check that these dependencies are actually imported and used in the code
   - Confirm the AI framework is an Agent SDK with required capabilities, not just a plain API client
3. Look for evidence of the required multi-stage pipeline:
   - A writer component that creates the initial draft
   - An accuracy-reviewer that checks for proper citations and hallucinations
   - A completeness-reviewer that checks for all required sections
   - A synthesis-reviewer that checks for coherent analysis
4. Check if the code has logic for:
   - Passing drafts between components with feedback
   - Looping back to writer when reviewers find issues
   - Re-running all previous reviewers after writer makes changes
5. Check if there's support for user feedback with [bracket-enclosed-comments]."""

RUBRIC_1_ARCHITECTURE = {
    "agent_sdk_identified": "str - Name of Agent SDK used, or 'None'",
    "agent_sdk_dependency": "str - (15 points) Does solution use qualifying Agent SDK (Claude Agent/Code SDK, Microsoft Agent Framework, Amplifier, OpenAI Codex CLI) for LLM interactions? Must provide automatic context management, rich tool ecosystem, and iterative code capabilities. NOT plain API clients.",
    "ddgs_dependency": "str - (10 points) Does the solution use ddgs (Dux Distributed Global Search) for finding reviews? Check both dependency files and actual imports/usage in code.",
    "writer_component": "str - (12 points) Is there a clear writer component that creates the initial markdown report draft?",
    "accuracy_reviewer": "str - (12 points) Is there an accuracy-reviewer component that validates citations and checks for hallucinations?",
    "completeness_reviewer": "str - (12 points) Is there a completeness-reviewer that checks for all required sections?",
    "synthesis_reviewer": "str - (12 points) Is there a synthesis-reviewer that validates coherent analysis and recommendations?",
    "feedback_loops": "str - (12 points) Does the code implement feedback loops where reviewers can send work back to the writer?",
    "sequential_review": "str - (10 points) After writer changes, do drafts go back through all previous reviewers (not just the one that failed)?",
    "user_feedback": "str - (5 points) Is there support for user feedback with bracket-enclosed comments?",
    "score": "float - Score between 0 and 100 based on the above criteria. Sum the points earned from each criterion.",
}


# Define Semantic Test 2: Run Tool and Validate CLI + Output

STEPS_2_RUN_AND_VALIDATE = """1. Find the README in the project directory that explains how to use the tool.
2. Based on the README, determine the correct command to run the tool with a simple test product like "ROG Xbox Ally X".
3. If the README mentions the --category flag, optionally test with --category "handheld game console".
4. Run the tool. This may take up to 15 minutes as it needs to search the web and go through multiple review stages.
   - If the tool fails or times out, note an overall score of 0 for this test!
5. Verify the CLI interface:
   - Tool accepts a product name as input
   - Tool runs without crashing or errors
   - Tool provides clear output or progress messages
   - Tool completes successfully
6. Find the generated markdown output file:
   - Should be timestamped and named after the product
   - Should be a valid markdown file (.md extension)
7. Examine the markdown report structure - check if it includes:
   - Overall consensus on product quality
   - Key strengths section
   - Key weaknesses section
   - Ratings or scores from reviews
   - Direct quotes from sources
   - Citations with URLs
   - A final recommendation
8. Verify citations are present and properly formatted with URLs."""

RUBRIC_2_RUN_AND_VALIDATE = {
    "readme_exists": "str - (5 points) Does a README exist with clear usage instructions?",
    "tool_runs_successfully": "str - (20 points) Does the tool run without errors or crashes?",
    "cli_accepts_product": "str - (5 points) Does the CLI accept a product name as input?",
    "tool_completes": "str - (10 points) Does the tool complete successfully within reasonable time?",
    "markdown_output_created": "str - (10 points) Is a timestamped markdown file created?",
    "has_consensus": "str - (10 points) Does the report include an overall consensus section?",
    "has_strengths": "str - (8 points) Are key strengths identified?",
    "has_weaknesses": "str - (8 points) Are key weaknesses identified?",
    "has_ratings": "str - (5 points) Are ratings or scores mentioned?",
    "has_quotes": "str - (5 points) Are there direct quotes from sources?",
    "has_citations": "str - (10 points) Are there citations with URLs?",
    "has_recommendation": "str - (4 points) Is there a final recommendation?",
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
    """Test script for product_review_finder task."""
    return asyncio.run(run_test(test_id, output_dir, instructions_file))


async def run_test(test_id: str, output_dir: Path, instructions_file: Path | None) -> int:
    instructions = get_instructions_from_file_or_default(instructions_file=instructions_file)

    try:
        logger.info("Running semantic test 1: Checking multi-stage architecture...")
        result_1 = await semantic_test(
            steps=STEPS_1_ARCHITECTURE,
            rubric=RUBRIC_1_ARCHITECTURE,
            context=instructions,
            working_dir=Path("/project"),
        )

        logger.info("Running semantic test 2: Running tool and validating CLI + output structure...")
        result_2 = await semantic_test(
            steps=STEPS_2_RUN_AND_VALIDATE,
            rubric=RUBRIC_2_RUN_AND_VALIDATE,
            context=instructions,
            working_dir=Path("/project"),
        )

        # Calculate final score with weighted average
        # Weights: architecture (40%), run and validate (60%)
        final_score = result_1.score * 0.40 + result_2.score * 0.60

        metadata = {
            "instructions": instructions,
            "semantic_test_1_architecture": {
                "score": result_1.score,
                "details": result_1.metadata,
            },
            "semantic_test_2_run_and_validate": {
                "score": result_2.score,
                "details": result_2.metadata,
            },
            "final_score": final_score,
            "scoring_weights": {
                "architecture": "40%",
                "run_and_validate": "60%",
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
