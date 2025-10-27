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


# Define Semantic Test 1: Check Dependencies

STEPS_1_DEPENDENCIES = """1. Explore the code in the project directory to find where dependencies are defined.
2. Look for dependency files like pyproject.toml, requirements.txt, package.json, etc.
3. Check if the code uses:
   - DDGS | Dux Distributed Global Search library for web search
   - Claude Code or Agent SDK (claude-code-sdk or claude-agent-sdk) for AI interactions
4. Verify these dependencies are actually imported and used in the implementation."""

RUBRIC_1_DEPENDENCIES = {
    "duckduckgo_dependency": "str - (50 points) Does the solution use ddgs Python library for searching news? Check both dependency files and actual imports/usage in code.",
    "claude_sdk_dependency": "str - (50 points) Does the solution use Claude Code or Agent SDK (claude-code-sdk or claude-agent-sdk) for AI interactions? Check both dependency files and actual imports/usage in code.",
    "score": "float - Score between 0 and 100 based on the above criteria. Sum the points earned from each criterion.",
}


# Define Semantic Test 2: Run Tool and Validate Output Structure

STEPS_2_RUN_AND_VALIDATE = """1. Find the README.md file that explains how to use the tool and what commands to run.
2. Based on the README, 
  - Install any required dependencies if not already installed.
  - Determine the correct command to run the tool with a simple test topic like "Python programming news".
3. Run the tool. This may take up to 15 minutes as it needs to search the web and process results.
4. Check if the tool runs successfully without errors.
   - If the tool fails, note down an overall score of 0!
5. Verify the CLI interface:
   - Accepts a topic as input (command-line argument or interactive prompt)
   - Provides clear output or progress messages
   - Completes successfully
6. Find the output file(s):
   - Should be a markdown file (.md extension)
   - Should contain the research results
7. Examine the markdown file structure:
   - Is there a summary section with findings?
   - Are there citations/references?
   - Are URLs included in the citations?
   - Are any dates included (publication dates of articles)?"""

RUBRIC_2_RUN_AND_VALIDATE = {
    "readme_exists": "str - (5 points) Does a README.md exist with clear usage and installation instructions?",
    "tool_runs_successfully": "str - (20 points) Does the tool run without errors when given a topic?",
    "cli_accepts_topic": "str - (10 points) Does the CLI properly accept a topic as input?",
    "markdown_output_created": "str - (15 points) Is a markdown file created with the results?",
    "has_summary": "str - (15 points) Does the markdown file contain a summary of findings?",
    "has_citations": "str - (15 points) Are there citations/references to source articles?",
    "citations_have_urls": "str - (10 points) Do citations include URLs to the source articles?",
    "citations_have_dates": "str - (10 points) Do citations include publication dates?",
    "score": "float - Score between 0 and 100 based on the above criteria. Sum the points earned from each criterion.",
}


# Define Semantic Test 3: Run with Specific Topic and Validate Quality

STEPS_3_QUALITY_VALIDATION = """1. Find the README.md file that explains how to use the tool and what commands to run.
2. Based on the README, 
  - Install any required dependencies if not already installed.
  - Determine the correct command to run the tool with a simple test topic like "Python programming news".1. Based on the README, determine the command to run the tool with this specific topic: \
"artificial intelligence breakthroughs 2025"
3. Run the tool with this topic. This may take several minutes.
   - If the tool fails, note down an overall score of 0!
4. After completion, find and read the generated markdown file.
5. Evaluate the search quality:
   - Are the articles actually relevant to AI breakthroughs?
   - Did it pick high quality sources?
   - Are the articles recent (2024-2025)?
6. Evaluate the AI analysis quality:
   - Does it synthesize information from multiple sources?
7. Evaluate citation quality:
   - Are there at least 3-5 different source articles cited?
   - Does each citation include the article URL?
   - Are there publication dates included in the generated synthesis?
   - Are the citations properly formatted and easy to verify?
8. Evaluate content grounding:
   - Is the summary based on actual information from the cited sources?
   - Are there any unsupported claims or hallucinations?
   - Does it properly attribute information to specific sources?"""

RUBRIC_3_QUALITY_VALIDATION = {
    "tool_runs_successfully": "str - (10 points) Does the tool run successfully with the test topic?",
    "articles_are_relevant": "str - (20 points) Are the found articles actually relevant to the topic?",
    "articles_are_recent": "str - (20 points) Are the articles recent (2024-2025)?",
    "synthesizes_multiple_sources": "str - (10 points) Does it synthesize information from multiple sources?",
    "has_multiple_citations": "str - (10 points) Are there at least 3-5 different source articles cited?",
    "citations_include_urls": "str - (5 points) Do all citations include URLs?",
    "citations_include_dates": "str - (5 points) Do all citations include publication dates?",
    "includes_dates": "str - (5 points) Are publication dates included in the summary or citations?",
    "content_is_grounded": "str - (5 points) Is the content grounded in actual source material without hallucinations?",
    "proper_attribution": "str - (10 points) Is information properly attributed to specific sources?",
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
    """Test script for news_research_tool task."""
    return asyncio.run(run_test(test_id, output_dir, instructions_file))


async def run_test(test_id: str, output_dir: Path, instructions_file: Path | None) -> int:
    instructions = get_instructions_from_file_or_default(instructions_file=instructions_file)

    try:
        logger.info("Running semantic test 1: Checking for correct dependencies...")
        result_1 = await semantic_test(
            steps=STEPS_1_DEPENDENCIES,
            rubric=RUBRIC_1_DEPENDENCIES,
            context=instructions,
            working_dir=Path("/project"),
        )

        logger.info("Running semantic test 2: Running tool and validating output structure...")
        result_2 = await semantic_test(
            steps=STEPS_2_RUN_AND_VALIDATE,
            rubric=RUBRIC_2_RUN_AND_VALIDATE,
            context=instructions,
            working_dir=Path("/project"),
        )

        
        logger.info("Running semantic test 3: Running tool with specific topic and validating quality...")
        result_3 = await semantic_test(
            steps=STEPS_3_QUALITY_VALIDATION,
            rubric=RUBRIC_3_QUALITY_VALIDATION,
            context=instructions,
            working_dir=Path("/project"),
        )

        # Calculate final score with weighted average
        # Weights: dependencies (20%), run and validate (30%), quality validation (50%)
        final_score = (
            result_1.score * 0.20
            + result_2.score * 0.30
            # + result_3.score * 0.50
        )

        metadata = {
            "instructions": instructions,
            "semantic_test_1_dependencies": {
                "score": result_1.score,
                "details": result_1.metadata,
            },
            "semantic_test_2_run_and_validate": {
                "score": result_2.score,
                "details": result_2.metadata,
            },
            "semantic_test_3_quality_validation": {
                "score": result_3.score,
                "details": result_3.metadata,
            },
            "final_score": final_score,
            "scoring_weights": {
                "dependencies": "20%",
                "run_and_validate": "30%",
                "quality_validation": "50%",
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
