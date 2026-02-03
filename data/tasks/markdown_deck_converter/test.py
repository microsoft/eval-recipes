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

# Define Semantic Test 1: CLI Functionality and PowerPoint Generation

STEPS_1_CLI_AND_GENERATION = """1. Find the README that the agent should have created to understand how to use the tool.
2. Verify the README contains:
   - Installation/setup instructions
   - Command examples for building presentations
   - Examples of using themes
   - Clear usage documentation
3. Test basic conversion:
   - Use the simple_pitch.md test file from the data/ directory
   - Run the tool to convert it to PowerPoint
   - Verify a .pptx file is created
4. Test with theme option:
   - Use any of the test markdown files
   - Try converting with a theme option (if available per the instructions)
   - Verify it runs without errors
5. Validate the tool handles the markdown structure:
   - Slides separated by `---`
   - Headers with `#` become slide titles
   - Bullet points and content are preserved
6. Check error handling:
   - Try running with invalid inputs
   - Verify the tool provides helpful error messages"""

RUBRIC_1_CLI_AND_GENERATION = {
    "readme_exists": "str - (10 points) Does a comprehensive README exist with clear usage instructions and examples?",
    "readme_quality": "str - (10 points) Does the README include installation, command examples, and theme usage?",
    "tool_runs_basic": "str - (25 points) Does the tool successfully convert a markdown file to PowerPoint format?",
    "pptx_file_created": "str - (25 points) Is a valid .pptx file created that can be opened?",
    "theme_support": "str - (15 points) Does the tool support theme options as specified in the instructions?",
    "markdown_structure": "str - (15 points) Does the tool properly parse markdown structure (--- separators, # headers)?",
    "score": "float - Score between 0 and 100 based on the above criteria. Sum the points earned from each criterion.",
}


# Define Semantic Test 2: PowerPoint Editability and Structure

STEPS_2_EDITABILITY_VALIDATION = """1. Convert one research_presentation.md from the data/ directory to PowerPoint using the tool.
2. Open and examine the generated .pptx file structure:
   - Use python-pptx library or similar to inspect the file programmatically
   - Check slide count matches markdown structure (count the --- separators)
   - Verify slides have proper titles from # headers
3. Validate editability requirements (THIS IS THE CRITICAL REQUIREMENT):
   - CRITICAL: Check if text is in editable placeholders/text frames (not absolute positioned text boxes)
   - Verify the PowerPoint uses proper slide layouts with placeholders
   - Ensure text boxes are NOT using absolute positioning that prevents editing
   - Check that content follows PowerPoint best practices for editability
   - The instructions specifically say "Use whatever PowerPoint has built-in for making editable text areas"
4. Examine slide content preservation:
   - Bullet points preserved as PowerPoint bullet lists
   - Tables (if any) converted appropriately
   - Text formatting maintained where possible
5. Test editability practically:
   - Can you conceptually move text elements around?
   - Would a non-technical person be able to edit this like a normal PowerPoint?"""

RUBRIC_2_EDITABILITY_VALIDATION = {
    "file_opens": "str - (10 points) Does the generated .pptx file open without errors?",
    "slide_count_correct": "str - (10 points) Does the slide count match the markdown structure (--- separators)?",
    "titles_correct": "str - (10 points) Are slide titles properly extracted from # headers?",
    "uses_placeholders": "str - (35 points) CRITICAL: Does the PowerPoint use proper placeholders/text frames instead of absolutely positioned text boxes? This is THE key requirement from the instructions.",
    "proper_layouts": "str - (15 points) Does the tool use PowerPoint slide layouts appropriately for editability?",
    "content_preserved": "str - (10 points) Is markdown content (bullets, text) properly preserved in the PowerPoint?",
    "actually_editable": "str - (10 points) Can text be easily moved and edited like a normal PowerPoint slide by non-technical users?",
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
    """Test script for markdown_deck_converter task."""
    return asyncio.run(run_test(test_id, output_dir, instructions_file))


async def run_test(test_id: str, output_dir: Path, instructions_file: Path | None) -> int:
    instructions = get_instructions_from_file_or_default(instructions_file=instructions_file)

    try:
        logger.info("Running semantic test 1: CLI functionality and PowerPoint generation...")
        result_1 = await semantic_test(
            steps=STEPS_1_CLI_AND_GENERATION,
            rubric=RUBRIC_1_CLI_AND_GENERATION,
            context=instructions,
            working_dir=Path("/project"),
        )

        logger.info("Running semantic test 2: PowerPoint editability validation (CRITICAL requirement)...")
        result_2 = await semantic_test(
            steps=STEPS_2_EDITABILITY_VALIDATION,
            rubric=RUBRIC_2_EDITABILITY_VALIDATION,
            context=instructions,
            working_dir=Path("/project"),
        )

        # Calculate final score with weighted average
        # Weights: CLI and generation (25%), Editability validation (45%), Comprehensive (30%)
        # Editability gets highest weight because it's the core requirement emphasized in instructions
        final_score = result_1.score * 0.4 + result_2.score * 0.6

        metadata = {
            "instructions": instructions,
            "semantic_test_1_cli_and_generation": {
                "score": result_1.score,
                "details": result_1.metadata,
            },
            "semantic_test_2_editability_validation": {
                "score": result_2.score,
                "details": result_2.metadata,
            },
            "final_score": final_score,
            "scoring_weights": {
                "cli_and_generation": "40%",
                "editability_validation": "60%",
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
