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


EXPECTED_OLLAMA_MODEL = "gemma3:4b-it-q4_K_M"


# Define Semantic Test 1: Dependencies, Implementation, and Prompt Quality

STEPS_1_IMPLEMENTATION = f"""1. Explore the code in the project directory to understand the implementation of this new image tagging tool.
2. Look for dependency files and the main Python script.
3. Verify the following:
   - The code uses Ollama for the local small language model
   - The specific model `{EXPECTED_OLLAMA_MODEL}` is referenced in the code
   - The prompt/system message includes clear examples of good tags (e.g., showing format like "outdoor, sunset, mountain, landscape")
   - The script can accept a folder path as input
   - The code outputs to a CSV file with columns: file_path, file_name, tags
4. Check if a README exists with installation and usage instructions
5. Verify the README explains how to install Ollama and pull the model"""

RUBRIC_1_IMPLEMENTATION = {
    "readme_exists": "str - (5 points) Does a README exist?",
    "readme_installation": "str - (5 points) Does the README explain how to install Ollama and pull the model?",
    "uses_ollama": "str - (20 points) Does the code use Ollama for image tagging?",
    "correct_model": f"str - (20 points) Is the model `{EXPECTED_OLLAMA_MODEL}` specified in the code?",
    "prompt_has_examples": "str - (25 points) Does the prompt/system message include clear examples of how tags should be formatted (e.g., 'outdoor, sunset, mountain, landscape')?",
    "folder_input": "str - (10 points) Can the script accept a folder path as input?",
    "csv_output_structure": "str - (15 points) Does the code output to a CSV file with columns: file_path, file_name, tags?",
    "score": "float - Score between 0 and 100 based on the above criteria. Sum the points earned from each criterion.",
}


# Define Semantic Test 2: Functional Run with Test Images

STEPS_2_FUNCTIONAL_RUN = """1. Ensure Ollama is running (start with `nohup ollama serve > /dev/null 2>&1 &` if needed, wait 5 seconds).
2. Find the README to understand how to run the script.
3. Locate the test images in the data directory.
   - Look for image files (png, jpg, jpeg, etc.) in the data/ directory
4. Based on the README, determine the command to run the script with the test images directory.
5. Run the script with the test images directory. This may take a few minutes.
   - If the script fails or times out after 15 minutes, note an overall score of 0 for this test!
6. After the script completes, find the generated CSV file.
7. Read the CSV file and verify:
   - It has the correct columns: file_path, file_name, tags (or similar)
   - There is at least one row for each test image found
   - Each row has tags generated (not empty)
   - Tags are in a reasonable format (comma-separated, space-separated, or similar)
8. Check the quality of the tags:
   - Since this is a small model, the tags don't need to be perfect or highly accurate
   - Tags should be English words/phrases (not gibberish or random characters)
   - Tags should be plausible descriptors that could apply to images (e.g., colors, objects, scenes, concepts)
   - It's acceptable if tags are generic or not fully accurate - the key is that the model attempted to generate reasonable descriptive text"""

RUBRIC_2_FUNCTIONAL_RUN = {
    "readme_instructions_clear": "str - (10 points) Does the README have clear instructions on how to run the script?",
    "script_runs_successfully": "str - (25 points) Does the script run without errors when given the test images directory?",
    "csv_file_created": "str - (15 points) Is a CSV file created after running the script?",
    "csv_has_correct_columns": "str - (15 points) Does the CSV have the required columns (file_path, file_name, tags or similar)?",
    "csv_has_all_images": "str - (10 points) Does the CSV contain entries for all test images found in the data directory?",
    "tags_generated": "str - (10 points) Are tags generated for each image (not empty)?",
    "tags_are_plausible": "str - (15 points) Are the tags plausible descriptors (English words/phrases, not gibberish) that could apply to images? Don't penalize for inaccuracy - focus on whether they're reasonable attempts at descriptive text.",
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
    """Test script for image_tagging task."""
    return asyncio.run(run_test(test_id, output_dir, instructions_file))


async def run_test(test_id: str, output_dir: Path, instructions_file: Path | None) -> int:
    instructions = get_instructions_from_file_or_default(instructions_file=instructions_file)

    try:
        logger.info("Running semantic test 1: Checking implementation and prompt quality...")
        result_1 = await semantic_test(
            steps=STEPS_1_IMPLEMENTATION,
            rubric=RUBRIC_1_IMPLEMENTATION,
            context=instructions,
            working_dir=Path("/project"),
        )

        logger.info("Running semantic test 2: Running script with test images and validating output...")
        result_2 = await semantic_test(
            steps=STEPS_2_FUNCTIONAL_RUN,
            rubric=RUBRIC_2_FUNCTIONAL_RUN,
            context=instructions,
            working_dir=Path("/project"),
        )

        # Calculate final score with weighted average
        # Weights: implementation (40%), functional run (60%)
        final_score = (
            result_1.score * 0.40
            + result_2.score * 0.60
        )

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
                "implementation": "40%",
                "functional_run": "60%",
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
