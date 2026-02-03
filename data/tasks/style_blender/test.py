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

# Define Semantic Test 1: Check for 3-Stage Pipeline Implementation

STEPS_1_PIPELINE_ARCHITECTURE = """1. Explore the code under scenarios/style_blender/ to understand the architecture.
2. Look for evidence of a 3-stage pipeline:
   - Stage 1: Style Analysis (extracting style from each writer)
   - Stage 2: Style Blending (combining profiles intelligently)
   - Stage 3: Content Generation (generating sample writings)
3. Check if each stage is clearly separated and performs its specific function.
4. Verify that the pipeline processes multiple writers (at least 2) as specified."""

RUBRIC_1_PIPELINE_ARCHITECTURE = {
    "stage_1_exists": "str - (30 points) Is there a clear Stage 1 that analyzes individual writer styles and extracts characteristics (tone, vocabulary, sentence structure, etc.)?",
    "stage_2_exists": "str - (30 points) Is there a clear Stage 2 that blends multiple style profiles intelligently (not just averaging)?",
    "stage_3_exists": "str - (30 points) Is there a clear Stage 3 that generates sample content in the blended style?",
    "pipeline_structure": "str - (10 points) Are the stages properly separated and organized in the code?",
    "score": "float - Score between 0 and 100 based on the above criteria. Sum the points earned from each criterion.",
}


# Define Semantic Test 2: Run Tool and Validate CLI Interface + Output Organization

STEPS_2_RUN_AND_VALIDATE = """1. Find the README.md under scenarios/style_blender/ to understand how to use the tool.
2. Create test input files with sample writings from at least 2 different fictional writers with distinct styles.
   - Create simple test files (e.g., writer1.txt and writer2.txt) with different writing characteristics
   - Each should have enough text to analyze (at least a few paragraphs)
3. Based on the README, determine the correct command to run the tool with these test inputs.
4. Run the tool. This may take several minutes as it involves AI generation.
5. Verify the CLI worked properly:
   - Tool runs without errors
   - Accepts input file/directory specifications
   - Properly validates inputs (checks for minimum 2 writers)
   - Provides clear output or error messages
6. Check the output organization:
   - A timestamped output directory was created (e.g., .data/style_blender/TIMESTAMP/)
   - Individual writer profiles exist (writer_*.json)
   - Blended profile exists (blended_profile.json)
   - Prompts are saved (prompt_*.txt files)
   - Index file exists (index.md)
   - Clean output/ subdirectory exists with only generated samples
   - Output/ subdirectory is separate from diagnostic files
7. Verify the generated samples exist and are in the expected format."""

RUBRIC_2_RUN_AND_VALIDATE = {
    "readme_exists": "str - (5 points) Does a comprehensive README.md exist with usage instructions?",
    "tool_runs_successfully": "str - (15 points) Does the tool run without errors when given proper inputs?",
    "cli_input_handling": "str - (10 points) Does the CLI accept input files/directories as specified?",
    "input_validation": "str - (10 points) Does the tool validate inputs (e.g., minimum 2 writers)?",
    "timestamped_directory": "str - (10 points) Is a timestamped output directory created?",
    "individual_profiles": "str - (10 points) Are individual writer profiles saved to JSON files?",
    "blended_profile": "str - (10 points) Is the blended profile saved to a JSON file?",
    "prompts_saved": "str - (10 points) Are prompts saved to disk for diagnostics?",
    "clean_output_subdir": "str - (10 points) Is there a clean output/ subdirectory with only samples?",
    "index_file": "str - (5 points) Is there an index.md file with overview?",
    "samples_generated": "str - (5 points) Are sample writings generated in the output/ subdirectory?",
    "score": "float - Score between 0 and 100 based on the above criteria. Sum the points earned from each criterion.",
}


# Define Semantic Test 3: Run Tool with Provided Test Data and Validate Output Quality

STEPS_3_RUN_WITH_TEST_DATA = """1. Use the provided test data in the data/ directory:
   - minimalist_prose/ contains 3 stories with simple, direct prose (short sentences, sparse description)
   - gothic_horror/ contains 3 tales with ornate, archaic language (complex sentences, cosmic themes)
2. Based on the README, determine the correct command to run the tool with these two writer directories as input.
3. Run the tool using these very distinct writing styles. This may take several minutes.
4. After the tool completes, examine the style analysis (Stage 1):
   - Check the individual writer profiles (writer_*.json files)
   - Verify the profiles capture the distinct characteristics:
     * Minimalist: short sentences, simple vocabulary, direct prose
     * Gothic: long sentences, archaic vocabulary, ornate descriptions
   - Ensure profiles are detailed and diagnostic
5. Examine the blended profile (blended_profile.json):
   - Check if it intelligently combines both styles (not just averaging)
   - Look for attribution showing contributions from each writer
6. Analyze the generated samples in the output/ subdirectory:
   - Are 3-5 samples generated?
   - Are they 300-500 words each?
   - Do they demonstrate actual blending of both styles?
   - Look for evidence of both minimalist clarity AND gothic atmosphere
   - Check for topic variety
   - Verify they're coherent, not just random mixing
7. Check diagnostic files are present (prompts, index.md)."""

RUBRIC_3_RUN_WITH_TEST_DATA = {
    "tool_runs_successfully": "str - (10 points) Does the tool run without errors with the test data?",
    "profiles_capture_minimalist": "str - (10 points) Does the minimalist_prose profile capture short sentences and simple vocabulary?",
    "profiles_capture_gothic": "str - (10 points) Does the gothic_horror profile capture ornate language and complex sentences?",
    "profiles_are_detailed": "str - (10 points) Are both profiles detailed enough to be diagnostic?",
    "blending_is_intelligent": "str - (15 points) Does the blended profile show intelligent combination rather than averaging?",
    "correct_sample_count": "str - (5 points) Are 3-5 samples generated?",
    "correct_sample_length": "str - (10 points) Are samples 300-500 words each?",
    "demonstrates_both_styles": "str - (15 points) Do samples show characteristics from both minimalist AND gothic styles?",
    "topic_variety": "str - (5 points) Do samples use different topics?",
    "coherent_writing": "str - (10 points) Are samples coherent and well-structured?",
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
    """Test script for style_blender task."""
    return asyncio.run(run_test(test_id, output_dir, instructions_file))


async def run_test(test_id: str, output_dir: Path, instructions_file: Path | None) -> int:
    instructions = get_instructions_from_file_or_default(instructions_file=instructions_file)

    try:
        logger.info("Running semantic test 1: Checking for 3-stage pipeline architecture...")
        result_1 = await semantic_test(
            steps=STEPS_1_PIPELINE_ARCHITECTURE,
            rubric=RUBRIC_1_PIPELINE_ARCHITECTURE,
            context=instructions,
            working_dir=Path("/project"),
        )

        logger.info("Running semantic test 2: Running tool and validating CLI + output organization...")
        result_2 = await semantic_test(
            steps=STEPS_2_RUN_AND_VALIDATE,
            rubric=RUBRIC_2_RUN_AND_VALIDATE,
            context=instructions,
            working_dir=Path("/project"),
        )

        logger.info("Running semantic test 3: Running tool with test data and validating complete output quality...")
        result_3 = await semantic_test(
            steps=STEPS_3_RUN_WITH_TEST_DATA,
            rubric=RUBRIC_3_RUN_WITH_TEST_DATA,
            context=instructions,
            working_dir=Path("/project"),
        )

        # Calculate final score with weighted average
        # Weights: pipeline (20%), run and validate (35%), test data run and full validation (45%)
        final_score = result_1.score * 0.20 + result_2.score * 0.35 + result_3.score * 0.45

        metadata = {
            "instructions": instructions,
            "semantic_test_1_pipeline_architecture": {
                "score": result_1.score,
                "details": result_1.metadata,
            },
            "semantic_test_2_run_and_validate": {
                "score": result_2.score,
                "details": result_2.metadata,
            },
            "semantic_test_3_run_with_test_data": {
                "score": result_3.score,
                "details": result_3.metadata,
            },
            "final_score": final_score,
            "scoring_weights": {
                "pipeline_architecture": "20%",
                "run_and_validate": "35%",
                "test_data_full_validation": "45%",
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
