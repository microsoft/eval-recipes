# Copyright (c) Microsoft. All rights reserved.

"""
Test script for PDF HR Q5 data analysis task.

Uses semantic test to check if the answer identifies the correct division
and percentage change.

Expected: Transportation & Electronics (+0.6 percentage points)

Scoring:
- 50 points for correct division name
- 50 points for correct percentage (within 1% tolerance)
"""

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

STEPS = """\
1. Find and read the file `data_analysis_answer.txt` in the project directory.
2. Check if the answer mentions the correct division: "Transportation & Electronics" \
(also accept "Transportation and Electronics").
3. Check if the answer includes the correct percentage change: 0.6 percentage points \
(within a 0.2 tolerance, so 0.58 to 0.62 is acceptable).
"""

RUBRIC = {
    "division_correct": "str - (50 points) Does the answer identify 'Transportation & Electronics' \
(or 'Transportation and Electronics') as the division?",
    "percentage_correct": "str - (50 points) Does the answer include the correct percentage change \
of 0.6 percentage points (within tolerance)?",
    "score": "float - Score between 0 and 100. Award 50 points for correct division, \
50 points for correct percentage.",
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
    """Test script for PDF HR Q5 data analysis task."""
    return asyncio.run(run_test(test_id, output_dir, instructions_file))


async def run_test(test_id: str, output_dir: Path, instructions_file: Path | None) -> int:
    instructions = get_instructions_from_file_or_default(instructions_file=instructions_file)

    try:
        logger.info("Running semantic test for Q5: Division with largest performance increase...")
        result = await semantic_test(
            steps=STEPS,
            rubric=RUBRIC,
            context=instructions,
            working_dir=Path("/project"),
        )

        metadata = {
            "instructions": instructions,
            "groundtruth": {
                "division": "Transportation & Electronics",
                "percentage_change": 0.6,
            },
            "semantic_test_result": {
                "score": result.score,
                "details": result.metadata,
            },
        }

        write_test_result(output_dir, test_id, result.score, metadata)
        logger.info(f"Test completed with score: {result.score:.1f}/100")
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
