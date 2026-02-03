# Copyright (c) Microsoft. All rights reserved.

"""
Test script for PDF HR Q4 data analysis task.

Uses semantic test to check if the answer contains the correct median salaries
for each performance rating.

Expected values:
- Rating 1: $60,000
- Rating 2: $63,550
- Rating 3: $80,700
- Rating 4: $85,600
- Rating 5: $89,250

Scoring: 20 points per correct rating (within 1% tolerance), 100 points total.
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

STEPS = """1. Find and read the file `data_analysis_answer.txt` in the project directory.
2. Compare the answer against the expected median salaries for each performance rating:
   - Rating 1: $60,000 (acceptable range: $59,400 - $60,600)
   - Rating 2: $63,550 (acceptable range: $62,915 - $64,186)
   - Rating 3: $80,700 (acceptable range: $79,893 - $81,507)
   - Rating 4: $85,600 (acceptable range: $84,744 - $86,456)
   - Rating 5: $89,250 (acceptable range: $88,358 - $90,143)
3. For each rating, check if the value in the answer is within the acceptable range.
4. Award 20 points for each correct rating."""

RUBRIC = {
    "rating_1_correct": "str - (20 points) Is Rating 1's median salary within tolerance?",
    "rating_2_correct": "str - (20 points) Is Rating 2's median salary within tolerance?",
    "rating_3_correct": "str - (20 points) Is Rating 3's median salary within tolerance?",
    "rating_4_correct": "str - (20 points) Is Rating 4's median salary within tolerance?",
    "rating_5_correct": "str - (20 points) Is Rating 5's median salary within tolerance?",
    "score": "float - Score between 0 and 100. Sum the points earned from each rating criterion.",
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
    """Test script for PDF HR Q4 data analysis task."""
    return asyncio.run(run_test(test_id, output_dir, instructions_file))


async def run_test(test_id: str, output_dir: Path, instructions_file: Path | None) -> int:
    instructions = get_instructions_from_file_or_default(instructions_file=instructions_file)

    try:
        logger.info("Running semantic test for Q4: Median salary by performance rating...")
        result = await semantic_test(
            steps=STEPS,
            rubric=RUBRIC,
            context=instructions,
            working_dir=Path("/project"),
        )

        metadata = {
            "instructions": instructions,
            "groundtruth": {
                "rating_1": 60000,
                "rating_2": 63550,
                "rating_3": 80700,
                "rating_4": 85600,
                "rating_5": 89250,
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
