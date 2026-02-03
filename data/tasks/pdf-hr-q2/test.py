# Copyright (c) Microsoft. All rights reserved.

"""
Test script for PDF HR Q2 data analysis task.

Checks if the answer contains the correct employee count (263) within 1% tolerance.
Scoring: 100 points if within 1% tolerance, 0 otherwise.
"""

from pathlib import Path
import re
import sys

import click

from eval_recipes.benchmarking.evaluation.test_utils import (
    get_instructions_from_file_or_default,
    get_test_id_from_env_or_default,
    load_text_from_path_or_content,
    write_test_result,
)

GROUNDTRUTH_VALUE = 263
TOLERANCE_PERCENT = 1.0


def parse_integer_value(text: str) -> int | None:
    """
    Parse an integer value from text, handling various formats.

    Handles formats like:
    - "263"
    - "263 employees"
    - "There are 263 employees"

    Returns None if no valid integer is found.
    """
    # Remove commas (thousand separators)
    cleaned = text.replace(",", "")

    # Try to find an integer number
    match = re.search(r"\b(\d+)\b", cleaned)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


@click.command()
@click.option(
    "--answer",
    type=str,
    default=None,
    help="Path to answer text file, or answer content as string",
)
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
def main(answer: str | None, test_id: str, output_dir: Path, instructions_file: Path | None) -> int:
    """Test script for PDF HR Q2 data analysis task."""
    # Use default if not provided
    if answer is None:
        answer = str(Path(__file__).parents[0] / "data_analysis_answer.txt")

    # Load instructions from file
    instructions = get_instructions_from_file_or_default(instructions_file=instructions_file)

    # Load answer text - detect if it's a file path or content
    llm_output = load_text_from_path_or_content(answer)

    # Parse the integer value from the answer
    parsed_value = parse_integer_value(llm_output)

    if parsed_value is None:
        print(f"\nCould not parse integer value from: {llm_output!r}")
        percent_difference = None
        score = 0
    else:
        # Calculate percent difference
        percent_difference = abs(parsed_value - GROUNDTRUTH_VALUE) / GROUNDTRUTH_VALUE * 100

        # Binary scoring: 100 if within tolerance, 0 otherwise
        score = 100 if percent_difference <= TOLERANCE_PERCENT else 0

        print(f"\nGroundtruth value: {GROUNDTRUTH_VALUE}")
        print(f"Parsed value: {parsed_value}")
        print(f"Percent difference: {percent_difference:.4f}%")
        print(f"Within {TOLERANCE_PERCENT}% tolerance: {percent_difference <= TOLERANCE_PERCENT}")

    print(f"Score: {score}")

    metadata = {
        "instructions": instructions,
        "groundtruth": GROUNDTRUTH_VALUE,
        "llm_output": llm_output,
        "parsed_value": parsed_value,
        "percent_difference": percent_difference,
        "tolerance_percent": TOLERANCE_PERCENT,
    }

    write_test_result(output_dir, test_id, score, metadata)
    return 0


if __name__ == "__main__":
    sys.exit(main())


"""
Sample command(s):
uv run data/tasks/pdf-hr-q2/test.py --answer "263" --test-id "test" --output-dir /tmp
uv run data/tasks/pdf-hr-q2/test.py --answer "263 employees" --test-id "test" --output-dir /tmp
"""
