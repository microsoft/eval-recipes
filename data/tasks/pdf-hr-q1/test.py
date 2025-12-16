# Copyright (c) Microsoft. All rights reserved.

"""
Test script for PDF HR Q1 data analysis task.

Checks if the answer in data_analysis_answer.txt is within 1% of the true value (MX$ 1,531,989.62).
Scoring: 100 points if within 1% tolerance, 0 otherwise.
"""

import re
import sys
from pathlib import Path

import click

from eval_recipes.benchmarking.test_utils import (
    get_instructions_from_file_or_default,
    get_test_id_from_env_or_default,
    load_text_from_path_or_content,
    write_test_result,
)

GROUNDTRUTH_VALUE = 1531989.62
TOLERANCE_PERCENT = 1.0


def parse_numeric_value(text: str) -> float | None:
    """
    Parse a numeric value from text, handling various formats.

    Handles formats like:
    - "MX$ 1,531,989.62"
    - "1,531,989.62"
    - "1531989.62"
    - "$1,531,989.62"
    - "1531989"

    Returns None if no valid number is found.
    """
    # Remove currency symbols and whitespace
    cleaned = re.sub(r"[MX$\s]", "", text)
    # Remove commas (thousand separators)
    cleaned = cleaned.replace(",", "")

    # Try to find a floating point or integer number
    match = re.search(r"-?\d+\.?\d*", cleaned)
    if match:
        try:
            return float(match.group())
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
    """Test script for PDF HR Q1 data analysis task."""
    # Use default if not provided
    if answer is None:
        answer = str(Path(__file__).parents[0] / "data_analysis_answer.txt")

    # Load instructions from file
    instructions = get_instructions_from_file_or_default(instructions_file=instructions_file)

    # Load answer text - detect if it's a file path or content
    llm_output = load_text_from_path_or_content(answer)

    # Parse the numeric value from the answer
    parsed_value = parse_numeric_value(llm_output)

    if parsed_value is None:
        print(f"\nCould not parse numeric value from: {llm_output!r}")
        percent_difference = None
        score = 0
    else:
        # Calculate percent difference
        percent_difference = abs(parsed_value - GROUNDTRUTH_VALUE) / GROUNDTRUTH_VALUE * 100

        # Binary scoring: 100 if within tolerance, 0 otherwise
        score = 100 if percent_difference <= TOLERANCE_PERCENT else 0

        print(f"\nGroundtruth value: {GROUNDTRUTH_VALUE:,.2f}")
        print(f"Parsed value: {parsed_value:,.2f}")
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
uv run data/tasks/pdf-hr-q1/test.py --answer "MX$ 1,531,989.62" --test-id "test" --output-dir /tmp
uv run data/tasks/pdf-hr-q1/test.py --answer "1531989.62" --test-id "test" --output-dir /tmp
"""
