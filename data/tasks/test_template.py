# Copyright (c) Microsoft. All rights reserved.

"""
TEMPLATE: Test script for benchmark tasks.

This template demonstrates the contract that all test.py files must follow.
Copy this file when creating new benchmark tasks and modify the task-specific logic.

CONTRACT FOR TEST SCRIPTS:

1. REQUIRED CLI OPTIONS:
   - --test-id: Test identifier (defaults to EVAL_RECIPES_TEST_ID env var, fallback to "dev")
   - --output-dir: Directory for result file (defaults to script's parent directory)
   - --instructions-file: Path to instructions file (defaults to ./instructions.txt in working directory)
     The harness writes instructions to this file to avoid environment variable size limits

2. TASK-SPECIFIC OPTIONS:
   - Add any task-specific inputs (e.g., --csv, --summary, --conclusion)
   - Support both file paths AND direct content strings for flexibility
   - Use load_text_from_path_or_content() helper for this pattern

3. RESULT FILE:
   - Must write: .eval_recipes_test_results_{test_id}.json
   - Must contain: {"score": float (0-100), "metadata": dict}
   - Use write_test_result() from test_utils for consistency

4. EXIT CODE:
   - Return 0 if test completes successfully (regardless of score)
   - Return non-zero only for script failures/exceptions

5. EXECUTION MODES:
   - Container mode: Instructions read from ./instructions.txt (written by harness)
   - Local/debug mode: Run with CLI args for custom paths

EXAMPLE USAGE:
=============

# Container execution (typical harness usage):
$ EVAL_RECIPES_TEST_ID=abc123 uv run test.py
# Instructions are read from ./instructions.txt which the harness writes

# Local debugging with instructions file:
$ uv run test.py --test-id dev --instructions-file my_instructions.txt --custom-input input.txt

# Local debugging with default instructions.txt:
$ uv run test.py --test-id dev --custom-input "Some content here"
# Reads from ./instructions.txt if it exists
"""

import sys
from pathlib import Path

import click

from eval_recipes.benchmarking.test_utils import (
    get_instructions_from_file_or_default,
    get_test_id_from_env_or_default,
    load_text_from_path_or_content,
    write_test_result,
)


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
    help="Directory to write result file (defaults to script directory)",
)
@click.option(
    "--instructions-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to instructions file (defaults to ./instructions.txt in working directory)",
)
@click.option(
    "--custom-input",
    type=str,
    default=None,
    help="Example task-specific input: path to file or direct content string",
)
def main(
    test_id: str,
    output_dir: Path,
    instructions_file: Path | None,
    custom_input: str | None,
) -> int:
    """
    Test script template demonstrating the formal benchmark test contract.

    This template shows all required elements and best practices for creating
    benchmark test scripts.
    """
    # Initialize metadata dict to store test details
    metadata: dict = {}

    # Load instructions from file (harness writes to ./instructions.txt in container)
    # Instructions contain what the agent was asked to do
    instructions = get_instructions_from_file_or_default(instructions_file=instructions_file)
    metadata["instructions"] = instructions

    # Example: Load custom input using the flexible helper
    # This supports both file paths and direct content strings
    if custom_input is not None:
        custom_content = load_text_from_path_or_content(custom_input)
        metadata["custom_input_length"] = len(custom_content)

        # TODO: Add your task-specific validation logic here
        # Example: Check if the agent's output meets requirements
        # score = calculate_score(custom_content, instructions)

    # Example scoring (replace with actual logic)
    score = 100.0
    metadata["example_check"] = "passed"

    # Write result using standardized utility
    # This ensures consistent format and score validation
    write_test_result(output_dir, test_id, score, metadata)

    print(f"Test completed with score: {score:.1f}/100")
    print(f"Result written to: {output_dir / f'.eval_recipes_test_results_{test_id}.json'}")

    # Return 0 to indicate successful test execution
    # The score in the result file is what matters for evaluation
    return 0


if __name__ == "__main__":
    sys.exit(main())


"""
Sample command(s):
# Local debugging - with custom instructions file:
uv run test_template.py --test-id dev --instructions-file my_instructions.txt --custom-input output.txt
"""
