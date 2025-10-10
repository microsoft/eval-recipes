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


# Define Semantic Test 1

STEPS_1_CLAUDE_AGENT_SDK_DEPENDENCY = """1. Explore the code that the agent generated in this project directory to see if it uses \
the Claude Agent SDK (claude-agent-sdk for Python, @anthropic-ai/claude-agent-sdk for TypeScript). \
More info is at https://docs.claude.com/en/api/agent-sdk/overview if you need it.
2. Look for where dependencies are defined (e.g., pyproject.toml, requirements.txt, package.json, etc.)
3. Check which dependencies are being imported and actually used in the code to create the email drafting solution."""

RUBRIC_1_CLAUDE_AGENT_SDK_DEPENDENCY = {
    "justification": "str - explanation of if the agent used the correct dependency",
    "score": "float - Score 100 if the agent's solution uses the Claude Agent SDK (claude-agent-sdk). Score 0 otherwise.",
}

# Define Semantic Test 2

STEPS_2_USE_TOOL = """1. Find the README under scenarios/email_drafting/ that the agent should have made to explain how the use the tool and the commands to run.
2. Based on the README, you should come up with the appropriate command to test the tool with the following inputs:
  - Use the set of sample emails available in the `emails` directory 
  - Use the source_notes.txt file that contains the notes for the new email
  - Use "sarah.chen" as the recipient.
3. Run the command to generate an email draft. Depending on how its implemented, this may take up to 10 minutes.
4. After the command finishes, first check if it wrote the email to a file.
5. If it did, did the content of the email match \
  a) the style of the previous emails in the emails dir \
  b) Was it also more aligned to previous emails written to Sara Chen?"""

RUBRIC_2_USE_TOOL = {
    "readme_exists": "str - (5 points) did you find a README file and it contained correct instructions",
    "correct tool inputs": "str - (15 points) did the command have all of the correct inputs: emails dir, notes file, recipient name",
    "tool ran without error": "str - (30 points) did the tool run without error",
    "tool produced file version": "str - (20 points) did the tool produce an email file that could be marked up with [bracket-enclosed-comments]?",
    "email file style": "str - (10 points) did the tool create an email that generally aligns with the style of previous emails?",
    "email recipient style": "str - (20 points) did the tool create an email that generally aligns with the style of previous emails written to Sara Chen?",
    "score": "float - Score between 0 and 100 based on the above criteria.",
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
    """Test script for email_drafting task."""
    return asyncio.run(run_test(test_id, output_dir, instructions_file))


async def run_test(test_id: str, output_dir: Path, instructions_file: Path | None) -> int:
    instructions = get_instructions_from_file_or_default(instructions_file=instructions_file)

    try:
        logger.info(f"Running semantic test 1 to check for Claude Agent SDK dependency...")
        result_1 = await semantic_test(
            steps=STEPS_1_CLAUDE_AGENT_SDK_DEPENDENCY,
            rubric=RUBRIC_1_CLAUDE_AGENT_SDK_DEPENDENCY,
            context=instructions,
            working_dir=Path("/project"),
        )

        logger.info(f"Running semantic test 2 to check if the tool works as intended...")
        result_2 = await semantic_test(
            steps=STEPS_2_USE_TOOL,
            rubric=RUBRIC_2_USE_TOOL,
            context=instructions,
            working_dir=Path("/project"),
        )

        final_score = (result_1.score + result_2.score) / 2
        metadata = {
            "instructions": instructions,
            "semantic_test_1_score": result_1.score,
            "semantic_test_1_metadata": result_1.metadata,
            "semantic_test_2_score": result_2.score,
            "semantic_test_2_metadata": result_2.metadata,
            "final_score": final_score,
        }

        write_test_result(output_dir, test_id, final_score, metadata)
        return 0

    except Exception as e:
        metadata = {
            "instructions": instructions,
            "error": str(e),
        }
        write_test_result(output_dir, test_id, 0, metadata)
        return 0


if __name__ == "__main__":
    sys.exit(main())
