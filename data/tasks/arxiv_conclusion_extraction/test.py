# Copyright (c) Microsoft. All rights reserved.

"""
Test script for arxiv paper conclusion extraction task.

Checks if Conclusion.txt contains the exact conclusion from https://arxiv.org/abs/2411.10541.
Scoring: 100 points for 0-5 character differences (buffer), -1 point per additional character difference (using Levenshtein distance).
"""

import sys
from pathlib import Path

import click

from eval_recipes.benchmarking.test_utils import (
    get_instructions_from_file_or_default,
    get_test_id_from_env_or_default,
    levenshtein_distance,
    load_text_from_path_or_content,
    write_test_result,
)

GROUNDTRUTH_CONCLUSION = """Our study reveals that the way prompts are formatted significantly impacts GPT-based models' performance, with no single format excelling universally. This finding questions current evaluation methods that often ignore prompt structure, potentially misjudging a model's true abilities. We advocate for diverse prompt formats in future LLM testing to accurately gauge and enhance their performance.
Regarding explainability, we observe that model size affects model's responses to prompt variations. For instance, GPT-4's performance is less influenced by prompt changes compared to GPT-3.5, suggesting that larger models may process prompts more consistently. This discovery prompts further research into LLM interpretability, aiming to refine AI adaptability and human-AI interaction."""


@click.command()
@click.option(
    "--conclusion",
    type=str,
    default=None,
    help="Path to conclusion text file, or conclusion content as string",
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
def main(conclusion: str | None, test_id: str, output_dir: Path, instructions_file: Path | None) -> int:
    """Test script for arxiv conclusion extraction task."""
    # Use default if not provided
    if conclusion is None:
        conclusion = str(Path(__file__).parents[0] / "Conclusion.txt")

    # Load instructions from file
    instructions = get_instructions_from_file_or_default(instructions_file=instructions_file)

    # Load conclusion text - detect if it's a file path or content
    llm_output = load_text_from_path_or_content(conclusion)

    edit_distance = levenshtein_distance(GROUNDTRUTH_CONCLUSION, llm_output)

    # Calculate score: start at 100, allow 5 character buffer, then subtract 1 for each additional difference
    score = max(0, 100 - max(0, edit_distance - 5))

    print(f"\nGroundtruth length: {len(GROUNDTRUTH_CONCLUSION)} characters")
    print(f"LLM output length: {len(llm_output)} characters")
    print(f"Edit distance: {edit_distance}")
    print(f"Score: {score:.1f}")

    metadata = {
        "instructions": instructions,
        "edit_distance": edit_distance,
        "groundtruth": GROUNDTRUTH_CONCLUSION,
        "llm_output": llm_output,
        "groundtruth_length": len(GROUNDTRUTH_CONCLUSION),
        "llm_output_length": len(llm_output),
    }

    write_test_result(output_dir, test_id, score, metadata)
    return 0


if __name__ == "__main__":
    sys.exit(main())


"""
Sample command(s):
uv run data/tasks/arxiv_conclusion_extraction/test.py --conclusion "Our study reveals that the way prompts are formatted significantly impacts GPT-based models' performance, with no single format excelling universally. This finding questions current evaluation methods that often ignore prompt structure, potentially misjudging a model's true abilities. We advocate for diverse prompt formats in future LLM testing to accurately gauge and enhance their performance.\nRegarding explainability, we observe that model size affects model's responses to prompt variations. For instance, GPT-4's performance is less influenced by prompt changes compared to GPT-3.5, suggesting that larger models may process prompts more consistently. This discovery prompts further research into LLM interpretability, aiming to refine AI adaptability and human-AI interaction." --test-id "claude_test" --output-dir /tmp
"""
