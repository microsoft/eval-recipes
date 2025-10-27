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

# Test data: Use a specific commit from Textualize/rich for deterministic testing
# This repo has:
# - README.md (markdown)
# - docs/ folder with .rst files (needs conversion)
# - Mix of documentation formats

# Define Semantic Test 1: CLI Interface, Basic Execution & LLM-Based Intelligence

STEPS_1_CLI_AND_EXECUTION = """1. Explore the code in the project directory to understand the implementation approach.
2. Check if the solution uses an LLM (Language Learning Model) to intelligently determine which documentation files to extract:
   - Look for API calls to OpenAI, Anthropic, or other LLM providers in the code
   - Check if the code uses an LLM to analyze file contents/names to decide if they're documentation
   - Note: Simple regex/file extension matching is NOT intelligent determination
3. Find and read the README file that explains how to use the tool.
4. Based on the README, determine the correct command to run the tool with these inputs:
   - Repository URL: https://github.com/Textualize/rich
   - Specific commit: 27c2d2df7523315de4b81577d414db8c1c7312f9
   - Note: If the tool doesn't support specifying commits, try using the URL as-is
5. Run the tool. This may take several minutes as it needs to clone/access the repository.
   - If the tool fails with errors, note the error and assign 0 for execution-related points
6. Check if the tool completed successfully:
   - Did it run without fatal errors?
   - Was an output directory created?
   - Does the output directory contain extracted documentation files?
7. Verify basic CLI usability:
   - Were the instructions in the README clear and correct?
   - Did the tool provide helpful output/progress messages?
   - Was the command syntax reasonable and intuitive?"""

RUBRIC_1_CLI_AND_EXECUTION = {
    "uses_llm_for_intelligent_extraction": "str - (30 points) Does the solution use an LLM to intelligently determine which files to extract? Full points for clear LLM usage for file selection. 0 points for simple regex/extension-based matching only.",
    "readme_exists_and_clear": "str - (10 points) Does a README exist with clear usage instructions for running the tool?",
    "cli_accepts_repo_url": "str - (5 points) Does the CLI properly accept a GitHub repository URL as input?",
    "tool_runs_successfully": "str - (30 points) Does the tool run without fatal errors when given a valid GitHub repo?",
    "output_directory_created": "str - (10 points) Is an output directory created with extracted documentation?",
    "helpful_output_messages": "str - (5 points) Does the tool provide clear progress/status messages during execution?",
    "basic_files_extracted": "str - (10 points) Are some documentation files (like README.md) present in the output?",
    "score": "float - Score between 0 and 100 based on the above criteria. Sum the points earned from each criterion.",
}


# Define Semantic Test 2: Documentation Extraction Quality & Format Conversion

STEPS_2_EXTRACTION_QUALITY = """1. First, read the groundtruth file at /project/expected_structure.yaml to understand what files and folders should be in the output.
   - Note there are THREE file lists: expected_files, expected_files_appendix, expected_files_reference
   - Total expected count is 49 files (20 + 2 + 27), and 75% threshold is 37+ files
2. Find and read the README to understand how to use the tool.
3. Run the tool with the Textualize/rich repository:
   - Repository URL: https://github.com/Textualize/rich
   - Specific commit: 27c2d2df7523315de4b81577d414db8c1c7312f9 (if supported)
   - If the tool fails or times out, score 0 for all execution-dependent criteria
4. After the tool completes, find the output directory that was created.
5. Compare the output against ALL THREE file lists from the groundtruth:
   - Check if the expected folders exist (docs/source/appendix, docs/source/reference)
   - Count how many of the expected_files are present (from docs/source/ root)
   - Count how many of the expected_files_appendix are present (from docs/source/appendix/)
   - Count how many of the expected_files_reference are present (from docs/source/reference/)
   - Calculate total: should have at least 37 files out of 49 total (75%+)
6. Count the TOTAL number of documentation files (.md, .mdx, .rst, .txt) extracted in the output:
   - This includes all doc files in all directories
   - Used to check if the tool over-extracted files (see "excessive_extraction_penalty" in rubric)
7. Verify files in subdirectories:
   - Check docs/source/appendix/ contains .md files (converted from box.rst, colors.rst)
   - Check docs/source/reference/ contains .md files (e.g., console.md, text.md, style.md)
   - These subdirectory files should count toward the 75% threshold
8. Verify RST to Markdown conversion quality:
   - Open 2-3 converted files from different locations (root, appendix, reference)
   - Verify they are readable markdown and were converted 1:1 from RST
   - Check conversion quality: headers, lists, code blocks should be properly formatted
9. Verify README.md extraction:
   - Is README.md present in the root of the output?
10. Verify directory structure preservation:
   - Does the output maintain the folder hierarchy (docs/source/ with appendix/ and reference/ subdirs)?
   - Are files in the correct subdirectories?"""

RUBRIC_2_EXTRACTION_QUALITY = {
    "groundtruth_folders_present": "str - (10 points) Are the expected folders from groundtruth present (docs/source/appendix, docs/source/reference)?",
    "groundtruth_total_coverage": "str - (25 points) What percentage of ALL expected files (49 total across all three lists) are present? Full points for 75%+ (37+ files), proportional scoring below that. Must check all three file lists.",
    "excessive_extraction_penalty": "str - (0 to -25 points penalty) Apply penalty if significantly more files than expected are extracted. 0 penalty for <= 55 total doc files. -10 points for 55-70 files. -25 for 70+ files. This ensures the tool is selective, not just grabbing everything.",
    "subdirectory_files_present": "str - (10 points) Are files present in subdirectories (appendix/ and reference/), not just the root docs/source/?",
    "readme_extracted": "str - (15 points) Is README.md present in the output root?",
    "rst_files_converted": "str - (15 points) Are .rst files converted to .md format (not copied as-is)? Check files from multiple locations.",
    "conversion_quality": "str - (15 points) Is the RST to Markdown conversion quality good (readable, proper formatting)? Check 2-3 files from different directories.",
    "directory_structure_preserved": "str - (5 points) Does the output maintain the original docs/source/ folder hierarchy including subdirectories?",
    "score": "float - Score between 0 and 100 based on the above criteria. Sum the points earned from each criterion, then subtract any penalty from excessive_extraction_penalty. Final score cannot go below 0.",
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
    """Test script for github_docs_extractor task."""
    return asyncio.run(run_test(test_id, output_dir, instructions_file))


async def run_test(test_id: str, output_dir: Path, instructions_file: Path | None) -> int:
    instructions = get_instructions_from_file_or_default(instructions_file=instructions_file)

    try:
        logger.info("Running semantic test 1: CLI interface and basic execution...")
        result_1 = await semantic_test(
            steps=STEPS_1_CLI_AND_EXECUTION,
            rubric=RUBRIC_1_CLI_AND_EXECUTION,
            context=instructions,
            working_dir=Path("/project"),
        )

        logger.info("Running semantic test 2: Documentation extraction quality and format conversion...")
        result_2 = await semantic_test(
            steps=STEPS_2_EXTRACTION_QUALITY,
            rubric=RUBRIC_2_EXTRACTION_QUALITY,
            context=instructions,
            working_dir=Path("/project"),
        )

        # Calculate final score with weighted average
        # Weights: CLI and execution (30%), extraction quality (70%)
        final_score = result_1.score * 0.30 + result_2.score * 0.70

        metadata = {
            "instructions": instructions,
            "test_repo_url": "https://github.com/Textualize/rich",
            "test_repo_commit": "27c2d2df7523315de4b81577d414db8c1c7312f9",
            "groundtruth_file": "expected_structure.yaml",
            "semantic_test_1_cli_execution": {
                "score": result_1.score,
                "details": result_1.metadata,
            },
            "semantic_test_2_extraction_quality": {
                "score": result_2.score,
                "details": result_2.metadata,
            },
            "final_score": final_score,
            "scoring_weights": {
                "cli_execution": "30%",
                "extraction_quality": "70%",
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
