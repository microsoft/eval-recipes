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

# Define Semantic Test 1: Run Tool and Validate CSV Output Structure

STEPS_1_RUN_AND_VALIDATE_CSV = """1. Find and read the README to understand how to run the CPSC recall tool.
2. Based on the README, install any required dependencies if needed.
3. Determine the correct command to run the tool with a recent month (use "January 2025" or "2025-01" as the test input).
4. Run the tool. This may take several minutes as it needs to fetch data from CPSC.
   - If the tool fails to run or produces errors, note the specific error and assign a score of 0 for tool execution.
5. After the tool completes, locate the output CSV file:
   - Check if a CSV file was created
   - Verify the file is in the expected location (as described in README or instructions)
6. Open and examine the CSV file structure:
   - Does it have headers?
   - Are the required columns present: product name, recall date, manufacturer, hazard description, recall URL?
   - Is the CSV properly formatted (valid CSV syntax)?
   - Can you parse it without errors?
7. Check data completeness:
   - Are there any records in the CSV?
   - Are critical fields populated (not all empty)?
   - Do the recall dates correspond to the requested month?"""

RUBRIC_1_RUN_AND_VALIDATE_CSV = {
    "readme_clear": "str - (5 points) Is the README clear about how to run the tool?",
    "tool_runs_successfully": "str - (25 points) Does the tool run without errors when given a valid month?",
    "csv_file_created": "str - (15 points) Is a CSV file created as output?",
    "csv_has_headers": "str - (10 points) Does the CSV have proper column headers?",
    "required_columns_present": "str - (20 points) Are all required columns present (product name, recall date, manufacturer, hazard, URL)?",
    "csv_properly_formatted": "str - (10 points) Is the CSV properly formatted and parseable?",
    "csv_has_data": "str - (10 points) Does the CSV contain recall records (not empty)?",
    "critical_fields_populated": "str - (5 points) Are critical fields like product name and hazard populated?",
    "score": "float - Score between 0 and 100 based on the above criteria. Sum the points earned from each criterion.",
}


# Define Semantic Test 2: Data Quality and Accuracy Validation

STEPS_2_DATA_QUALITY = """1. Find and read the README to understand how to run the tool.
2. Run the tool with a specific month: "December 2024" (or "2024-12").
3. After the tool completes, locate and open the generated CSV file.
4. Validate data quality and accuracy:
   - Check that recall dates in the CSV are actually from December 2024
   - Verify that URLs are valid CPSC recall URLs (should contain "cpsc.gov" or similar official CPSC domain)
   - Examine manufacturer fields - do they contain actual company names (not empty or generic)?
   - Examine hazard descriptions - are they meaningful descriptions of actual hazards?
   - Check product names - are they specific product descriptions?
5. Test URL validity:
   - Pick 2-3 random URLs from the CSV
   - Make HTTP requests to verify these URLs are actually accessible (return 200/300-level responses)
   - Award full points if the tested URLs work, partial/no points if they fail
6. Test with an alternative month format to verify robustness:
   - Run the tool again with "2024-11" (November 2024)
   - Verify it handles the different format correctly
   - Check that the output CSV contains recalls from November 2024
7. Validate that the tool fetches real CPSC data:
   - For the 2-3 URLs tested above, if accessible, check if the information in the CSV matches what's on the CPSC website"""

RUBRIC_2_DATA_QUALITY = {
    "tool_runs_successfully": "str - (10 points) Does the tool run successfully with the test month?",
    "recall_dates_correct_month": "str - (20 points) Are the recall dates actually from the requested month (December 2024)?",
    "urls_are_valid_cpsc": "str - (10 points) Are the URLs valid CPSC recall URLs (proper format, contain cpsc.gov)?",
    "urls_actually_work": "str - (15 points) When testing 2-3 random URLs with HTTP requests, are they actually accessible?",
    "manufacturers_meaningful": "str - (10 points) Do manufacturer fields contain actual company names?",
    "hazards_meaningful": "str - (10 points) Are hazard descriptions meaningful and specific?",
    "products_specific": "str - (10 points) Are product names specific and descriptive?",
    "alternative_format_works": "str - (10 points) Does the tool work with alternative month format (2024-11)?",
    "data_matches_cpsc_website": "str - (5 points) When spot-checking accessible URLs, does the CSV data match the actual CPSC pages?",
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
    """Test script for cpsc_recall_monitor task."""
    return asyncio.run(run_test(test_id, output_dir, instructions_file))


async def run_test(test_id: str, output_dir: Path, instructions_file: Path | None) -> int:
    instructions = get_instructions_from_file_or_default(instructions_file=instructions_file)

    try:
        logger.info("Running semantic test 1: Running tool and validating CSV output structure...")
        result_1 = await semantic_test(
            steps=STEPS_1_RUN_AND_VALIDATE_CSV,
            rubric=RUBRIC_1_RUN_AND_VALIDATE_CSV,
            context=instructions,
            working_dir=Path("/project"),
        )

        logger.info("Running semantic test 2: Validating data quality and accuracy...")
        result_2 = await semantic_test(
            steps=STEPS_2_DATA_QUALITY,
            rubric=RUBRIC_2_DATA_QUALITY,
            context=instructions,
            working_dir=Path("/project"),
        )

        # Calculate final score with weighted average
        # Weights: CSV output validation (50%), data quality (50%)
        final_score = result_1.score * 0.50 + result_2.score * 0.50

        metadata = {
            "instructions": instructions,
            "semantic_test_1_csv_output": {
                "score": result_1.score,
                "details": result_1.metadata,
            },
            "semantic_test_2_data_quality": {
                "score": result_2.score,
                "details": result_2.metadata,
            },
            "final_score": final_score,
            "scoring_weights": {
                "csv_output_validation": "50%",
                "data_quality": "50%",
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
