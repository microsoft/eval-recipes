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


STEPS_1_RUN_MICROSOFT_2025 = """1. Find and read the README file to understand how to use the tool.
2. Based on the README instructions, determine the command to run the tool with these inputs:
   - Company name: "Microsoft Corporation"
   - Date: April 2025 (or "April" and "2025" depending on how the tool expects input)
3. Run the tool with these inputs. This may take up to 10 minutes as it needs to fetch SEC data.
   - If the tool fails or errors, note the score as 0 and document the error.
4. After the tool completes, verify the following outputs exist:
   - A filing file was saved (check for .pdf, .docx, .txt, .htm, or .html files)
   - A CSV file was created
5. Read the CSV file and extract:
   - Total revenue for the quarter
   - EPS (earnings per share) for the quarter
   - Summary of market risk
6. Validate the extracted data against the ground truth:
   - Total Revenue should be $70.066 billion for the quarter
   - Diluted EPS should be $3.46 or Basic EPS which was $3.47
   - Risk summary should key sections like Commercial, Productivity and Business Processes and Intelligent Cloud , and More Personal Computing, and so on.
7. Check if the filing file downloaded corresponds to the correct 10-Q (Q3 FY2025, ended March 2025). You are provided a pdf version of the file at data/10q_microsoft_apr_2025.pdf
8. Verify the CSV is properly formatted with appropriate headers and readable data."""

RUBRIC_1_RUN_MICROSOFT_2025 = {
    "tool_runs_successfully": "str - (15 points) Does the tool run without critical errors?",
    "filing_file_created": "str - (10 points) Was a filing file saved (pdf, docx, txt, htm, or html)?",
    "csv_file_created": "str - (10 points) Was a CSV file created?",
    "correct_10q_retrieved": "str - (10 points) Does the filing correspond to the correct 10-Q (Q3 FY2025, March, 2025)?",
    "revenue_extracted": "str - (15 points) Is total revenue extracted in the CSV? $70.066 billion for the quarter",
    "eps_extracted": "str - (15 points) Is the correct diluted EPS extracted in the CSV? $3.46 or Basic EPS which was $3.47",
    "risks_extracted": "str - (15 points) Is a risks section factually extracted and summarized in the CSV?",
    "csv_formatting": "str - (10 points) Is the CSV properly formatted with headers and readable data?",
    "score": "float - Score between 0 and 100 based on the above criteria. Sum the points earned from each criterion.",
}

STEPS_2_RUN_MICROSOFT_2017 = """1. Find and read the README file to understand how to use the tool.
2. Based on the README instructions, determine the command to run the tool with these inputs:
   - Company name: "Microsoft Corporation"
   - Date: October 2017 (or "October" and "2017" depending on how the tool expects input)
3. Run the tool with these inputs. This may take up to 10 minutes.
   - If the tool fails or errors, note the score as 0 and document the error.
4. After the tool completes, verify the following outputs exist:
   - A filing file was saved
   - A CSV file was created
5. Read the CSV file and extract:
   - Total revenue for the quarter
   - EPS for the quarter
   - Summary of risk factors
6. Validate the extracted data against the ground truth:
   - Total Revenue should be $24,538 billion
   - Diluted EPS should be $0.85 basic or $0.84 diluted
   - Risk factors summary should include factors/sections directly from the filing like Competition in the technology sector, Business model competition, and so on
7. Check if the filing file downloaded corresponds to the correct 10-Q (Q1 FY2018, ended September 2017). You are provided a pdf version of the file at data/10q_microsoft_oct_2017.htm
8. Verify the tool correctly found the "first 10-Q before October 2017" which would be the Sept 2017 filing."""

RUBRIC_2_RUN_MICROSOFT_2017 = {
    "tool_runs_successfully": "str - (15 points) Does the tool run without critical errors for this second test case?",
    "filing_file_created": "str - (10 points) Was a filing file saved?",
    "csv_file_created": "str - (10 points) Was a CSV file created?",
    "correct_10q_retrieved": "str - (10 points) Does the filing correspond to the correct 10-Q (Q1 FY2018, Sept 2017)?",
    "revenue_extracted": "str - (15 points) Is total revenue extracted in the CSV? Only award  points if value is $24.538B.",
    "eps_extracted": "str - (15 points) Is diluted EPS extracted in the CSV? Only award points if value is $0.85 basic or $0.84 diluted",
    "risks_extracted": "str - (15 points) Is a risks summary extracted in the CSV and is it factual based on the actual 10Q?",
    "csv_formatting": "str - (10 points) Is the CSV properly formatted?",
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
    """Test script for sec_10q_extractor task."""
    return asyncio.run(run_test(test_id, output_dir, instructions_file))


async def run_test(test_id: str, output_dir: Path, instructions_file: Path | None) -> int:
    instructions = get_instructions_from_file_or_default(instructions_file=instructions_file)

    try:
        logger.info("Running semantic test 1: Testing with Microsoft April 2025 data...")
        result_1 = await semantic_test(
            steps=STEPS_1_RUN_MICROSOFT_2025,
            rubric=RUBRIC_1_RUN_MICROSOFT_2025,
            context=instructions,
            working_dir=Path("/project"),
        )

        logger.info("Running semantic test 2: Testing with Microsoft October 2017 data...")
        result_2 = await semantic_test(
            steps=STEPS_2_RUN_MICROSOFT_2017,
            rubric=RUBRIC_2_RUN_MICROSOFT_2017,
            context=instructions,
            working_dir=Path("/project"),
        )

        # Calculate final score with equal weighting
        final_score = (result_1.score + result_2.score) / 2

        metadata = {
            "instructions": instructions,
            "ground_truth_2025": {
                "period": "Q3 FY2025 ended March 31, 2025",
                "total_revenue": "$70,066 million",
                "diluted_eps": "$3.46",
                "market_risk_keywords": ["foreign currency", "interest rate", "credit risk", "equity"],
            },
            "ground_truth_2017": {
                "period": "Q1 FY2018 ended September 30, 2017",
                "total_revenue": "$24,538 million",
                "diluted_eps": "$0.84",
            },
            "semantic_test_1_microsoft_2025": {
                "score": result_1.score,
                "details": result_1.metadata,
            },
            "semantic_test_2_microsoft_2017": {
                "score": result_2.score,
                "details": result_2.metadata,
            },
            "final_score": final_score,
            "scoring_weights": {
                "microsoft_2025_test": "50%",
                "microsoft_2017_test": "50%",
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
