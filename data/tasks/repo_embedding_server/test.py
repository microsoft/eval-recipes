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


EXPECTED_OLLAMA_MODEL = "embeddinggemma:300m-qat-q8_0"

TEST_REPO_URL = "https://github.com/microsoft/eval-recipes"



# Define Semantic Test 1: Core Dependencies and Architecture

STEPS_1_DEPENDENCIES_AND_ARCHITECTURE = f"""1. Explore the code in the project directory to understand the implementation.
2. Look for dependency files like pyproject.toml, requirements.txt, or similar.
3. Verify the following dependencies are installed and used:
   - FastAPI for the web server
   - ChromaDB Python library for vector storage
   - Ollama client library (`ollama` in Python)
4. Check the code to confirm:
   - The Ollama model `{EXPECTED_OLLAMA_MODEL}` is specified
   - ChromaDB is used directly as a Python library (not running as a separate service) with either disk-based or in-memory storage
   - The server reads from a file called repos.txt
5. Verify basic server structure exists (FastAPI app, routes/endpoints defined)."""

RUBRIC_1_DEPENDENCIES_AND_ARCHITECTURE = {
    "fastapi_dependency": "str - (20 points) Is FastAPI installed and used for the web server?",
    "chromadb_dependency": "str - (20 points) Is ChromaDB Python library installed and used for vector storage?",
    "ollama_client_dependency": "str - (15 points) Is an Ollama client library installed and used?",
    "correct_ollama_model": f"str - (15 points) Is the model `{EXPECTED_OLLAMA_MODEL}` specified in the code?",
    "chromadb_usage": "str - (15 points) Is ChromaDB used directly as a library (not as a separate service) with disk and/or in-memory storage?",
    "repos_txt_reading": "str - (10 points) Does the code read from a repos.txt file?",
    "server_structure": "str - (5 points) Does a basic FastAPI server structure with endpoints exist?",
    "score": "float - Score between 0 and 100 based on the above criteria. Sum the points earned from each criterion.",
}


# Define Semantic Test 2: Comprehensive Server Functionality

STEPS_2_COMPREHENSIVE_FUNCTIONALITY = f"""1. Find the README or documentation to understand how to start the server and use the endpoints.
2. Ensure Ollama is running (start it with `nohup ollama serve > /dev/null 2>&1 &` if needed, wait 5 seconds, full documentation at https://github.com/ollama/ollama).
3. Create a `repos.txt` file in the working directory with this repository URL: {TEST_REPO_URL}
4. Start the FastAPI server based on the README instructions.
   - Verify the server starts without errors
   - The server should begin indexing. This may take a while depending on the repo size.
5. Test the query endpoint after waiting 3 minutes:
   - Make an HTTP request to query the repository
   - Verify the response includes search results (or empty list if no embeddings yet)
   - Verify the response includes a boolean field indicating indexing status (e.g., `is_indexing_complete` or similar)
   - Verify that embeddings were created for the repository
   - Verify that results are returned even if indexing isn't complete (as long as at least 1 embedding exists)
6. Test the re-index endpoint:
   - Make an HTTP request to trigger manual re-indexing
   - Verify the endpoint responds successfully
7. Test file watching functionality:
   - While the server is still running, add `https://github.com/microsoft/amplifier/` to repos.txt
   - Wait 1-2 minutes for the server to detect the change
   - Query the newly added repository
   - Verify that the server detected the change and started indexing the new repo (check status or logs)
8. Test persistence:
   - Stop the server
   - Start the server again
   - Query the original repository
   - Verify that previously indexed embeddings still exist (persistence to disk worked)
9. Test error handling:
   - Try querying a non-existent repository
   - Verify the server handles the error gracefully (doesn't crash, returns appropriate response)
10. Stop the server after testing."""

RUBRIC_2_COMPREHENSIVE_FUNCTIONALITY = {
    "readme_exists": "str - (5 points) Does a README exist with clear instructions on starting the server?",
    "server_starts_and_reads_repos": "str - (10 points) Does the server start without errors and read repos.txt on startup?",
    "query_endpoint_functionality": "str - (20 points) Does the query endpoint exist, respond, return results (or empty list), and return indexing status?",
    "partial_results_support": "str - (5 points) Does the server return results even when indexing isn't complete (if at least 1 embedding exists)?",
    "reindex_endpoint": "str - (5 points) Does a manual re-index endpoint exist and respond successfully?",
    "indexing_and_embeddings": "str - (20 points) Does the server index the repository and create embeddings (verifiable through query endpoint)?",
    "file_watching": "str - (15 points) Does the server detect changes to repos.txt while running and index newly added repositories?",
    "persistence": "str - (10 points) Are embeddings persisted to disk and available after server restart?",
    "error_handling": "str - (5 points) Does the server handle errors gracefully (e.g., querying non-existent repo)?",
    "stable_operation": "str - (5 points) Does the server run without crashes throughout all testing?",
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
    """Test script for repo_embedding_server task."""
    return asyncio.run(run_test(test_id, output_dir, instructions_file))


async def run_test(test_id: str, output_dir: Path, instructions_file: Path | None) -> int:
    instructions = get_instructions_from_file_or_default(instructions_file=instructions_file)

    try:
        logger.info("Running semantic test 1: Checking dependencies and architecture...")
        result_1 = await semantic_test(
            steps=STEPS_1_DEPENDENCIES_AND_ARCHITECTURE,
            rubric=RUBRIC_1_DEPENDENCIES_AND_ARCHITECTURE,
            context=instructions,
            working_dir=Path("/project"),
        )

        logger.info("Running semantic test 2: Testing comprehensive server functionality...")
        result_2 = await semantic_test(
            steps=STEPS_2_COMPREHENSIVE_FUNCTIONALITY,
            rubric=RUBRIC_2_COMPREHENSIVE_FUNCTIONALITY,
            context=instructions,
            working_dir=Path("/project"),
        )

        # Calculate final score with weighted average
        # Weights: dependencies (25%), comprehensive functionality (75%)
        final_score = (
            result_1.score * 0.25
            + result_2.score * 0.75
        )

        metadata = {
            "instructions": instructions,
            "semantic_test_1_dependencies_and_architecture": {
                "score": result_1.score,
                "details": result_1.metadata,
            },
            "semantic_test_2_comprehensive_functionality": {
                "score": result_2.score,
                "details": result_2.metadata,
            },
            "final_score": final_score,
            "scoring_weights": {
                "dependencies_and_architecture": "25%",
                "comprehensive_functionality": "75%",
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
