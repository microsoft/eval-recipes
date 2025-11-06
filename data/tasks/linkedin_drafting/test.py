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


# Define Semantic Test 1: Dependencies and Architecture

AGENT_SDK_DEFINITION = """The solution should use an Agent SDK, such as Claude Agent/Code SDK, Microsoft Agent Framework, Microsoft Amplifier (https://github.com/microsoft/amplifier/tree/next), OpenAI Codex CLI, or others that are similarly capable. These SDKs must have the following functionality:
- Automatic Context Management to ensure your agent doesn't run out of context.
- Rich tool ecosystem: File operations, code execution, web search, and MCP extensibility
- Excels at code generation and effectively gives the agent a "computer" where it can find appropriate files, write and edit files, lint the code, run it, debug, edit, and sometimes take these actions iteratively until it succeeds.
- APIs like OpenAI's chat completions or Responses API, Anthropic's Messages API, or Azure OpenAI alone are NOT sufficient and should not recieve any credit."""

STEPS_1_DEPENDENCIES = f"""{AGENT_SDK_DEFINITION}

1. Explore the code that was created to understand the implementation.
2. Look for where dependencies are defined (e.g., pyproject.toml, requirements.txt, package.json, etc.)
3. Check if the solution uses an Agent SDK (see definition above):
   - Check which dependencies are listed in dependency files
   - Verify these dependencies are being imported in the code
   - Confirm they are actually used in the implementation (not just imported)
   - Verify the SDK provides the required agent capabilities, not just plain API calls
4. Check if the solution uses an image generation API:
   - Look for OpenAI image generation API usage (DALL-E 3, gpt-image-1, or similar)
   - Check if the code actually calls these APIs to generate images (not just create prompts)
   - Verify imports and actual usage in the code (e.g., openai.images.generate or similar API calls)
   - Confirm the images are actually being generated, not just prompts created
5. Look for evidence of separate stages or agents in the architecture:
   - Check if there are separate components/functions/modules for key workflow parts:
     * Image generation stage
     * Research for references stage
     * Review against existing style stage
   - This could be implemented as:
     * Multiple agents in an agentic loop
     * Separate prompts for different stages
     * Modular functions that handle distinct parts of the workflow
   - The solution should NOT be just one monolithic prompt doing everything"""

RUBRIC_1_DEPENDENCIES = {
    "agent_sdk_identified": "str - Name of Agent SDK found, or 'None'",
    "agent_sdk_usage": "str - (35 points) Does solution use qualifying Agent SDK (Claude Agent/Code SDK, Microsoft Agent Framework, Amplifier, OpenAI Codex CLI)? Must provide automatic context management, rich tool ecosystem, and iterative code capabilities. NOT plain API clients. Check dependency files, imports, and actual usage.",
    "image_generation_api": "str - (35 points) Does the solution use OpenAI's image generation API (DALL-E 3, gpt-image-1, or similar)? Check dependencies and actual API calls in code. Verify images are actually generated, not just prompts.",
    "separate_stages_or_agents": "str - (30 points) Does the solution have evidence of separate stages/agents for key workflow parts (image generation, research for references, review against existing style)? Could be agentic loops, separate prompts, or modular functions. Not just one monolithic prompt.",
    "score": "float - Score between 0 and 100 based on the above criteria. Sum the points earned from each criterion.",
}


# Define Semantic Test 2: Run Tool, Validate Outputs, and Assess Quality

STEPS_2_RUN_AND_VALIDATE = """1. Find the README or documentation that explains how to use the tool.
2. Locate the test data in the project:
   - past_posts/ directory with 3 sample posts
   - topic_notes.txt with notes about databases and controlled burns
3. Based on the README, determine the correct command to run the tool with these inputs.
4. Run the tool with the test data. This may take up to 15 minutes as it involves AI generation.
   - If the tool fails to complete after 15 minutes or errors out, note an overall score of 0 for this test!
5. After the tool completes, verify the CLI interface:
   - Does it accept past posts directory as input?
   - Does it accept topic notes as input?
   - Does it run without errors?
   - Does it have an option to output a social media version?
6. Check the outputs:
   - Is a post generated?
   - Are images or image prompts generated? Are the images embedded as Markdown links in the post?
   - Is there a social media version?
7. Analyze the writing style by reading all 3 sample posts carefully:
   - **MOST DISTINCTIVE CHARACTERISTIC**: Creative metaphors from completely unexpected domains
     * Post 1: Cicadas (biology) → Caching strategies (engineering)
     * Post 2: Medieval walled cities (history) → Microservices architecture (engineering)
     * Post 3: Jazz improvisation (music) → Debugging techniques (engineering)
   - Other style elements:
     * Short paragraphs with clear structure
8. Read the generated post and evaluate quality:
   - **CRITICAL**: Does it use a creative metaphor from an unexpected domain (not software/tech)?
     * The metaphor should come from biology, history, music, sports, nature, etc.
     * It should be used throughout the post to explain the technical concept
     * This is the signature characteristic - without this, it's not matching the style
   - Does it use short paragraphs with clear structure?"""

# SCORING GUIDANCE: The most distinctive characteristic of this writing style is the use of creative metaphors
# from completely unexpected domains (biology, history, music, etc.) to explain technical concepts.
# A post without this metaphor characteristic should receive low scores even if other elements are present.

RUBRIC_2_RUN_AND_VALIDATE = {
    "readme_exists": "str - (5 points) Does clear documentation exist explaining how to run the tool?",
    "accepts_past_posts": "str - (5 points) Does the CLI accept past posts as input?",
    "accepts_topic_notes": "str - (5 points) Does the CLI accept topic notes as input?",
    "tool_runs_successfully": "str - (20 points) Does the tool run without errors?",
    "generates_post": "str - (25 points) Does it generate a post?",
    "has_social_media_option": "str - (10 points) Does the tool have an option to output a social media version?",
    "uses_unexpected_metaphor": "str - (25 points) CRITICAL - Does the post use a creative metaphor from an unexpected domain (biology, history, music, sports, nature, etc.) to explain a technical concept? This is the signature characteristic of the style.",
    "short_paragraphs_structure": "str - (5 points) Does it use short paragraphs with clear structure?",
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
    """Test script for linkedin_drafting task."""
    return asyncio.run(run_test(test_id, output_dir, instructions_file))


async def run_test(test_id: str, output_dir: Path, instructions_file: Path | None) -> int:
    instructions = get_instructions_from_file_or_default(instructions_file=instructions_file)

    try:
        logger.info("Running semantic test 1: Checking dependencies and architecture...")
        result_1 = await semantic_test(
            steps=STEPS_1_DEPENDENCIES,
            rubric=RUBRIC_1_DEPENDENCIES,
            context=instructions,
            working_dir=Path("/project"),
        )

        logger.info("Running semantic test 2: Running tool, validating outputs, and assessing quality...")
        result_2 = await semantic_test(
            steps=STEPS_2_RUN_AND_VALIDATE,
            rubric=RUBRIC_2_RUN_AND_VALIDATE,
            context=instructions,
            working_dir=Path("/project"),
        )

        # Calculate final score with weighted average
        # Weights: dependencies and architecture (30%), run/validate/quality (70%)
        final_score = (
            result_1.score * 0.30
            + result_2.score * 0.70
        )

        metadata = {
            "instructions": instructions,
            "semantic_test_1_dependencies_and_architecture": {
                "score": result_1.score,
                "details": result_1.metadata,
            },
            "semantic_test_2_run_validate_quality": {
                "score": result_2.score,
                "details": result_2.metadata,
            },
            "final_score": final_score,
            "scoring_weights": {
                "dependencies_and_architecture": "30%",
                "run_validate_quality": "70%",
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
