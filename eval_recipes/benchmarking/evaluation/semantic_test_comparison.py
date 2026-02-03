# Copyright (c) Microsoft. All rights reserved.

import json
import os
from pathlib import Path
import secrets
import shutil
import tempfile

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from dotenv import load_dotenv
from liquid import render

from eval_recipes.benchmarking.schemas import ComparisonResult

load_dotenv()


def _write_to_log(log_file: Path | None, message: str) -> None:
    """Write a message to the log file if provided."""
    if log_file:
        with log_file.open("a", encoding="utf-8") as f:
            f.write(message + "\n")


COMPARISON_SYSTEM_PROMPT = """You are performing a comparative evaluation of multiple AI agents' work on the same task.
It is of utmost importance that you remain impartial, critical, and objective in your evaluation.
Each agent's work is in a separate directory with a randomized name to ensure blind comparison.
Your goal is to explore each agent's output and determine how to rank the agents' work.

For effective comparisons:
- Spawn specialized sub-agents to explore each agent's output individually
- If an agent built an app, figure out how to build and run it to evaluate functionality
- Focus on evaluating the final deliverables based on the original task instructions. \
The other files available might be context or intermediate work products. \
You should only explore them as necessary to effectively compare the two agent's final outputs.
- You MUST be critical. There is always room for improvement.
- Do not consider aspects that don't matter, like the file sizes.
- Your final answer will be a JSON file with your reasoning and rankings. Your reasoning should be in bullet point format.
- For anything visual like a ppt or application, figure out how to view it as images so you can evaluate it on that dimension as well.

RULES:
- You must **NEVER** under ANY circumstances change the code or files that were created by the agent. \
You must use its code and outputs as is, changing its output is akin to a teacher changing a student's exam answers.
- Your goal is NOT to troubleshoot or debug the agent's work, but to evaluate it as is. \
If it is not working after following the steps and instructions that the agent may have created. Move on, and evaluate it as is. \
- You should not need to get any API keys - they are provided to you as env vars already. \
However, you can install dependencies based on the instructions if needed. \
If after following the instructions whatever you are testing is not working, move on and evaluate as is. DO NOT try to fix it.
- Never try to read in large PDFs or binary files directly. Write code to parse them into text instead.
- If the tool times out or does not complete in the time stated by either the instructions or the agent's own comments - that is a failure.\
Do not keep trying to run or fix things.
- There may be remnants of created files and build artifacts from when the agent previous ran or was tested. \
These file outputs should NOT be considered as part of your evaluation - make sure to validate based on what the agent did during **your** current audit only.
- These rules are ABSOLUTE and NON-NEGOTIABLE."""

COMPARISON_PROMPT = """Each agent was asked to complete the following task:
{{original_task}}

The agents' outputs are available in the following directories:
{% for dir in directories %}
- {{dir}}
{% endfor %}

There may be other files present in this workspace that were not created by the agents, but were already there such as the user's files.

{% if guidelines %}
## Evaluation Guidelines
{{guidelines}}
{% endif %}

1. First explore what each agent produced.
2. Analyze the relative strengths and weaknesses of each agent's outcome.
3. State your reasoning and then create a relative ranking of the agents from best to worst based on their outputs. \
In the NEXT step you will write a JSON file with your final reasoning and rankings.

IMPORTANT: Under all circumstances, you must follow the rules defined in your system prompt."""

GENERATE_COMPARISON_RESULT_PROMPT = """Now write your comparison analysis.

First, write reasoning in digestible, but factual and accurate bullet points explaining:
1. What each agent produced
2. The strengths and weaknesses of each agent's approach
3. Your ranking and an explanation of why.

Keep it specific, concise, and in bullet point format. It should be no more than 10 bullet points of 1-2 sentences each. \
Whenever you mention an agent, be sure to refer to it by its directory name exactly as it appears (e.g., agent_ab12).

The reasoning and your final ranking should go in a JSON file at ./comparison_output/result.json with this exact schema:
{
    "reasoning": "Your specific and concise reasoning here",
    "rankings": ["agent_xxx", "agent_yyy", ...]  // Directory names ordered best to worst
}

Make sure the JSON is valid and can be parsed."""

COMPARISON_JSON_FIX_PROMPT = """The JSON file at ./comparison_output/result.json could not be parsed.
Error: {{error}}

Please fix the JSON file and ensure it is valid. The schema must be:
{
    "reasoning": "string",
    "rankings": ["agent_xxx", "agent_yyy", ...]
}"""


async def semantic_test_comparison(
    original_task: str,
    directories: list[Path],
    guidelines: str | None = None,
    log_file: Path | None = None,
) -> ComparisonResult:
    """
    Compare multiple agents' work on the same task.

    Args:
        original_task: The original task/instructions given to all agents
        directories: List of paths where each agent's /project/ folder contents live.
                     Each directory contains one agent's complete output.
                     Directories will be randomly renamed internally for blind comparison.
        guidelines: Optional task-specific evaluation guidelines (e.g., how to evaluate a PPT)
        log_file: Optional path to write Claude SDK messages to (instead of logger.info)

    Returns:
        ComparisonResult with reasoning and rankings (mapped back to original indices)

    Raises:
        ValueError: If fewer than 2 directories provided or if result parsing fails
    """
    if len(directories) < 2:
        raise ValueError("At least 2 directories are required for comparison")

    result: ComparisonResult | None = None

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text("", encoding="utf-8")

    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = Path(tmp_dir)
        name_to_index = _create_anonymous_workspace(directories, workspace)
        anon_dir_names = list(name_to_index.keys())

        comparison_prompt = render(
            COMPARISON_PROMPT, original_task=original_task, directories=anon_dir_names, guidelines=guidelines
        )

        options = ClaudeAgentOptions(
            system_prompt=COMPARISON_SYSTEM_PROMPT,
            allowed_tools=[
                "Read",
                "Glob",
                "Grep",
                "Bash",
                "Write",
                "WebFetch",
                "WebSearch",
                "TodoRead",
                "TodoWrite",
                "Agent",
            ],
            permission_mode="acceptEdits",
            cwd=workspace,
            env={"ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", "")},
        )

        async with ClaudeSDKClient(options=options) as client:
            # Step 1: Send comparison prompt
            await client.query(comparison_prompt)
            async for message in client.receive_response():
                _write_to_log(log_file, str(message))

            # Step 2: Send generate result prompt
            await client.query(GENERATE_COMPARISON_RESULT_PROMPT)
            async for message in client.receive_response():
                _write_to_log(log_file, str(message))

            # Step 3: Parse result with retry
            result_file = workspace / "comparison_output" / "result.json"
            result = await _parse_comparison_result_with_retry(client, result_file, name_to_index, log_file)

    if result is None:
        raise RuntimeError("Comparison result was not set")
    return result


def _create_anonymous_workspace(directories: list[Path], workspace: Path) -> dict[str, int]:
    """
    Copy agent directories into workspace with randomized names for blind comparison.

    Args:
        directories: List of paths to agent directories
        workspace: Directory where copies will be created

    Returns:
        Mapping of anonymous name -> original index
    """
    name_to_index: dict[str, int] = {}
    for i, directory in enumerate(directories):
        while True:
            random_suffix = secrets.token_hex(2)
            anon_name = f"agent_{random_suffix}"
            if anon_name not in name_to_index:
                break
        dest_path = workspace / anon_name
        shutil.copytree(directory, dest_path)
        name_to_index[anon_name] = i
    return name_to_index


async def _parse_comparison_result_with_retry(
    client: ClaudeSDKClient,
    result_file: Path,
    name_to_index: dict[str, int],
    log_file: Path | None = None,
    max_retries: int = 3,
) -> ComparisonResult:
    """
    Parse JSON result, asking agent to fix if invalid.

    Args:
        client: Active ClaudeSDKClient session
        result_file: Path to the JSON result file
        name_to_index: Mapping of anonymous names to original indices
        log_file: Optional path to write Claude SDK messages to
        max_retries: Maximum number of retry attempts
    """
    for attempt in range(max_retries):
        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
            rankings = [name_to_index[name] for name in data["rankings"]]
            return ComparisonResult(
                reasoning=data["reasoning"],
                rankings=rankings,
                anonymous_to_index=name_to_index,
            )
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            if attempt < max_retries - 1:
                fix_prompt = render(COMPARISON_JSON_FIX_PROMPT, error=str(e))
                await client.query(fix_prompt)
                async for message in client.receive_response():
                    _write_to_log(log_file, str(message))
            else:
                raise ValueError(f"Failed to parse comparison result after {max_retries} attempts: {e}") from e
    raise ValueError("Unexpected exit from retry loop")
