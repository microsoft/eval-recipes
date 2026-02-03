# Copyright (c) Microsoft. All rights reserved.

import json
import os
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from dotenv import load_dotenv
from liquid import render
from loguru import logger

from eval_recipes.benchmarking.schemas import SemanticTestResult

load_dotenv()

AUDIT_SYSTEM_PROMPT = """You are performing a quality and compliance audit of another AI agent's deliverables. \
It is of utmost importance that you remain impartial, critical, and objective in your evaluation.
You will be provided a set of steps to take to perform the audit and a rubric to evaluate against.
The agent's work was done within a Docker container, \
so your first goal will be to explore the container according \
to the provided steps and gather the necessary information to complete the audit.

CRITICAL: You must be aware of large files, particularly PDFs. \
When trying to read a PDF, do not read the file directly. Write code to parse it into text instead."""

AUDIT_PROMPT = """The agent was asked to do the following:
{{context}}

You will evaluate the agent's work against the following rubric:
{{rubric}}

You should not include any other fields that are not present in the rubric's schema.
{% if agent_log_hint %}

AGENT LOG LOCATION: If your evaluation requires examining the agent's raw logs or session data, they may be available at: {{agent_log_hint}}
{% endif %}

Now take the following steps (make a todo list):
{{steps}}

Do not take any actions that are not related to figuring out how to complete the rubric based on the steps above. \
You can take different steps if as you explore it becomes necessary, but you must be focused on the rubric provided.

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


GENERATE_RUBRIC_INSTRUCTIONS_PROMPT = """Now make a structured JSON report that addresses the following rubric:
{{rubric}}

You must place the JSON file at the path ./audit_output/rubric.json so that it can be parsed later. \
Make sure the JSON is valid and can be parsed.
IMPORTANT: Under all circumstances, you must follow the rules defined in your system prompt."""


async def semantic_test(
    steps: str,
    rubric: dict[str, Any],
    context: str,
    working_dir: Path,
    agent_log_hint: str | None = None,
) -> SemanticTestResult:
    """
    A semantic test uses an agent in a recipe-like way to "audit" the actions of another agent.

    Args:
        steps: Instructions for what steps to take during the audit
        rubric: JSON schema defining the evaluation rubric (must contain a "score" field)
        context: Context information about the agent's work
        working_dir: Working directory for the Claude agent (where it will explore files)
        agent_log_hint: Optional hint about where the agent stores its logs in the container

    Returns:
        SemanticTestResult with score and metadata from the rubric evaluation

    Raises:
        ValueError: If rubric schema does not contain a "score" field
        FileNotFoundError: If rubric file is not created at the expected path
    """
    if "score" not in rubric:
        raise ValueError("Rubric schema must contain a 'score' field")

    # Convert rubric dict to formatted string for prompts
    rubric_str = json.dumps(rubric, indent=2)
    audit_prompt = render(AUDIT_PROMPT, context=context, rubric=rubric_str, steps=steps, agent_log_hint=agent_log_hint)
    generate_prompt = render(GENERATE_RUBRIC_INSTRUCTIONS_PROMPT, rubric=rubric_str)

    options = ClaudeAgentOptions(
        system_prompt=AUDIT_SYSTEM_PROMPT,
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
        cwd=working_dir,
        env={"ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", "")},
    )

    async with ClaudeSDKClient(options=options) as client:
        # Step 1: Send audit prompt
        await client.query(audit_prompt)
        async for message in client.receive_response():
            logger.info(str(message))

        # Step 2: Send generate rubric prompt
        await client.query(generate_prompt)
        async for message in client.receive_response():
            logger.info(str(message))

    # Read the generated rubric file (relative to working directory)
    rubric_file = working_dir / "audit_output" / "rubric.json"
    if not rubric_file.exists():
        raise FileNotFoundError(f"Rubric file not found at {rubric_file}")

    rubric_data = json.loads(rubric_file.read_text())
    score = float(rubric_data.pop("score"))
    return SemanticTestResult(score=score, metadata=rubric_data)
