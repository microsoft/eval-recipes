# Copyright (c) Microsoft. All rights reserved.

import json
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from liquid import render
from loguru import logger
from pydantic import BaseModel, field_validator

AUDIT_SYSTEM_PROMPT = """You are performing a quality and compliance audit of another AI agent's deliverables. \
It is of utmost importance that you remain impartial, critical, and objective in your evaluation.
You will be provided a set of steps to take to perform the audit and a rubric to evaluate against.
The agent's work was done within a Docker container, \
so your first goal will be to explore the container according \
to the provided steps and gather the necessary information to complete the audit."""

AUDIT_PROMPT = """The agent was asked to do the following:
{{context}}

You will evaluate the agent's work against the following rubric:
{{rubric}}

You should not include any other fields that are not present in the rubric's schema.

Now take the following steps (make a todo list):
{{steps}}

Do not take any actions that are not related to figuring out how to complete the rubric based on the steps above. \
You can take different steps if as you explore it becomes necessary, but you must be focused on the rubric provided."""


GENERATE_RUBRIC_INSTRUCTIONS_PROMPT = """Now make a structured JSON report that addresses the following rubric:
{{rubric}}

You must place the JSON file at the path ./audit_output/rubric.json so that it can be parsed later. \
Make sure the JSON is valid and can be parsed."""


class SemanticTestResult(BaseModel):
    score: float  # Scores must be between 0 and 100
    metadata: dict[str, Any]  # All other fields from the rubric

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        """Clamp score to 0-100 range."""
        return max(0.0, min(100.0, v))


async def semantic_test(steps: str, rubric: dict[str, Any], context: str, working_dir: Path) -> SemanticTestResult:
    """
    A semantic test uses an agent in a recipe-like way to "audit" the actions of another agent.

    Args:
        steps: Instructions for what steps to take during the audit
        rubric: JSON schema defining the evaluation rubric (must contain a "score" field)
        context: Context information about the agent's work
        working_dir: Working directory for the Claude agent (where it will explore files)

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
    audit_prompt = render(AUDIT_PROMPT, context=context, rubric=rubric_str, steps=steps)
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
            "TodoRead",
            "TodoWrite",
            "WebSearch",
            "Agent",
            "Write",
        ],
        permission_mode="acceptEdits",
        cwd=working_dir,
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
    metadata = rubric_data  # All other fields go into metadata
    return SemanticTestResult(score=score, metadata=metadata)
