# Copyright (c) Microsoft. All rights reserved.

"""Failure analysis script to run inside a container.

This script is copied into the container and executed there to analyze task failures.
It uses Claude Agent SDK to examine logs and generate a failure report.
"""

import argparse
import asyncio
import inspect
import json
import os
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from liquid import render

from eval_recipes.benchmarking.evaluation.semantic_test import semantic_test

TASK_ANALYSIS_SYSTEM_PROMPT = """You are an expert at analyzing benchmark task failures. \
Your goal is to identify WHY an agent failed at a task.

ABOUT BENCHMARKING:
Benchmarking is a way of evaluating AI agents on real-world tasks within sandboxed environments. \
Each task includes a natural language instruction, test scripts for verification, and runs in isolated containers. \
Tasks range from programming to document creation.
The tasks are executed in two stages: The first, the agent in a fresh container (with proper prerequisites and secrets pre-installed) \
attempts to complete the task. instructions.txt describes the task for the agent, and then agent_output.log is the trace of the agent's actions.
The second stage runs test.py to verify if the task was completed successfully.
You will be provided with test.py which contains the tests that were run to generate a final score. \
The tests can be either deterministic or semantic (in that they use a highly specialized agent to verify correctness).
- test_output.log contains the raw output that the test runner generated. Warning, this file can be VERY long so be smart about reading it.
- test_results.json contains the test results including the final score that was given.
test.py will often call a semantic test function, this is its signature so that you understand how it works:
{{semantic_test_signature}}

IMPORTANT: The files provided can be VERY LONG. You MUST:
1. Read files incrementally (use offset/limit with Read tool)
2. Focus on failure points in test_output.log
3. Trace back through agent_output.log to find what the agent did wrong
4. Cross-reference with instructions.txt and test.py to understand expected behavior
5. You are in the container where the agent ran, so you can further explore what happened as needed.

ANALYSIS APPROACH:
- Start with test_results.json to identify which tests failed and get a rough understanding of why.
- Read test.py to get a high level understanding of what the goals of the tests are.
- Then start to investigate the agent_output.log to find where the agent went wrong.
- Compare agent's approach with instructions.txt and test.py to understand expected behavior
- Identify the root cause: wrong approach, missing steps, incorrect assumptions, bugs in the agent, etc.
- RARELY, the tests themselves may have issues. If you feel strongly that this is the case, call it out.

FAILURE CLASSIFICATION:
After your analysis, you MUST classify the failure into one of these categories:
1. AGENT_ERROR: The agent made a mistake (wrong approach, bugs, incorrect assumptions)
2. INFRASTRUCTURE_ERROR: External failure (Docker issues, API timeouts, network errors, missing dependencies in base image)
3. TEST_ISSUE: The test itself has problems (flaky test, incorrect expectations)

Set VALID_TRIAL to false ONLY for INFRASTRUCTURE_ERROR failures."""

TASK_ANALYSIS_USER_PROMPT = """Now analyze this failed task based on the directive in your system prompt. \
After following that analysis approach, write this output:

OUTPUT: Create TWO files:

1. `FAILURE_REPORT.md` with:
- Executive Summary (2-3 sentences)
- Test Failures (which tests failed, error messages)
- Agent Actions Timeline (key steps agent took)
- Root Cause Analysis (why it failed)
- What Should Have Been Done

2. `failure_metadata.json` with this exact structure:
```json
{
  "classification": "AGENT_ERROR|INFRASTRUCTURE_ERROR|TEST_ISSUE",
  "valid_trial": true|false
}
```

Set valid_trial to false ONLY for INFRASTRUCTURE_ERROR failures.

Make sure that your report is factual and accurate. Do not make assumptions that are not supported by the logs and files provided.
---

<original_task_name>
{{task_name}}
</original_task_name>

<original_instructions>
{{original_instructions}}
</original_instructions>

Now start by making a todo list of steps you will take to analyze the failure and make sure the last todo is to write the report to FAILURE_REPORT.md."""


def load_failure_metadata(metadata_path: Path) -> tuple[str | None, bool]:
    """Load metadata from the failure_metadata.json file.

    Returns:
        Tuple of (failure_category, valid_trial)
    """
    if not metadata_path.exists():
        return None, True

    try:
        content = json.loads(metadata_path.read_text(encoding="utf-8"))
        failure_category = content.get("classification", "").lower() or None
        valid_trial = content.get("valid_trial", True)
        return failure_category, valid_trial
    except (json.JSONDecodeError, OSError):
        return None, True


async def run_analysis(task_name: str, output_path: str) -> None:
    """Run failure analysis inside the container and write results."""
    working_dir = Path("/project")

    # Read instructions
    instructions_path = working_dir / "instructions.txt"
    instructions = instructions_path.read_text(encoding="utf-8") if instructions_path.exists() else ""

    # Get semantic_test signature for context
    semantic_test_signature = inspect.getsource(semantic_test).split("\n\n")[0]

    # Render prompts
    rendered_system_prompt = render(
        TASK_ANALYSIS_SYSTEM_PROMPT,
        semantic_test_signature=semantic_test_signature,
    )

    rendered_user_prompt = render(
        TASK_ANALYSIS_USER_PROMPT,
        task_name=task_name,
        original_instructions=instructions,
    )

    # Configure Claude Agent SDK to work in /project
    options = ClaudeAgentOptions(
        system_prompt=rendered_system_prompt,
        cwd=str(working_dir),
        allowed_tools=["Read", "Grep", "Write", "Bash"],
        max_turns=30,
        permission_mode="default",
        env={"ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", "")},
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query(rendered_user_prompt)
        async for message in client.receive_response():
            # Print messages - they'll be captured in the exec log
            print(message)

        # Check if both output files were created
        report_path = working_dir / "FAILURE_REPORT.md"
        metadata_path = working_dir / "failure_metadata.json"
        if not report_path.exists() or not metadata_path.exists():
            missing = []
            if not report_path.exists():
                missing.append("FAILURE_REPORT.md")
            if not metadata_path.exists():
                missing.append("failure_metadata.json")
            print(f"Missing files: {missing}, asking agent to create them...")
            await client.query(
                f"The following required files were not found: {', '.join(missing)}. "
                "Please create them now with the required format."
            )
            async for message in client.receive_response():
                print(message)

    # Load metadata from JSON file
    metadata_path = working_dir / "failure_metadata.json"
    failure_category, valid_trial = load_failure_metadata(metadata_path)

    result = {
        "failure_category": failure_category,
        "valid_trial": valid_trial,
    }
    Path(output_path).write_text(json.dumps(result), encoding="utf-8")
    print(f"Analysis result written to {output_path}: {result}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run failure analysis inside a container")
    parser.add_argument("--task-name", required=True, help="Name of the task being analyzed")
    parser.add_argument("--output-path", required=True, help="Path to write the analysis result JSON")
    args = parser.parse_args()

    asyncio.run(run_analysis(args.task_name, args.output_path))


if __name__ == "__main__":
    main()
