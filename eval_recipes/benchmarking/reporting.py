import contextlib
import inspect
import os
from pathlib import Path
import shutil
import tempfile

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from dotenv import load_dotenv
from liquid import render
from loguru import logger

from eval_recipes.benchmarking.semantic_test import semantic_test

load_dotenv()

TASK_REPORT_SYSTEM_PROMPT = """You are an expert at analyzing benchmark task failures. Your goal is to identify WHY an agent failed at a task.

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
- trial_result.json contains the trial results including the final score that was given, along with timing information.
test.py will often call a semantic test function, this is its signature so that you understand how it works:
{{semantic_test_signature}}

IMPORTANT: The files provided can be VERY LONG. You MUST:
1. Read files incrementally (use offset/limit with Read tool)
2. Focus on failure points in test_output.log
3. Trace back through agent_output.log to find what the agent did wrong
4. Cross-reference with instructions.txt and test.py to understand expected behavior

ANALYSIS APPROACH:
- Start with trial_result.json to identify which tests failed and get a rough understanding of why.
- Read test.py to get a high level understanding of what the goals of the tests are.
- Then start to investigate the agent_output.log to find where the agent went wrong.
- Compare agent's approach with instructions.txt and test.py to understand expected behavior
- Identify the root cause: wrong approach, missing steps, incorrect assumptions, bugs in the agent, etc.
- RARELY, the tests themselves may have issues. If you feel strongly that this is the case, call it out."""

TASK_REPORT_USER_PROMPT = """Now analyze this failed task based on the directive in your system prompt. After following that analysis approach, write this output:

OUTPUT: Create a detailed report in `FAILURE_REPORT.md` with:
- Executive Summary (2-3 sentences)
- Test Failures (which tests failed, error messages)
- Agent Actions Timeline (key steps agent took)
- Root Cause Analysis (why it failed)
- What Should Have Been Done

Make sure that your report is factual and accurate. Do not make assumptions that are not supported by the logs and files provided.
---

<original_task_name>
{{task_name}}
</original_task_name>

<original_instructions>
{{original_instructions}}
</original_instructions>

Now start by making a todo list of steps you will take to analyze the failure and make sure the last todo is to write the report to FAILURE_REPORT.md."""


async def generate_trial_report(
    trial_dir: Path,
    task_directory: Path,
    trial_score: float,
    trial_number: int,
    report_score_threshold: float = 85.0,
) -> bool:
    """
    Generate a failure report for a single trial.

    Args:
        trial_dir: Path to the trial directory (e.g., runs/agent_task/trial_1/)
        task_directory: Path to the task definition directory containing instructions.txt
        trial_score: Score achieved by this trial
        trial_number: Trial number
        report_score_threshold: Skip if score >= threshold (default: 85.0)

    Returns:
        True if report was generated, False if skipped

    Creates:
        FAILURE_REPORT_trial_N.md in the trial directory
    """
    if trial_score >= report_score_threshold:
        logger.info(f"Skipping report for trial {trial_number} (score {trial_score:.1f} >= {report_score_threshold})")
        return False

    if not trial_dir.exists():
        logger.warning(f"Trial directory not found: {trial_dir}")
        return False

    task_name = task_directory.name

    # Read task instructions
    instructions_file = task_directory / "instructions.txt"
    if not instructions_file.exists():
        raise FileNotFoundError(f"Instructions file not found: {instructions_file}")
    task_instructions = instructions_file.read_text(encoding="utf-8")

    logger.info(f"Generating failure report for trial {trial_number} (score: {trial_score:.1f})")

    # Get semantic_test signature dynamically using inspect
    semantic_test_signature = inspect.getsource(semantic_test).split("\n\n")[0]

    # Render system prompt
    rendered_system_prompt = render(
        TASK_REPORT_SYSTEM_PROMPT,
        semantic_test_signature=semantic_test_signature,
    )

    # Create a temp dir to place files for Claude Agent SDK to work
    temp_dir = Path(tempfile.mkdtemp(prefix=f"benchmark_report_{task_name}_trial{trial_number}_"))

    try:
        # Collect files from trial directory
        files_to_copy = {
            "agent_output.log": trial_dir / "agent_output.log",
            "test_output.log": trial_dir / "test_output.log",
            "trial_result.json": trial_dir / "trial_result.json",
            "test.py": trial_dir / "test.py",
            "instructions.txt": task_directory / "instructions.txt",
        }

        # Copy available files to temp directory
        for dest_name, source_path in files_to_copy.items():
            if source_path.exists():
                dest_path = temp_dir / dest_name
                shutil.copy2(source_path, dest_path)

        # Create AGENTS.md with @reference to eval_recipes documentation
        agents_md_content = """# Evaluation Recipes Documentation

@eval_recipes_readme.md"""
        agents_md_path = temp_dir / "AGENTS.md"
        agents_md_path.write_text(agents_md_content, encoding="utf-8")

        # Copy LOW_LEVEL_API.md as eval_recipes_readme.md
        low_level_api_path = Path(__file__).parents[2] / "docs" / "LOW_LEVEL_API.md"
        if low_level_api_path.exists():
            eval_recipes_readme_path = temp_dir / "eval_recipes_readme.md"
            eval_recipes_readme_path.write_text(low_level_api_path.read_text(encoding="utf-8"), encoding="utf-8")

        # Update prompt to include trial context
        task_report_prompt = render(
            TASK_REPORT_USER_PROMPT,
            task_name=f"{task_name} (Trial {trial_number})",
            original_instructions=task_instructions,
        )

        options = ClaudeAgentOptions(
            system_prompt=rendered_system_prompt,
            cwd=str(temp_dir),
            allowed_tools=["Read", "Grep", "Write"],
            max_turns=30,
            permission_mode="default",
            env={"ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", "")},
        )
        messages = []  # store messages for debugging purposes only
        async with ClaudeSDKClient(options=options) as client:
            await client.query(task_report_prompt)
            async for _message in client.receive_response():
                messages.append(_message)

            # Check if the report was created at the expected path
            report_path = temp_dir / "FAILURE_REPORT.md"
            if not report_path.exists():
                logger.warning(
                    f"Report not found at expected path: {report_path}. Asking agent to verify the location..."
                )
                await client.query(
                    f"The report was not found at {report_path}. "
                    "Can you please double check if it is in the correct location and has the correct file name? "
                    "The file should be named 'FAILURE_REPORT.md' and placed in the current working directory."
                )
                async for _message in client.receive_response():
                    messages.append(_message)

        # Get the report from the temp dir and move it to the trial directory
        report_path = temp_dir / "FAILURE_REPORT.md"
        if report_path.exists():
            generated_content = report_path.read_text(encoding="utf-8")
            # Prepend metadata header
            metadata_header = f"""---
**Task**: {task_name}
**Trial**: {trial_number}
**Score**: {trial_score:.1f}%
---

"""
            final_content = metadata_header + generated_content

            output_path = trial_dir / f"FAILURE_REPORT_trial_{trial_number}.md"
            output_path.write_text(final_content, encoding="utf-8")
            logger.info(f"Failure report for trial {trial_number} saved to: {output_path}")
            return True

        logger.warning(f"Failed to generate report for trial {trial_number}")
        return False
    finally:
        if temp_dir.exists():
            # Best effort cleanup
            with contextlib.suppress(Exception):
                shutil.rmtree(temp_dir)


async def generate_agent_consolidated_report(
    agent_name: str,
    runs_dir: Path,
    task_names: list[str],
    num_trials: int,
) -> bool:
    """
    Generate a consolidated report for a single agent synthesizing all failure reports.

    Args:
        agent_name: Name of the agent
        runs_dir: Base directory containing all run outputs
        task_names: List of task names this agent ran
        num_trials: Number of trials per task

    Returns:
        True if report was generated, False if no failure reports found
    """
    # Collect failure reports from filesystem
    failure_reports = []
    for task_name in task_names:
        base_run_dir = runs_dir / f"{agent_name}_{task_name}"
        for trial_num in range(1, num_trials + 1):
            report_path = base_run_dir / f"trial_{trial_num}" / f"FAILURE_REPORT_trial_{trial_num}.md"
            if report_path.exists():
                content = report_path.read_text(encoding="utf-8")
                report_identifier = f"{agent_name}_{task_name}_trial_{trial_num}"
                failure_reports.append(f"## Report: {report_identifier}\n\n{content}\n\n{'=' * 80}\n")

    if not failure_reports:
        logger.info(f"No failure reports found for agent '{agent_name}' - skipping consolidated report")
        return False

    logger.info(f"Generating consolidated report for agent '{agent_name}' ({len(failure_reports)} report(s))")

    temp_dir = Path(tempfile.mkdtemp(prefix=f"benchmark_consolidated_report_{agent_name}_"))

    try:
        all_reports = "\n".join(failure_reports)
        consolidated_user_prompt = render(CONSOLIDATED_REPORT_USER_PROMPT, all_reports=all_reports)

        options = ClaudeAgentOptions(
            system_prompt=CONSOLIDATED_REPORT_SYSTEM_PROMPT,
            cwd=str(temp_dir),
            allowed_tools=["Write"],
            max_turns=10,
            permission_mode="default",
            env={"ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", "")},
        )

        messages = []
        async with ClaudeSDKClient(options=options) as client:
            await client.query(consolidated_user_prompt)
            async for msg in client.receive_response():
                messages.append(msg)

            # Check if the report was created at the expected path
            report_path = temp_dir / "CONSOLIDATED_REPORT.md"
            if not report_path.exists():
                logger.warning(
                    f"Report not found at expected path: {report_path}. Asking agent to verify the location..."
                )
                await client.query(
                    f"The report was not found at {report_path}. "
                    "Can you please double check if it is in the correct location and has the correct file name? "
                    "The file should be named 'CONSOLIDATED_REPORT.md' and placed in the current working directory."
                )
                async for msg in client.receive_response():
                    messages.append(msg)

        # Get the report from the temp dir and move it to the runs dir
        report_path = temp_dir / "CONSOLIDATED_REPORT.md"
        if report_path.exists():
            output_path = runs_dir / f"CONSOLIDATED_REPORT_{agent_name}.md"
            shutil.copy2(report_path, output_path)
            logger.info(f"Consolidated report for '{agent_name}' saved to: {output_path}")
            return True

        logger.warning(f"Failed to generate consolidated report for agent '{agent_name}'")
        # Create a placeholder
        output_path = runs_dir / f"CONSOLIDATED_REPORT_{agent_name}.md"
        output_path.write_text(
            f"# Consolidated Report for {agent_name}\n\nNo report was generated. Check the logs for details.\n",
            encoding="utf-8",
        )
        return False
    finally:
        if temp_dir.exists():
            # Best effort cleanup
            with contextlib.suppress(Exception):
                shutil.rmtree(temp_dir)


CONSOLIDATED_REPORT_SYSTEM_PROMPT = """You are an expert at synthesizing benchmark failure analysis reports and you will be synthesizing across many such reports. \
It is critical that your final report is factual and accurate based on the provided reports. \
Do not make assumptions and inferences that are not directly supported by the reports.

ABOUT BENCHMARKING:
Benchmarking is a way of evaluating AI agents on real-world tasks within sandboxed environments. \
Each task includes a natural language instruction, test scripts for verification, and runs in isolated containers. \
Tasks range from programming to document creation.

Each task can be run multiple times (trials) to assess consistency. \
You will see multiple reports for the same task but different trial numbers. \
This helps identify if failures are consistent or sporadic.

ABOUT THESE REPORTS:
Each individual report was generated by analyzing:
- The task instruction (what the agent was asked to do)
- Agent transcript (full conversation showing all agent actions and tool uses)
- Test output (which tests passed/failed and why)
- Original Dockerfile (environment setup)
- Test files (pytest requirements)

Each report header contains:
- Task name: The benchmark task being evaluated
- Trial number: Which trial run (tasks are run multiple times)
- Score: The final score for that specific trial (0-100)

When analyzing, consider:
- Do the same tasks fail consistently across trials? (indicates systemic issues)
- Do some tasks fail sporadically? (indicates non-deterministic problems)

YOUR GOAL:
Identify patterns, common failure modes, and systemic issues across multiple task failures.
Then write a consolidated report summarizing key findings and synthesis about where the agent struggled.

ANALYSIS APPROACH:
- Look for recurring failure patterns (e.g., similar root causes across tasks)
- Identify agent weaknesses (e.g., poor error handling, missing validation, incorrect assumptions)
- Group failures by type (e.g., environment setup issues, logic errors, test misunderstanding)

OUTPUT: Write your consolidated analysis to `CONSOLIDATED_REPORT.md` using the Write tool following the structure provided in the user's message."""

CONSOLIDATED_REPORT_USER_PROMPT = """Analyze the following benchmark task failure reports and create a consolidated analysis.

Your mission: Identify patterns, common failure modes, and systemic issues across all failed tasks.

OUTPUT: Create a comprehensive report in `CONSOLIDATED_REPORT.md` following this structure:

Create sections for each failure category:
   - Group failures by type (e.g., environment issues, logic errors, test misunderstanding)
   - Create up to 5 categories. If there are tasks that don't fit, create an "Other" category.
   - Count and list tasks in each category
   - Include a common root cause analysis for each category as to what went wrong based on the individual reports

Finally, AT THE END, go back and add an executive summary (2-4 sentences) at the top of the report.
   - Overall assessment of agent performance
   - Key patterns observed (list of failure categories, what they are, and the amount of tasks in that category)

DO NOT add:
- A conclusion and appendix
- Recommendations for improvement
- An assessment of the agent's strengths
- Anything else that does not fit the structure above
---

# Individual Task Reports

{{all_reports}}"""
