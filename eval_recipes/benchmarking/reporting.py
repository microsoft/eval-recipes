import inspect
import json
from pathlib import Path
import shutil
import tempfile

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from liquid import render
from loguru import logger

from eval_recipes.benchmarking.semantic_test import semantic_test

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
- test_results.json contains higher level information about the outcome of the tests including the final score that was given.
test.py will often call a semantic test function, this is its signature so that you understand how it works:
{{semantic_test_signature}}

It can also call other evaluation recipes from the eval_recipes package, this is the high level README of what it can do:
{{eval_recipes_readme}}

IMPORTANT: The files provided can be VERY LONG. You MUST:
1. Read files incrementally (use offset/limit with Read tool)
2. Focus on failure points in test_output.log
3. Trace back through agent_output.log to find what the agent did wrong
4. Cross-reference with instructions.txt and test.py to understand expected behavior

ANALYSIS APPROACH:
- Start with test_results.json to identify which tests failed and get a rough understanding of why.
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


async def generate_task_report(benchmark_output_dir: Path, task_directory: Path) -> None:
    """
    Generate detailed failure analysis reports for a benchmark task.
    Generates a separate report for each non-perfect trial.

    Args:
        benchmark_output_dir: Path to the benchmark run directory containing logs
        task_directory: Path to the task directory containing instructions.txt

    Creates:
        FAILURE_REPORT_trial_N.md in trial_N/ subdirectories for each non-perfect trial
    """
    # Get task name from directory name
    task_name = task_directory.name

    # Read task instructions from instructions.txt in task directory
    instructions_file = task_directory / "instructions.txt"
    if not instructions_file.exists():
        raise FileNotFoundError(f"Instructions file not found: {instructions_file}")
    task_instructions = instructions_file.read_text()

    # Check if this is a multi-trial run by looking for aggregated_results.json
    aggregated_results_path = benchmark_output_dir / "aggregated_results.json"
    if not aggregated_results_path.exists():
        logger.warning(f"No aggregated_results.json found in {benchmark_output_dir}")
        return

    # Load aggregated results to find which trials need reports
    with aggregated_results_path.open() as f:
        aggregated_data = json.load(f)

    trials_data = aggregated_data.get("trials", [])
    if not trials_data:
        logger.warning(f"No trials found in aggregated results for {benchmark_output_dir}")
        return

    # Read LOW_LEVEL_API.md for eval_recipes context
    low_level_api_path = Path(__file__).parents[2] / "docs" / "LOW_LEVEL_API.md"
    eval_recipes_readme = low_level_api_path.read_text() if low_level_api_path.exists() else "Not available"

    # Get semantic_test signature dynamically using inspect
    semantic_test_signature = inspect.getsource(semantic_test).split("\n\n")[0]  # Get function def and docstring

    # Render system prompt with placeholders
    rendered_system_prompt = render(
        TASK_REPORT_SYSTEM_PROMPT,
        semantic_test_signature=semantic_test_signature,
        eval_recipes_readme=eval_recipes_readme,
    )

    # Generate reports for each non-perfect trial
    for trial_data in trials_data:
        trial_number = trial_data.get("trial_number")
        trial_score = trial_data.get("score", 0)
        if trial_score >= 85.0:
            logger.info(f"Skipping report for trial {trial_number} (score >= 85%)")
            continue

        trial_dir = benchmark_output_dir / f"trial_{trial_number}"
        if not trial_dir.exists():
            logger.warning(f"Trial directory not found: {trial_dir}")
            continue

        logger.info(f"Generating failure report for trial {trial_number} (score: {trial_score})")

        # Create a temp dir to place files for Claude Agent SDK to work
        temp_dir = Path(tempfile.mkdtemp(prefix=f"benchmark_report_{task_name}_trial{trial_number}_"))

        try:
            # Collect files from trial directory
            files_to_copy = {
                "agent_output.log": trial_dir / "agent_output.log",
                "test_output.log": trial_dir / "test_output.log",
                "test_results.json": trial_dir / "test_results.json",
                "test.py": trial_dir / "test.py",
                "instructions.txt": task_directory / "instructions.txt",
            }

            # Copy available files to temp directory
            for dest_name, source_path in files_to_copy.items():
                if source_path.exists():
                    dest_path = temp_dir / dest_name
                    shutil.copy2(source_path, dest_path)

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
                generated_content = report_path.read_text()
                # Prepend metadata header to each report
                metadata_header = f"""---
**Task**: {task_name}
**Trial**: {trial_number}
**Score**: {trial_score:.1f}%
---

"""
                final_content = metadata_header + generated_content

                output_path = trial_dir / f"FAILURE_REPORT_trial_{trial_number}.md"
                output_path.write_text(final_content)
                logger.info(f"Failure report for trial {trial_number} saved to: {output_path}")
        finally:
            if temp_dir.exists():
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


def _extract_agent_name(run_dir_name: str, known_agents: list[str]) -> str:
    """
    Extract agent name from run directory name.

    The directory format is {agent_name}_{task_name}, where both can contain underscores.
    We match against known agent names to find the correct agent.

    Args:
        run_dir_name: Name of the run directory (e.g., "claude_code_style_blender")
        known_agents: List of known agent names from the agents directory

    Returns:
        The extracted agent name (e.g., "claude_code")
    """
    # Try to match against known agents, preferring the longest match
    # This handles cases where one agent name is a prefix of another (e.g., "gpt" vs "gpt4")
    matching_agents = [agent for agent in known_agents if run_dir_name.startswith(agent + "_")]

    if matching_agents:
        # Return the longest matching agent name
        return max(matching_agents, key=len)

    # Fallback: use heuristic if no known agents match
    # This handles cases where the agents directory might not be available
    parts = run_dir_name.split("_")
    return parts[0] if parts else run_dir_name


def _group_reports_by_agent(benchmarks_output_dir: Path) -> dict[str, list[tuple[str, Path]]]:
    """
    Group failure reports by agent name.

    Args:
        benchmarks_output_dir: Directory containing benchmark run outputs

    Returns:
        Dictionary mapping agent name to list of (report_identifier, report_path) tuples
        Report identifier format: "{agent}_{task}_trial_{N}"
    """
    # Try to discover known agents from the agents directory
    # Look for agents directory relative to the repo structure
    repo_root = Path(__file__).parents[2]
    agents_dir = repo_root / "data" / "agents"
    known_agents: list[str] = []

    if agents_dir.exists() and agents_dir.is_dir():
        known_agents = [d.name for d in agents_dir.iterdir() if d.is_dir()]
        logger.info(f"Found {len(known_agents)} known agent(s): {', '.join(known_agents)}")

    agent_reports: dict[str, list[tuple[str, Path]]] = {}
    for run_dir in benchmarks_output_dir.iterdir():
        if not run_dir.is_dir():
            continue

        agent_name = _extract_agent_name(run_dir.name, known_agents)
        if agent_name not in agent_reports:
            agent_reports[agent_name] = []

        # Look for trial subdirectories with failure reports
        trial_dirs = list(run_dir.glob("trial_*"))
        if not trial_dirs:
            logger.warning(f"No trial directories found in {run_dir.name}, skipping")
            continue

        # Collect reports from each trial
        for trial_dir in sorted(trial_dirs):
            for report_path in trial_dir.glob("FAILURE_REPORT_trial_*.md"):
                report_identifier = f"{run_dir.name}_{trial_dir.name}"
                agent_reports[agent_name].append((report_identifier, report_path))

    return agent_reports


async def generate_summary_report(benchmarks_output_dir: Path) -> None:
    """
    Generate consolidated reports synthesizing individual task failure reports.

    This function now creates a separate consolidated report for each agent found
    in the benchmark run directory.

    Args:
        benchmarks_output_dir: Directory containing benchmark run outputs with FAILURE_REPORT.md files

    Creates:
        CONSOLIDATED_REPORT_{agent_name}.md files for each agent in benchmarks_output_dir
    """
    # Group failure reports by agent
    agent_reports = _group_reports_by_agent(benchmarks_output_dir)

    if not agent_reports:
        logger.warning("No failure reports found to consolidate")
        return

    logger.info(f"Found {len(agent_reports)} agent(s) to generate consolidated reports for")

    # Generate a consolidated report for each agent
    for agent_name, reports in agent_reports.items():
        logger.info(f"Generating consolidated report for agent '{agent_name}' ({len(reports)} task(s))")

        # Collect all failure reports for this agent
        failure_reports = []
        for run_dir_name, report_path in reports:
            content = report_path.read_text()
            failure_reports.append(f"## Report for Task: {run_dir_name}\n\n{content}\n\n{'=' * 80}\n")

        temp_dir = Path(tempfile.mkdtemp(prefix=f"benchmark_consolidated_report_{agent_name}_"))
        report_path = temp_dir / "CONSOLIDATED_REPORT.md"
        try:
            all_reports = "\n".join(failure_reports)
            consolidated_user_prompt = render(CONSOLIDATED_REPORT_USER_PROMPT, all_reports=all_reports)
            options = ClaudeAgentOptions(
                system_prompt=CONSOLIDATED_REPORT_SYSTEM_PROMPT,
                cwd=str(temp_dir),
                allowed_tools=["Write"],
                max_turns=10,
                permission_mode="default",
            )
            messages = []
            async with ClaudeSDKClient(options=options) as client:
                await client.query(consolidated_user_prompt)
                async for msg in client.receive_response():
                    messages.append(msg)
                    continue

                # Check if the report was created at the expected path
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
                        continue

            # Get the report from the temp dir and move it to the benchmark output dir
            if report_path.exists():
                output_path = benchmarks_output_dir / f"CONSOLIDATED_REPORT_{agent_name}.md"
                shutil.copy2(report_path, output_path)
                logger.info(f"Consolidated report for '{agent_name}' saved to: {output_path}")
            else:
                logger.warning(f"No consolidated report was generated for agent '{agent_name}'. Check the logs.")
                # Create a placeholder
                output_path = benchmarks_output_dir / f"CONSOLIDATED_REPORT_{agent_name}.md"
                output_path.write_text(
                    f"# Consolidated Report for {agent_name}\n\nNo report was generated. Check the logs for details.\n"
                )

        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
