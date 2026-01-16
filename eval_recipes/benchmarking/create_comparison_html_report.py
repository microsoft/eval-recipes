# Copyright (c) Microsoft. All rights reserved

from collections import defaultdict
from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
import re

import yaml

from eval_recipes.benchmarking.schemas import (
    AggregateReport,
    ComparisonBenchmarkResults,
    ComparisonRunResult,
    FinalAggregateReport,
)


@dataclass
class TaskInfo:
    """Information loaded from task directory."""

    task_name: str
    categories: list[str]
    instructions: str
    task_info_data: dict  # Full task_info from task.yaml


@dataclass
class TaskRankingsData:
    """Rankings data for a single task within an agent combination."""

    task_info: TaskInfo
    agent_ranks: dict[str, list[int]]  # Agent name -> list of ranks per comparison run
    num_comparison_runs: int
    comparison_runs: list[ComparisonRunResult]  # To access reasoning for each run


@dataclass
class OverviewMetrics:
    """Cross-task aggregation metrics for an agent combination."""

    overall_avg_rank: dict[str, float]  # agent_name -> avg rank across all tasks/runs
    overall_win_rate: dict[str, float]  # agent_name -> win rate (0-100)
    task_wins: dict[str, int]  # agent_name -> number of tasks won
    task_ties: int  # number of tasks with ties
    mean_kendalls_w: float | None


def _format_name(name: str) -> str:
    """Format a snake_case name for display."""
    special_names = {
        "gh_cli": "GitHub CLI",
        "amplifier_v1": "Amplifier Claude",
        "amplifier_v2": "Amplifier",
        "amplifier_v2_aoai": "Amplifier AOAI",
        "amplifier_v2_toolkit": "Amplifier Toolkit",
        "amplifier_foundation": "Amplifier Foundation",
        "claude_code": "Claude Code",
    }
    name_lower = name.lower()
    if name_lower in special_names:
        return special_names[name_lower]

    words = name.split("_")
    formatted_words = []
    for word in words:
        word_lower = word.lower()
        if word_lower in ("api", "llm", "ai", "gpt", "csv", "json", "yaml", "html", "pdf", "cli", "url"):
            formatted_words.append(word.upper())
        else:
            formatted_words.append(word.capitalize())
    return " ".join(formatted_words)


def _load_comparison_results(results_file: Path) -> ComparisonBenchmarkResults:
    """Load comparison results from raw_results.json."""
    with results_file.open(encoding="utf-8") as f:
        data = json.load(f)
    return ComparisonBenchmarkResults.model_validate(data)


def _load_task_info(tasks_dir: Path, task_name: str) -> TaskInfo:
    """Load task categories and instructions from task directory."""
    task_dir = tasks_dir / task_name
    categories: list[str] = []
    instructions = "Instructions not available."
    task_info_data: dict = {}

    # Load categories from task.yaml
    task_yaml_path = task_dir / "task.yaml"
    if task_yaml_path.exists():
        with task_yaml_path.open(encoding="utf-8") as f:
            task_yaml = yaml.safe_load(f) or {}
        task_info_data = task_yaml.get("task_info", {})
        categories = task_info_data.get("categories", [])

    # Load instructions from instructions.txt
    instructions_path = task_dir / "instructions.txt"
    if instructions_path.exists():
        instructions = instructions_path.read_text(encoding="utf-8")

    return TaskInfo(
        task_name=task_name, categories=categories, instructions=instructions, task_info_data=task_info_data
    )


def _load_aggregate_report(runs_dir: Path, comparison_folder_name: str) -> AggregateReport | None:
    """Load aggregate report from disk if it exists."""
    report_path = runs_dir / comparison_folder_name / "aggregate_report.json"
    if not report_path.exists():
        return None

    with report_path.open(encoding="utf-8") as f:
        data = json.load(f)
    return AggregateReport.model_validate(data)


def _load_final_aggregate_report(runs_dir: Path) -> FinalAggregateReport | None:
    """Load final aggregate report from disk if it exists."""
    report_path = runs_dir / "final_aggregate_report.json"
    if not report_path.exists():
        return None

    with report_path.open(encoding="utf-8") as f:
        data = json.load(f)
    return FinalAggregateReport.model_validate(data)


def _get_comparison_folder_name(task_name: str, agent_names: list[str]) -> str:
    """Generate a folder name for a comparison."""
    return f"{task_name}-{'_vs_'.join(agent_names)}"


def _group_runs_by_agent_combination(
    results: ComparisonBenchmarkResults,
) -> dict[tuple[str, ...], list[ComparisonRunResult]]:
    """Group comparison runs by their unique agent combination (sorted tuple)."""
    groups: dict[tuple[str, ...], list[ComparisonRunResult]] = defaultdict(list)
    for run in results.comparison_runs:
        key = tuple(sorted(run.agent_names))
        groups[key].append(run)
    return dict(groups)


def _get_agent_rank_positions(rankings: list[int], agent_names: list[str]) -> dict[str, int]:
    """
    Convert rankings array to per-agent 1-indexed rank positions.

    rankings is ordered best-to-worst, where each element is an index into agent_names.
    e.g., rankings = [1, 0] with agent_names = ["A", "B"] means B won (rank 1), A lost (rank 2).
    """
    return {agent_names[idx]: rank + 1 for rank, idx in enumerate(rankings)}


def _deanonymize_reasoning(
    reasoning: str,
    anonymous_to_index: dict[str, int],
    agent_names: list[str],
) -> str:
    """Replace anonymous agent identifiers with actual agent names in reasoning text."""
    result = reasoning
    for anon_name, index in anonymous_to_index.items():
        if 0 <= index < len(agent_names):
            actual_name = _format_name(agent_names[index])
            result = re.sub(re.escape(anon_name), actual_name, result, flags=re.IGNORECASE)
    return result


def _calculate_kendalls_w(agent_ranks: dict[str, list[int]]) -> float | None:
    """
    Calculate Kendall's W (coefficient of concordance).

    Measures agreement among m comparison runs ranking n agents.
    W ranges from 0 (no agreement) to 1 (perfect agreement).

    Formula: W = 12 * S / (m² * (n³ - n))
    Where S = sum of squared deviations of rank sums from mean.
    """
    if not agent_ranks:
        return None

    agents = list(agent_ranks.keys())
    n = len(agents)  # number of agents (items)
    m = len(agent_ranks[agents[0]])  # number of comparisons (judges)

    if m < 2 or n < 2:
        return None  # Need at least 2 comparisons and 2 agents

    # Calculate rank sums for each agent
    rank_sums = [sum(agent_ranks[agent]) for agent in agents]

    # Calculate mean rank sum
    mean_rank_sum = sum(rank_sums) / n

    # Calculate S (sum of squared deviations)
    s = sum((r - mean_rank_sum) ** 2 for r in rank_sums)

    # Calculate W
    w = 12 * s / (m**2 * (n**3 - n))

    return w


def _get_rank_color(avg_rank: float, num_agents: int) -> str:
    """Get background color for agent based on average rank value.

    Uses HSL interpolation for clean green -> yellow -> red gradient.
    Scale: avg_rank of 1.0 = green, avg_rank of num_agents = red.
    """
    if num_agents <= 1:
        return "#22c55e"  # Green for single agent

    # Calculate ratio based on actual rank value (1 = best, num_agents = worst)
    ratio = (avg_rank - 1) / (num_agents - 1)
    ratio = max(0, min(1, ratio))  # Clamp to [0, 1]

    # Interpolate hue from green (120) to red (0) through yellow (60)
    # This follows the color wheel naturally
    hue = 120 * (1 - ratio)  # 120 = green, 60 = yellow, 0 = red
    saturation = 75
    lightness = 38  # Darker for better white text contrast

    # Convert HSL to RGB
    c = (1 - abs(2 * lightness / 100 - 1)) * saturation / 100
    x = c * (1 - abs((hue / 60) % 2 - 1))
    m = lightness / 100 - c / 2

    if hue < 60:
        r, g, b = c, x, 0
    elif hue < 120:
        r, g, b = x, c, 0
    else:
        r, g, b = 0, c, x

    r = int((r + m) * 255)
    g = int((g + m) * 255)
    b = int((b + m) * 255)

    return f"#{r:02x}{g:02x}{b:02x}"


def _build_task_rankings_data(
    runs: list[ComparisonRunResult],
    tasks_dir: Path,
) -> dict[str, TaskRankingsData]:
    """Build per-task rankings tables for runs with same agent combination."""
    # Group runs by task
    task_runs: dict[str, list[ComparisonRunResult]] = defaultdict(list)
    for run in runs:
        task_runs[run.task_name].append(run)

    result: dict[str, TaskRankingsData] = {}
    for task_name, runs_for_task in sorted(task_runs.items()):
        if not runs_for_task:
            continue

        task_info = _load_task_info(tasks_dir, task_name)
        agent_names = runs_for_task[0].agent_names

        # Initialize agent_ranks for each agent
        agent_ranks: dict[str, list[int]] = {name: [] for name in agent_names}

        # Sort runs by comparison_run_num for consistent column ordering
        sorted_runs = sorted(runs_for_task, key=lambda r: r.comparison_run_num)

        for run in sorted_runs:
            rank_positions = _get_agent_rank_positions(run.result.rankings, run.agent_names)
            for agent_name in agent_names:
                agent_ranks[agent_name].append(rank_positions.get(agent_name, len(agent_names)))

        result[task_name] = TaskRankingsData(
            task_info=task_info,
            agent_ranks=agent_ranks,
            num_comparison_runs=len(sorted_runs),
            comparison_runs=sorted_runs,
        )

    return result


def _compute_overview_metrics(tasks_data: dict[str, TaskRankingsData]) -> OverviewMetrics:
    """Compute cross-task aggregation metrics for an agent combination.

    Args:
        tasks_data: Per-task rankings data for all tasks in this agent combination.

    Returns:
        OverviewMetrics with aggregated statistics.
    """
    if not tasks_data:
        return OverviewMetrics(
            overall_avg_rank={},
            overall_win_rate={},
            task_wins={},
            task_ties=0,
            mean_kendalls_w=None,
        )

    # Collect all ranks per agent across all tasks
    total_ranks: dict[str, list[int]] = defaultdict(list)
    for task_data in tasks_data.values():
        for agent, ranks in task_data.agent_ranks.items():
            total_ranks[agent].extend(ranks)

    # Overall Average Rank
    overall_avg_rank = {agent: sum(ranks) / len(ranks) for agent, ranks in total_ranks.items() if ranks}

    # Overall Win Rate (percentage ranked #1)
    total_wins: dict[str, int] = defaultdict(int)
    total_comparisons = 0
    for task_data in tasks_data.values():
        for agent, ranks in task_data.agent_ranks.items():
            total_wins[agent] += sum(1 for r in ranks if r == 1)
        # Each comparison run has exactly one #1, count total comparison runs
        if task_data.agent_ranks:
            first_agent = next(iter(task_data.agent_ranks.keys()))
            total_comparisons += len(task_data.agent_ranks[first_agent])

    overall_win_rate = {
        agent: (wins / total_comparisons * 100) if total_comparisons > 0 else 0 for agent, wins in total_wins.items()
    }

    # Task Wins (which agent won each task based on avg rank)
    task_wins: dict[str, int] = defaultdict(int)
    task_ties = 0
    for task_data in tasks_data.values():
        if not task_data.agent_ranks:
            continue
        avg_ranks = {agent: sum(ranks) / len(ranks) for agent, ranks in task_data.agent_ranks.items() if ranks}
        if not avg_ranks:
            continue
        best_rank = min(avg_ranks.values())
        winners = [agent for agent, rank in avg_ranks.items() if rank == best_rank]
        if len(winners) == 1:
            task_wins[winners[0]] += 1
        else:
            task_ties += 1

    # Mean Kendall's W across all tasks
    w_values = []
    for task_data in tasks_data.values():
        w = _calculate_kendalls_w(task_data.agent_ranks)
        if w is not None:
            w_values.append(w)
    mean_kendalls_w = sum(w_values) / len(w_values) if w_values else None

    return OverviewMetrics(
        overall_avg_rank=overall_avg_rank,
        overall_win_rate=overall_win_rate,
        task_wins=dict(task_wins),
        task_ties=task_ties,
        mean_kendalls_w=mean_kendalls_w,
    )


def _generate_methodology_html() -> str:
    """Generate HTML content for the Methodology tab explaining how results are computed."""
    return """
        <div class="methodology-container">
            <div class="methodology-section">
                <h2>Overview</h2>
                <p>
                    This report presents results from <strong>comparison-based evaluation</strong>,
                    where multiple AI agents are evaluated on the same tasks and ranked relative to each other.
                    Rather than assigning absolute scores, an judge agent (an LLM agent with shell access) determines which agent performed better
                    on each task.
                </p>
                <div class="workflow-diagram">
                    <div class="workflow-step">
                        <div class="step-number">1</div>
                        <div class="step-content">
                            <strong>Trial Execution</strong>
                            <span>Each agent completes the task independently</span>
                        </div>
                    </div>
                    <div class="workflow-arrow">&#8594;</div>
                    <div class="workflow-step">
                        <div class="step-number">2</div>
                        <div class="step-content">
                            <strong>Blind Comparison</strong>
                            <span>Judge agent evaluates anonymized outputs</span>
                        </div>
                    </div>
                    <div class="workflow-arrow">&#8594;</div>
                    <div class="workflow-step">
                        <div class="step-number">3</div>
                        <div class="step-content">
                            <strong>Aggregation</strong>
                            <span>Results compiled across multiple runs</span>
                        </div>
                    </div>
                </div>
            </div>

            <div class="methodology-section">
                <h2>Evaluation Approach</h2>
                <p>The judge agent (an LLM agent with shell access) follows these principles when evaluating agent outputs:</p>
                <ul>
                    <li><strong>Task-focused:</strong> Evaluates deliverables against the original task instructions</li>
                    <li><strong>Critical assessment:</strong> Identifies both strengths and weaknesses objectively</li>
                    <li><strong>No modification:</strong> Evaluates outputs as-is without fixing or debugging</li>
                    <li><strong>Ordinal ranking:</strong> Produces a ranking from best to worst with reasoning</li>
                </ul>
            </div>

            <div class="methodology-section">
                <h2>Multi-Trial Consistency</h2>
                <p>
                    Each comparison is run <strong>multiple times</strong> to measure consistency and reduce
                    variance from any single evaluation. This approach:
                </p>
                <ul>
                    <li>Captures uncertainty in relative rankings</li>
                    <li>Identifies when agents produce similar quality outputs (low agreement)</li>
                    <li>Provides more robust results than single-run evaluations</li>
                </ul>
            </div>

            <div class="methodology-section">
                <h2>Metrics Explained</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <h3>Average Rank</h3>
                        <p>
                            The mean position across all comparison runs. <strong>Lower is better</strong>
                            (1 = best possible).
                        </p>
                        <div class="metric-example">
                            Example: If an agent ranks 1st, 1st, and 2nd across 3 runs,
                            the average rank is (1+1+2)/3 = 1.33
                        </div>
                    </div>

                    <div class="metric-card">
                        <h3>Win Rate</h3>
                        <p>
                            The percentage of comparison runs where the agent was ranked #1.
                            <strong>Higher is better.</strong>
                        </p>
                        <div class="metric-example">
                            Example: If an agent ranks 1st in 5 out of 7 runs,
                            the win rate is 5/7 = 71%
                        </div>
                    </div>

                    <div class="metric-card">
                        <h3>Task Wins</h3>
                        <p>
                            The count of tasks where the agent achieved the best (lowest)
                            average rank among all agents compared.
                        </p>
                        <div class="metric-example">
                            Example: "3/5" means the agent won 3 out of 5 tasks
                        </div>
                    </div>

                    <div class="metric-card">
                        <h3>Kendall's W (Agreement)</h3>
                        <p>
                            A statistical measure of inter-rater agreement, indicating how consistently
                            the judge ranked agents across multiple runs.
                        </p>
                        <div class="metric-formula">
                            W = 12S / (m&sup2;(n&sup3; - n))
                        </div>
                        <p class="metric-note">
                            where S = sum of squared deviations, m = number of runs, n = number of agents
                        </p>
                        <div class="metric-interpretation">
                            <div class="interp-row"><span class="interp-value">W &lt; 0.33</span><span>Low agreement (outputs may be similar or evaluation inconsistent)</span></div>
                            <div class="interp-row"><span class="interp-value">W &ge; 0.33</span><span>Moderate agreement (results likely significant)</span></div>
                            <div class="interp-row"><span class="interp-value">W &ge; 0.67</span><span>Strong agreement (clear performance differences)</span></div>
                            <div class="interp-row"><span class="interp-value">W = 1.0</span><span>Perfect agreement (identical rankings every run)</span></div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="methodology-section">
                <h2>Analysis Generation</h2>
                <p>
                    The report includes two levels of LLM-generated analysis:
                </p>
                <ul>
                    <li>
                        <strong>Task-level Aggregate Analysis:</strong> Synthesizes patterns across
                        multiple comparison runs for each task, explaining why certain agents
                        consistently ranked higher or lower.
                    </li>
                    <li>
                        <strong>Final Analysis:</strong> Synthesizes patterns across all tasks,
                        identifying overall strengths and weaknesses of each agent.
                    </li>
                </ul>
            </div>
        </div>
    """


def _generate_html(
    agent_combination_data: dict[tuple[str, ...], dict[str, TaskRankingsData]],
    runs_dir: Path,
) -> str:
    """Generate the complete HTML report."""
    timestamp = runs_dir.name
    markdown_content: dict[str, str] = {}

    html_parts = [
        """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comparison Benchmark Results</title>
    <script src="https://cdn.jsdelivr.net/npm/marked@9.1.6/marked.min.js" defer></script>
    <style>
        :root {
            --primary: #2563eb;
            --success: #10b981;
            --warning: #f59e0b;
            --background: #ffffff;
            --surface: #f3f4f6;
            --text: #1f2937;
            --text-muted: #6b7280;
            --border: #e5e7eb;
        }

        * { box-sizing: border-box; }

        body {
            margin: 0;
            padding: 2rem;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--background);
            color: var(--text);
            line-height: 1.6;
        }

        .container { max-width: 1400px; margin: 0 auto; }

        h1 { margin-top: 0; margin-bottom: 0.5rem; font-size: 2rem; }
        .subtitle { color: var(--text-muted); margin-bottom: 2rem; font-size: 0.9rem; }

        /* Tabs */
        .tabs {
            display: flex;
            gap: 0.5rem;
            border-bottom: 2px solid var(--border);
            margin-bottom: 2rem;
            flex-wrap: wrap;
        }

        .tab {
            padding: 0.75rem 1.5rem;
            background: transparent;
            border: none;
            border-bottom: 3px solid transparent;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 500;
            color: var(--text-muted);
            transition: all 0.2s;
        }

        .tab:hover { color: var(--text); background: var(--surface); }
        .tab.active { color: var(--primary); border-bottom-color: var(--primary); }

        .tab-content { display: none; }
        .tab-content.active { display: block; }

        /* Task Section */
        .task-section {
            background: white;
            border: 1px solid var(--border);
            border-radius: 8px;
            margin-bottom: 1rem;
            overflow: hidden;
        }

        .task-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 1rem 1.5rem;
            cursor: pointer;
            transition: background 0.2s;
        }

        .task-header:hover { background: var(--surface); }

        .task-name { font-weight: 600; font-size: 1.1rem; }

        .expand-icon {
            margin-left: auto;
            color: var(--text-muted);
            transition: transform 0.2s;
        }

        .task-header.expanded .expand-icon { transform: rotate(90deg); }

        /* Category Pills */
        .category-pill {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            background: #e0e7ff;
            color: #3730a3;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 500;
        }

        /* Agent Score Badges */
        .agent-score {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            color: white;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 500;
        }

        /* Task Content */
        .task-content {
            display: none;
            padding: 0 1.5rem 1.5rem;
            border-top: 1px solid var(--border);
            background: var(--surface);
        }

        .task-content.open { display: block; }

        /* Subsections */
        .subsection {
            margin-top: 1rem;
        }

        .subsection-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.75rem 1rem;
            background: white;
            border: 1px solid var(--border);
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.9rem;
        }

        .subsection-header:hover { background: #fafafa; }

        .subsection-header.expanded .expand-icon { transform: rotate(90deg); }

        .subsection-content {
            display: none;
            padding: 1rem;
            background: white;
            border: 1px solid var(--border);
            border-top: none;
            border-radius: 0 0 6px 6px;
        }

        .subsection-content.open { display: block; }

        /* Nested Collapsibles (for reasoning) */
        .nested-collapsible-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.5rem 0.75rem;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 4px;
            cursor: pointer;
            font-weight: 500;
            font-size: 0.85rem;
            margin-bottom: 0.25rem;
        }

        .nested-collapsible-header:hover { background: #e9ecef; }
        .nested-collapsible-header.expanded .expand-icon { transform: rotate(90deg); }

        .nested-collapsible-content {
            display: none;
            padding: 0.75rem;
            background: white;
            border: 1px solid var(--border);
            border-top: none;
            border-radius: 0 0 4px 4px;
            margin-top: -0.25rem;
            margin-bottom: 0.5rem;
        }

        .nested-collapsible-content.open { display: block; }

        /* Average Rank Summary */
        .avg-rank-summary {
            padding: 0.25rem 0 0.5rem 0;
            margin-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
        }

        .avg-rank-summary h4 {
            margin: 0 0 0.5rem 0;
            font-size: 0.9rem;
            color: var(--text);
        }

        .avg-rank-list {
            margin: 0;
            padding-left: 1.5rem;
            font-size: 0.9rem;
        }

        .avg-rank-list li {
            margin-bottom: 0.25rem;
        }

        .avg-rank-list .agent-name {
            font-style: italic;
        }

        .agreement-line {
            margin: 0 0 0.25rem 0;
            font-size: 0.9rem;
        }

        .agreement-note {
            margin: 0 0 0.5rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
            font-size: 0.75rem;
            color: var(--text-muted);
        }

        .section-heading {
            margin: 0.75rem 0 0.25rem 0;
            font-size: 1rem;
            font-weight: 600;
            color: var(--text);
        }

        /* Markdown Content */
        .markdown-content {
            max-height: 400px;
            overflow-y: auto;
            font-size: 0.9rem;
            line-height: 1.6;
        }

        .markdown-content pre {
            background: var(--surface);
            padding: 0.75rem;
            border-radius: 4px;
            overflow-x: auto;
        }

        /* Rankings Table */
        .rankings-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }

        .rankings-table th,
        .rankings-table td {
            padding: 0.5rem 0.75rem;
            text-align: center;
            border: 1px solid var(--border);
        }

        .rankings-table th {
            background: var(--surface);
            font-weight: 600;
        }

        .rankings-table td:first-child {
            text-align: left;
            font-weight: 500;
        }

        .rank-1 { background: #d1fae5; color: #065f46; font-weight: 600; }
        .rank-2 { background: #fef3c7; color: #92400e; }
        .rank-3 { background: #fee2e2; color: #991b1b; }

        /* Overview Section */
        .overview-section {
            background: white;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }

        .overview-section h2 {
            margin: 0 0 1rem 0;
            font-size: 1.25rem;
            color: var(--text);
        }

        .overview-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1rem;
        }

        .overview-metric {
            background: var(--surface);
            border-radius: 6px;
            padding: 1rem;
        }

        .overview-metric h3 {
            margin: 0 0 0.5rem 0;
            font-size: 0.9rem;
            color: var(--text-muted);
            font-weight: 600;
        }

        .overview-metric-list {
            margin: 0;
            padding: 0;
            list-style: none;
        }

        .overview-metric-list li {
            display: flex;
            justify-content: space-between;
            padding: 0.25rem 0;
            font-size: 0.9rem;
        }

        .overview-metric-list .agent-name {
            font-weight: 500;
        }

        .overview-metric-list .value {
            color: var(--text-muted);
        }

        .overview-single-value {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--text);
        }

        .overview-single-note {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }

        /* Methodology Tab Styles */
        .methodology-container {
            width: 100%;
        }

        .methodology-section {
            background: white;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }

        .methodology-section h2 {
            margin: 0 0 1rem 0;
            font-size: 1.25rem;
            color: var(--text);
        }

        .methodology-section p {
            margin: 0 0 0.75rem 0;
            line-height: 1.7;
        }

        .methodology-section ul {
            margin: 0;
            padding-left: 1.5rem;
        }

        .methodology-section li {
            margin-bottom: 0.5rem;
            line-height: 1.6;
        }

        .methodology-section code {
            background: var(--surface);
            padding: 0.15rem 0.4rem;
            border-radius: 3px;
            font-size: 0.9em;
        }

        .workflow-diagram {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            margin-top: 1.5rem;
            flex-wrap: wrap;
        }

        .workflow-step {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            background: var(--surface);
            padding: 1rem 1.25rem;
            border-radius: 8px;
            border: 1px solid var(--border);
        }

        .step-number {
            width: 2rem;
            height: 2rem;
            background: var(--primary);
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            flex-shrink: 0;
        }

        .step-content {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }

        .step-content strong {
            font-size: 0.95rem;
        }

        .step-content span {
            font-size: 0.8rem;
            color: var(--text-muted);
        }

        .workflow-arrow {
            font-size: 1.5rem;
            color: var(--text-muted);
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1rem;
        }

        .metric-card {
            background: var(--surface);
            border-radius: 6px;
            padding: 1rem;
        }

        .metric-card h3 {
            margin: 0 0 0.5rem 0;
            font-size: 1rem;
            color: var(--text);
        }

        .metric-card p {
            margin: 0 0 0.5rem 0;
            font-size: 0.9rem;
        }

        .metric-example {
            font-size: 0.8rem;
            color: var(--text-muted);
            font-style: italic;
            padding: 0.5rem;
            background: white;
            border-radius: 4px;
            margin-top: 0.5rem;
        }

        .metric-formula {
            font-family: 'Times New Roman', serif;
            font-size: 1.1rem;
            text-align: center;
            padding: 0.75rem;
            background: white;
            border-radius: 4px;
            margin: 0.5rem 0;
        }

        .metric-note {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-align: center;
            margin: 0 0 0.75rem 0;
        }

        .metric-interpretation {
            font-size: 0.85rem;
        }

        .interp-row {
            display: flex;
            gap: 0.75rem;
            padding: 0.35rem 0;
            border-bottom: 1px solid var(--border);
        }

        .interp-row:last-child {
            border-bottom: none;
        }

        .interp-value {
            font-weight: 600;
            min-width: 5.5rem;
            font-family: monospace;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Comparison Benchmark Results</h1>
        <div class="subtitle">Run: """,
        escape(timestamp),
        """</div>

        <div class="tabs">""",
    ]

    # Generate tab buttons - Methodology tab first (active by default)
    html_parts.append("""
            <button class="tab active" onclick="switchTab(event, 'methodology')">Methodology</button>""")

    combo_ids: dict[tuple[str, ...], str] = {}
    for i, combo in enumerate(sorted(agent_combination_data.keys())):
        combo_id = f"combo-{i}"
        combo_ids[combo] = combo_id
        formatted_names = " vs ".join(_format_name(name) for name in combo)
        html_parts.append(f"""
            <button class="tab" onclick="switchTab(event, '{combo_id}')">{escape(formatted_names)}</button>""")

    html_parts.append("""
        </div>
""")

    # Generate Methodology tab content (active by default)
    html_parts.append(f"""
        <div class="tab-content active" id="tab-methodology">
            {_generate_methodology_html()}
        </div>
""")

    # Generate tab content for each agent combination
    for combo, tasks_data in sorted(agent_combination_data.items()):
        combo_id = combo_ids[combo]

        html_parts.append(f"""
        <div class="tab-content" id="tab-{combo_id}">""")

        # Generate overview section with cross-task metrics
        overview = _compute_overview_metrics(tasks_data)
        num_tasks = len(tasks_data)

        html_parts.append("""
            <div class="overview-section">
                <h2>Overview</h2>
                <div class="overview-grid">""")

        # Average Rank (sorted by rank, best first)
        html_parts.append("""
                    <div class="overview-metric">
                        <h3 title="Mean rank across all tasks and comparison runs. Lower is better.">Average Rank</h3>
                        <ul class="overview-metric-list">""")
        sorted_avg_ranks = sorted(overview.overall_avg_rank.items(), key=lambda x: x[1])
        for agent, avg_rank in sorted_avg_ranks:
            html_parts.append(f"""
                            <li><span class="agent-name">{escape(_format_name(agent))}</span><span class="value">{avg_rank:.2f}</span></li>""")
        html_parts.append("""
                        </ul>
                    </div>""")

        # Win Rate (sorted by win rate, highest first)
        html_parts.append("""
                    <div class="overview-metric">
                        <h3 title="Percentage of comparison runs where the agent was ranked #1, across all tasks.">Win Rate over all Comparisons</h3>
                        <ul class="overview-metric-list">""")
        sorted_win_rates = sorted(overview.overall_win_rate.items(), key=lambda x: x[1], reverse=True)
        for agent, win_rate in sorted_win_rates:
            html_parts.append(f"""
                            <li><span class="agent-name">{escape(_format_name(agent))}</span><span class="value">{win_rate:.0f}%</span></li>""")
        html_parts.append("""
                        </ul>
                    </div>""")

        # Task Wins (sorted by wins, highest first)
        html_parts.append("""
                    <div class="overview-metric">
                        <h3 title="Number of tasks where the agent had the best (lowest) average rank.">Number of Overall Task Wins</h3>
                        <ul class="overview-metric-list">""")
        # Ensure all agents appear even if they have 0 wins
        all_agents = set(overview.overall_avg_rank.keys())
        task_wins_with_zeros = {agent: overview.task_wins.get(agent, 0) for agent in all_agents}
        sorted_task_wins = sorted(task_wins_with_zeros.items(), key=lambda x: x[1], reverse=True)
        for agent, wins in sorted_task_wins:
            html_parts.append(f"""
                            <li><span class="agent-name">{escape(_format_name(agent))}</span><span class="value">{wins}/{num_tasks}</span></li>""")
        if overview.task_ties > 0:
            html_parts.append(f"""
                            <li><span class="agent-name" style="font-style: italic;">Ties</span><span class="value">{overview.task_ties}</span></li>""")
        html_parts.append("""
                        </ul>
                    </div>""")

        # Mean Kendall's W
        w_display = f"{overview.mean_kendalls_w:.2f}" if overview.mean_kendalls_w is not None else "N/A"
        html_parts.append(f"""
                    <div class="overview-metric">
                        <h3 title="Average Kendall's W across all tasks. Measures how consistently the grader ranked agents (0 = random, 1 = perfect agreement).">Mean Grader Agreement</h3>
                        <div class="overview-single-value">{w_display}</div>
                        <div class="overview-single-note">Mean Kendall's W across tasks. An agreement value of >=0.33 typically means the result is significant. Smaller values may indicate either the grader is not capable of accurately ranking or the two outputs are very similar.</div>
                    </div>""")

        html_parts.append("""
                </div>""")

        # Final Analysis section (if available)
        final_report = _load_final_aggregate_report(runs_dir)
        if final_report:
            final_analysis_id = f"final-analysis-{combo_id}"
            markdown_content[final_analysis_id] = final_report.analysis

            html_parts.append(f"""
                <div class="subsection" style="margin-top: 1rem;">
                    <div class="subsection-header expanded" onclick="toggleSubsection(this)">
                        <span>Final Analysis <em style="font-size: 0.8em; font-weight: normal; color: var(--text-muted);">LLM-generated summary explaining overall results</em></span>
                        <span class="expand-icon">&#9654;</span>
                    </div>
                    <div class="subsection-content open">
                        <div class="markdown-content" id="{final_analysis_id}"></div>
                    </div>
                </div>""")

        html_parts.append("""
            </div>""")

        for task_name, task_data in sorted(tasks_data.items()):
            task_id = f"{combo_id}-{task_name.replace('-', '_')}"
            formatted_task = _format_name(task_name)

            # Task header with categories
            html_parts.append(f"""
            <div class="task-section">
                <div class="task-header" onclick="toggleTaskSection(this)">
                    <span class="task-name">{escape(formatted_task)}</span>""")

            for category in task_data.task_info.categories:
                html_parts.append(f"""
                    <span class="category-pill">{escape(category)}</span>""")

            # Add agent scores with color gradient
            header_agent_ranks = []
            for agent_name in task_data.agent_ranks:
                ranks = task_data.agent_ranks[agent_name]
                avg_rank = sum(ranks) / len(ranks) if ranks else 0
                header_agent_ranks.append((agent_name, avg_rank))
            header_agent_ranks.sort(key=lambda x: x[1])  # Sort by avg rank (best first)
            num_agents = len(header_agent_ranks)

            for agent_name, avg_rank in header_agent_ranks:
                color = _get_rank_color(avg_rank, num_agents)
                formatted_agent = _format_name(agent_name)
                html_parts.append(f"""
                    <span class="agent-score" style="background: {color};">{escape(formatted_agent)}: {avg_rank:.2f}</span>""")

            html_parts.append("""
                    <span class="expand-icon">&#9654;</span>
                </div>
                <div class="task-content">
                    <h3 class="section-heading">Summary Metrics</h3>""")

            # Average rank summary (non-collapsible)
            html_parts.append("""
                    <div class="avg-rank-summary">
                        <h4>Average Ranking across comparisons (lower is better)</h4>
                        <ul class="avg-rank-list">""")

            # Calculate and display average rank for each agent, sorted by avg rank
            agent_avg_ranks = []
            for agent_name in task_data.agent_ranks:
                ranks = task_data.agent_ranks[agent_name]
                avg_rank = sum(ranks) / len(ranks) if ranks else 0
                agent_avg_ranks.append((agent_name, avg_rank))

            # Sort by average rank (best first)
            agent_avg_ranks.sort(key=lambda x: x[1])

            for agent_name, avg_rank in agent_avg_ranks:
                formatted_agent = _format_name(agent_name)
                html_parts.append(f"""
                            <li><span class="agent-name">{escape(formatted_agent)}</span>: {avg_rank:.2f}</li>""")

            html_parts.append("""
                        </ul>
                    </div>""")

            # Win rate summary (non-collapsible)
            html_parts.append("""
                    <div class="avg-rank-summary">
                        <h4>Win Rate (percentage of comparisons ranked #1)</h4>
                        <ul class="avg-rank-list">""")

            # Calculate win rate for each agent
            agent_win_rates = []
            for agent_name in task_data.agent_ranks:
                ranks = task_data.agent_ranks[agent_name]
                wins = sum(1 for r in ranks if r == 1)
                win_rate = (wins / len(ranks) * 100) if ranks else 0
                agent_win_rates.append((agent_name, win_rate, wins, len(ranks)))

            # Sort by win rate (highest first)
            agent_win_rates.sort(key=lambda x: x[1], reverse=True)

            for agent_name, win_rate, wins, total in agent_win_rates:
                formatted_agent = _format_name(agent_name)
                html_parts.append(f"""
                            <li><span class="agent-name">{escape(formatted_agent)}</span>: {win_rate:.0f}% ({wins}/{total})</li>""")

            html_parts.append("""
                        </ul>
                    </div>""")

            # Kendall's W agreement metric
            kendalls_w = _calculate_kendalls_w(task_data.agent_ranks)
            w_display = f"{kendalls_w:.2f}" if kendalls_w is not None else "N/A"
            html_parts.append(f"""
                    <p class="agreement-line"><strong>Agreement (<a href="https://en.wikipedia.org/wiki/Kendall%27s_W" target="_blank">Kendall's W</a>):</strong> {w_display}</p>
                    <p class="agreement-note"><em>An agreement value of &gt;=0.33 typically means the result is significant. Smaller values may indicate either the grader is not capable of accurately ranking or the two outputs are very similar.</em></p>""")

            # Aggregate Report subsection (if available)
            if task_data.comparison_runs:
                first_run = task_data.comparison_runs[0]
                comparison_folder_name = _get_comparison_folder_name(task_name, first_run.agent_names)
                aggregate_report = _load_aggregate_report(runs_dir, comparison_folder_name)

                if aggregate_report:
                    aggregate_id = f"aggregate-{task_id}"
                    markdown_content[aggregate_id] = aggregate_report.analysis

                    html_parts.append(f"""
                    <div class="subsection">
                        <div class="subsection-header" onclick="toggleSubsection(this)">
                            <span>Aggregate Analysis <em style="font-size: 0.8em; font-weight: normal; color: var(--text-muted);">LLM-generated summary explaining why rankings occurred</em></span>
                            <span class="expand-icon">&#9654;</span>
                        </div>
                        <div class="subsection-content">
                            <div class="markdown-content" id="{aggregate_id}"></div>
                        </div>
                    </div>""")

            # Instructions subsection
            instructions_id = f"instructions-{task_id}"
            markdown_content[instructions_id] = task_data.task_info.instructions

            html_parts.append(f"""
                    <div class="subsection">
                        <div class="subsection-header" onclick="toggleSubsection(this)">
                            <span>Task Instructions <em style="font-size: 0.8em; font-weight: normal; color: var(--text-muted);">The prompt given to the agent</em></span>
                            <span class="expand-icon">&#9654;</span>
                        </div>
                        <div class="subsection-content">
                            <div class="markdown-content" id="{instructions_id}"></div>
                        </div>
                    </div>""")

            # Rankings table subsection
            html_parts.append("""
                    <div class="subsection">
                        <div class="subsection-header" onclick="toggleSubsection(this)">
                            <span>Rankings <em style="font-size: 0.8em; font-weight: normal; color: var(--text-muted);">The rankings assigned by the grader agent, repeated multiple times.</em></span>
                            <span class="expand-icon">&#9654;</span>
                        </div>
                        <div class="subsection-content">
                            <table class="rankings-table">
                                <thead>
                                    <tr>
                                        <th>Agent</th>""")

            for i in range(task_data.num_comparison_runs):
                html_parts.append(f"""
                                        <th>Comparison {i + 1}</th>""")

            html_parts.append("""
                                    </tr>
                                </thead>
                                <tbody>""")

            # Sort agents for consistent ordering
            for agent_name in sorted(task_data.agent_ranks.keys()):
                ranks = task_data.agent_ranks[agent_name]
                formatted_agent = _format_name(agent_name)
                html_parts.append(f"""
                                    <tr>
                                        <td>{escape(formatted_agent)}</td>""")

                for rank in ranks:
                    rank_class = f' class="rank-{rank}"' if rank <= 3 else ""
                    html_parts.append(f"""
                                        <td{rank_class}>{rank}</td>""")

                html_parts.append("""
                                    </tr>""")

            html_parts.append("""
                                </tbody>
                            </table>
                        </div>
                    </div>""")

            # Comparison Reasoning subsection
            html_parts.append("""
                    <div class="subsection">
                        <div class="subsection-header" onclick="toggleSubsection(this)">
                            <span>Comparison Reasoning <em style="font-size: 0.8em; font-weight: normal; color: var(--text-muted);">Final reasoning the grader gave for its assignments</em></span>
                            <span class="expand-icon">&#9654;</span>
                        </div>
                        <div class="subsection-content">""")

            for i, run in enumerate(task_data.comparison_runs):
                reasoning_id = f"reasoning-{task_id}-{i}"
                deanon_reasoning = _deanonymize_reasoning(
                    run.result.reasoning,
                    run.result.anonymous_to_index,
                    run.agent_names,
                )
                markdown_content[reasoning_id] = deanon_reasoning

                html_parts.append(f"""
                            <div class="nested-collapsible-header" onclick="toggleNestedCollapsible(this)">
                                <span>Comparison {i + 1}</span>
                                <span class="expand-icon">&#9654;</span>
                            </div>
                            <div class="nested-collapsible-content">
                                <div class="markdown-content" id="{reasoning_id}"></div>
                            </div>""")

            html_parts.append("""
                        </div>
                    </div>""")

            # Task Metadata subsection
            task_info_data = task_data.task_info.task_info_data
            difficulty = task_info_data.get("difficulty", "unknown")
            categories_str = (
                ", ".join(_format_name(cat) for cat in task_data.task_info.categories)
                if task_data.task_info.categories
                else "None"
            )

            html_parts.append(f"""
                    <div class="subsection">
                        <div class="subsection-header" onclick="toggleSubsection(this)">
                            <span>Task Metadata</span>
                            <span class="expand-icon">&#9654;</span>
                        </div>
                        <div class="subsection-content">
                            <div style="display: grid; grid-template-columns: auto 1fr; gap: 0.5rem 1rem; font-size: 0.9rem;">
                                <strong>Difficulty:</strong>
                                <span>{escape(difficulty.title() if isinstance(difficulty, str) else str(difficulty))}</span>
                                <strong>Categories:</strong>
                                <span>{escape(categories_str)}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>""")

        html_parts.append("""
        </div>""")

    # JavaScript
    markdown_json = json.dumps(markdown_content)
    html_parts.append(f"""
    </div>

    <script>
        function switchTab(event, tabId) {{
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById('tab-' + tabId).classList.add('active');
            event.currentTarget.classList.add('active');
        }}

        function toggleTaskSection(header) {{
            header.classList.toggle('expanded');
            const content = header.nextElementSibling;
            if (content) content.classList.toggle('open');
        }}

        function toggleSubsection(header) {{
            header.classList.toggle('expanded');
            const content = header.nextElementSibling;
            if (content) content.classList.toggle('open');
        }}

        function toggleNestedCollapsible(header) {{
            header.classList.toggle('expanded');
            const content = header.nextElementSibling;
            if (content) content.classList.toggle('open');
        }}

        window.addEventListener('load', function() {{
            const markdownContent = {markdown_json};
            for (const [id, content] of Object.entries(markdownContent)) {{
                const elem = document.getElementById(id);
                if (elem && typeof marked !== 'undefined') {{
                    elem.innerHTML = marked.parse(content);
                }}
            }}
        }});
    </script>
</body>
</html>""")

    return "".join(html_parts)


def create_comparison_html_report(
    runs_dir: Path,
    tasks_dir: Path,
    output_path: Path | None = None,
) -> Path:
    """
    Create an HTML report for comparison benchmark runs.

    Args:
        runs_dir: Directory containing raw_results.json
        tasks_dir: Directory containing task definitions (task.yaml, instructions.txt)
        output_path: Optional output path (default: comparison_report.html in runs_dir)

    Returns:
        Path to the generated report
    """
    results_file = runs_dir / "raw_results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    results = _load_comparison_results(results_file)
    if not results.comparison_runs:
        raise ValueError(f"No comparison results found in {results_file}")

    # Group runs by agent combination
    combo_runs = _group_runs_by_agent_combination(results)

    # Build per-task rankings for each combination
    agent_combination_data: dict[tuple[str, ...], dict[str, TaskRankingsData]] = {}
    for combo, runs in combo_runs.items():
        agent_combination_data[combo] = _build_task_rankings_data(runs, tasks_dir)

    # Generate HTML
    html_content = _generate_html(agent_combination_data, runs_dir)

    # Write to file
    if output_path is None:
        output_path = runs_dir / "comparison_report.html"

    output_path.write_text(html_content, encoding="utf-8")
    print(f"Comparison report generated: {output_path}")

    return output_path
