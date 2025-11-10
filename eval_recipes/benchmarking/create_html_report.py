# Copyright (c) Microsoft. All rights reserved.

from collections import defaultdict
from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TaskResult:
    """Results for a single task run (aggregated across trials)."""

    task_name: str
    agent_name: str
    score: float  # Mean score across trials
    instructions: str
    metadata: dict
    task_yaml_data: dict
    # Trial statistics
    num_trials: int = 1
    trial_scores: list[float] | None = None  # Individual trial scores
    trials: list[dict] | None = None  # Full trial data including metadata and failure_report per trial
    std_dev: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    median_score: float = 0.0
    num_perfect_trials: int = 0
    # Timing statistics
    mean_agent_duration_seconds: float = 0.0
    median_agent_duration_seconds: float = 0.0


def _load_task_results(benchmarks_output_dir: Path, tasks_directory: Path) -> list[TaskResult]:
    """
    Load all task results from benchmark output directory.

    Loads aggregated results across all trials.

    Args:
        benchmarks_output_dir: Directory containing benchmark run outputs
        tasks_directory: Directory containing task definitions

    Returns:
        List of TaskResult objects with aggregated trial statistics
    """
    results = []

    # Find all task run directories (format: {agent_name}_{task_name})
    for run_dir in benchmarks_output_dir.iterdir():
        if not run_dir.is_dir():
            continue

        # Parse directory name to get agent and task
        dir_name = run_dir.name
        agent_name = None
        task_name = None
        task_dir_path = None

        for task_dir in tasks_directory.iterdir():
            if not task_dir.is_dir():
                continue
            # Check if directory name ends with the task name
            if dir_name.endswith(f"_{task_dir.name}"):
                task_name = task_dir.name
                agent_name = dir_name[: -(len(task_name) + 1)]  # Remove _{task_name}
                task_dir_path = task_dir
                break

        if not agent_name or not task_name or not task_dir_path:
            continue

        # Load aggregated results
        aggregated_results_path = run_dir / "aggregated_results.json"
        if not aggregated_results_path.exists():
            continue

        with aggregated_results_path.open() as f:
            aggregated_data = json.load(f)

        mean_score = aggregated_data.get("mean_score", 0.0)
        median_score = aggregated_data.get("median_score", 0.0)
        std_dev = aggregated_data.get("std_dev", 0.0)
        min_score = aggregated_data.get("min_score", 0.0)
        max_score = aggregated_data.get("max_score", 0.0)
        num_trials = aggregated_data.get("num_trials", 1)
        num_perfect_trials = aggregated_data.get("num_perfect_trials", 0)

        # Extract timing data
        mean_agent_duration = aggregated_data.get("mean_agent_duration_seconds", 0.0)
        median_agent_duration = aggregated_data.get("median_agent_duration_seconds", 0.0)

        # Extract individual trial scores
        trials = aggregated_data.get("trials", [])
        trial_scores = [trial.get("score", 0.0) for trial in trials]

        # Get metadata from first trial (they should all have same instructions)
        metadata = trials[0].get("metadata", {}) if trials else {}
        instructions = metadata.get("instructions", "No instructions available")

        # Load task.yaml to get all task configuration
        task_yaml_data = {}
        task_yaml_path = task_dir_path / "task.yaml"
        if task_yaml_path.exists():
            with task_yaml_path.open() as f:
                task_yaml_data = yaml.safe_load(f) or {}

        # Attach failure reports to each trial
        for trial in trials:
            trial_num = trial.get("trial_number")
            trial_dir = run_dir / f"trial_{trial_num}"
            report_path = trial_dir / f"FAILURE_REPORT_trial_{trial_num}.md"

            if report_path.exists():
                trial["failure_report"] = report_path.read_text()
            else:
                trial["failure_report"] = None

        results.append(
            TaskResult(
                task_name=task_name,
                agent_name=agent_name,
                score=mean_score,
                instructions=instructions,
                metadata=metadata,
                task_yaml_data=task_yaml_data,
                num_trials=num_trials,
                trial_scores=trial_scores,
                trials=trials,
                std_dev=std_dev,
                min_score=min_score,
                max_score=max_score,
                median_score=median_score,
                num_perfect_trials=num_perfect_trials,
                mean_agent_duration_seconds=mean_agent_duration,
                median_agent_duration_seconds=median_agent_duration,
            )
        )

    return results


def _extract_categorical_dimensions(results: list[TaskResult]) -> dict[str, set[str]]:
    """
    Extract all categorical dimensions from task_yaml_data.

    This makes the report flexible to future task.yaml schema changes.

    Args:
        results: List of TaskResult objects

    Returns:
        Dictionary mapping dimension name to set of possible values
    """
    dimensions: dict[str, set[str]] = defaultdict(set)

    # Fields to exclude from metrics display
    exclude_fields = {"non_deterministic_evals"}

    for result in results:
        task_info = result.task_yaml_data.get("task_info", {})
        for key, value in task_info.items():
            # Skip excluded fields
            if key in exclude_fields:
                continue

            # Handle string and boolean fields
            if isinstance(value, (str, bool)):
                dimensions[key].add(str(value))
            # Handle list fields (like categories)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        dimensions[key].add(item)

    return dict(dimensions)


def _calculate_metrics_by_dimension(results: list[TaskResult], dimension: str) -> dict[str, dict[str, float | int]]:
    """
    Calculate metrics grouped by a specific dimension from task_info.

    Supports both single-value dimensions (e.g., difficulty) and multi-value
    dimensions (e.g., categories) where a task can belong to multiple values.

    Args:
        results: List of TaskResult objects
        dimension: The dimension key from task_info (e.g., "difficulty", "categories")

    Returns:
        Dictionary mapping dimension values to metrics (avg, count, perfect)
    """

    metrics: dict[str, dict[str, Any]] = {}

    for result in results:
        task_info = result.task_yaml_data.get("task_info", {})
        dimension_value = task_info.get(dimension, "unknown")

        # Handle list-based dimensions (like categories)
        values = [str(v) for v in dimension_value] if isinstance(dimension_value, list) else [str(dimension_value)]

        # Add this task's score to all its values
        for value in values:
            if value not in metrics:
                metrics[value] = {"scores": [], "count": 0, "perfect": 0}

            metrics[value]["scores"].append(result.score)
            metrics[value]["count"] += 1
            if result.score == 100.0:
                metrics[value]["perfect"] += 1

    # Calculate averages and remove scores list
    final_metrics: dict[str, dict[str, float | int]] = {}
    for value, value_metrics in metrics.items():
        scores: list[float] = value_metrics["scores"]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        final_metrics[value] = {
            "avg": avg_score,
            "count": value_metrics["count"],
            "perfect": value_metrics["perfect"],
        }

    return final_metrics


def _group_results_by_agent(results: list[TaskResult]) -> dict[str, list[TaskResult]]:
    """Group results by agent name."""
    grouped: dict[str, list[TaskResult]] = defaultdict(list)
    for result in results:
        grouped[result.agent_name].append(result)
    return dict(grouped)


def _format_name(name: str) -> str:
    """
    Format a snake_case name for display.
    """
    # Special case for specific full names
    name_lower = name.lower()
    special_names = {
        "gh_cli": "GitHub CLI",
        "amplifier_v2": "Amplifier Next",
        "amplifier_v2_aoai": "Amplifier Next AOAI",
    }
    if name_lower in special_names:
        return special_names[name_lower]

    # Split on underscores and capitalize each word
    words = name.split("_")
    formatted_words = []
    for word in words:
        word_lower = word.lower()
        # Special case for common acronyms and company names
        if word_lower == "openai":
            formatted_words.append("OpenAI")
        elif word_lower == "arxiv":
            formatted_words.append("ArXiv")
        elif word_lower == "gdpval":
            formatted_words.append("GDPVal")
        elif word_lower in ("api", "llm", "ai", "gpt", "csv", "json", "yaml", "html", "pdf", "cli", "url"):
            formatted_words.append(word.upper())
        else:
            formatted_words.append(word.capitalize())
    return " ".join(formatted_words)


def _format_agent_name(agent_name: str) -> str:
    """Format agent name for display."""
    return _format_name(agent_name)


def _generate_html(results: list[TaskResult], benchmarks_output_dir: Path) -> str:
    """Generate HTML report from task results."""
    # Load all consolidated reports (one per agent)
    consolidated_reports: dict[str, str] = {}
    for report_path in benchmarks_output_dir.glob("CONSOLIDATED_REPORT_*.md"):
        agent_name = report_path.stem.replace("CONSOLIDATED_REPORT_", "")
        consolidated_reports[agent_name] = report_path.read_text()

    # Group results by agent
    agent_results = _group_results_by_agent(results)

    # Extract categorical dimensions (flexible for future task.yaml changes)
    all_dimensions = _extract_categorical_dimensions(results)

    # Get timestamp from directory name
    timestamp = benchmarks_output_dir.name

    # Collect all markdown content to render after page load
    markdown_content: dict[str, str] = {}

    # Generate HTML
    html_parts = [
        """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent Benchmark Run Results</title>
    <script src="https://cdn.jsdelivr.net/npm/marked@9.1.6/marked.min.js" defer></script>
    <style>
        :root {
            --primary: #2563eb;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --background: #ffffff;
            --surface: #f3f4f6;
            --text: #1f2937;
            --text-muted: #6b7280;
            --border: #e5e7eb;
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            padding: 2rem;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--background);
            color: var(--text);
            line-height: 1.6;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        h1 {
            margin-top: 0;
            margin-bottom: 0.5rem;
            font-size: 2rem;
            color: var(--text);
        }

        .subtitle {
            color: var(--text-muted);
            margin-bottom: 2rem;
            font-size: 0.9rem;
        }

        /* Tab Styles */
        .tabs {
            display: flex;
            gap: 0.5rem;
            border-bottom: 2px solid var(--border);
            margin-bottom: 2rem;
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

        .tab:hover {
            color: var(--text);
            background: var(--surface);
        }

        .tab.active {
            color: var(--primary);
            border-bottom-color: var(--primary);
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        /* Summary Stats */
        .overall-score {
            font-size: 3rem;
            font-weight: 700;
            color: var(--text);
            margin-bottom: 0.5rem;
        }

        .task-count {
            color: var(--text-muted);
            font-size: 1.1rem;
            margin-bottom: 1rem;
        }

        /* Consistency Metrics */
        .consistency-metrics {
            display: flex;
            gap: 2rem;
            margin-bottom: 2rem;
            padding: 1rem 1.5rem;
            background: var(--surface);
            border-radius: 8px;
        }

        .consistency-item {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }

        .consistency-item .metric-label {
            font-size: 0.85rem;
            color: var(--text-muted);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .consistency-item .metric-value {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--text);
        }

        .consistency-item .metric-explanation {
            font-size: 0.75rem;
            color: var(--text-muted);
            font-style: italic;
        }

        /* Metrics Section */
        .metrics-section {
            margin-bottom: 2rem;
        }

        .metrics-section h2 {
            font-size: 1.5rem;
            margin-bottom: 1rem;
            color: var(--text);
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .metric-card {
            background: white;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
        }

        .metric-card h3 {
            margin: 0 0 0.75rem 0;
            font-size: 0.85rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
        }

        .metric-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--border);
        }

        .metric-item:last-child {
            border-bottom: none;
        }

        .metric-label {
            font-weight: 500;
        }

        .metric-value {
            display: flex;
            align-items: baseline;
            gap: 0.5rem;
        }

        .metric-score {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text);
        }

        .metric-count {
            font-size: 0.85rem;
            color: var(--text-muted);
        }

        /* Collapsible Sections */
        .collapsible-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 1.5rem;
            background: white;
            border: 1px solid var(--border);
            border-radius: 8px;
            cursor: pointer;
            margin-bottom: 0.5rem;
            transition: background 0.2s;
        }

        .collapsible-header:hover {
            background: var(--surface);
        }

        .collapsible-header h3 {
            margin: 0;
            font-size: 1.1rem;
            color: var(--text);
        }

        .collapsible-icon {
            color: var(--text-muted);
            transition: transform 0.2s;
            font-size: 1.2rem;
        }

        .collapsible-header.expanded .collapsible-icon {
            transform: rotate(90deg);
        }

        .collapsible-content {
            display: none;
            margin-bottom: 1.5rem;
            padding: 1.5rem;
            background: white;
            border: 1px solid var(--border);
            border-radius: 8px;
            border-top-left-radius: 0;
            border-top-right-radius: 0;
            margin-top: -0.5rem;
        }

        .collapsible-content.open {
            display: block;
        }

        /* Task List */
        .task-list {
            margin-top: 2rem;
        }

        .task-list h2 {
            font-size: 1.5rem;
            margin-bottom: 1rem;
        }

        .task-item {
            background: white;
            border: 1px solid var(--border);
            border-radius: 8px;
            margin-bottom: 1rem;
            overflow: hidden;
        }

        .task-item-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 1.5rem;
            cursor: pointer;
            transition: background 0.2s;
        }

        .task-item-header:hover {
            background: var(--surface);
        }

        .task-name {
            font-weight: 600;
            font-size: 1.1rem;
            color: var(--text);
        }

        .task-score {
            font-size: 1.5rem;
            font-weight: 600;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            min-width: 80px;
            text-align: center;
        }

        .score-perfect {
            background: #d1fae5;
            color: #065f46;
        }

        .score-good {
            background: #fed7aa;
            color: #92400e;
        }

        .score-poor {
            background: #fee2e2;
            color: #991b1b;
        }

        .task-details {
            display: none;
            padding: 1rem 1.5rem;
            border-top: 1px solid var(--border);
            background: var(--surface);
        }

        .task-details.open {
            display: block;
        }

        .task-detail-section {
            margin-bottom: 1rem;
        }

        .task-detail-section:last-child {
            margin-bottom: 0;
        }

        /* Nested Collapsible */
        .nested-collapsible-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 1rem;
            background: white;
            border: 1px solid var(--border);
            border-radius: 6px;
            cursor: pointer;
            transition: background 0.2s;
            margin-bottom: 0.25rem;
        }

        .nested-collapsible-header:hover {
            background: var(--surface);
        }

        .nested-collapsible-header h4 {
            margin: 0;
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--text);
        }

        .nested-collapsible-content {
            display: none;
            padding: 1rem;
            background: white;
            border: 1px solid var(--border);
            border-radius: 6px;
            border-top-left-radius: 0;
            border-top-right-radius: 0;
            margin-top: -0.25rem;
            margin-bottom: 0.5rem;
        }

        .nested-collapsible-content.open {
            display: block;
        }

        /* Markdown Content */
        .markdown-content {
            max-height: 400px;
            overflow-y: auto;
            font-size: 0.9rem;
            line-height: 1.6;
        }

        .markdown-content.tall {
            max-height: 600px;
        }

        .markdown-content pre {
            background: var(--surface);
            padding: 0.75rem;
            border-radius: 4px;
            overflow-x: auto;
        }

        .markdown-content code {
            background: var(--surface);
            padding: 0.2em 0.4em;
            border-radius: 3px;
            font-size: 0.85em;
        }

        .markdown-content pre code {
            background: transparent;
            padding: 0;
        }

        .markdown-content h1, .markdown-content h2, .markdown-content h3 {
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }

        .markdown-content h1 { font-size: 1.5rem; }
        .markdown-content h2 { font-size: 1.25rem; }
        .markdown-content h3 { font-size: 1.1rem; }

        .markdown-content ul, .markdown-content ol {
            padding-left: 1.5rem;
        }

        .markdown-content blockquote {
            border-left: 3px solid var(--border);
            padding-left: 1rem;
            margin-left: 0;
            color: var(--text-muted);
        }

        .markdown-content table {
            border-collapse: collapse;
            width: 100%;
            margin: 1rem 0;
        }

        .markdown-content th, .markdown-content td {
            border: 1px solid var(--border);
            padding: 0.5rem;
            text-align: left;
        }

        .markdown-content th {
            background: var(--surface);
        }

        /* JSON Content */
        .json-content {
            background: var(--surface);
            padding: 1rem;
            border-radius: 4px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.85rem;
            max-height: 400px;
            overflow-y: auto;
        }

        .no-report {
            color: var(--text-muted);
            font-style: italic;
            padding: 1rem;
        }

        /* Comparison Tables */
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 2rem;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }

        .comparison-table th,
        .comparison-table td {
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }

        .comparison-table th {
            background: var(--surface);
            font-weight: 600;
            color: var(--text);
            user-select: none;
        }

        .comparison-table tbody tr:hover {
            background: var(--surface);
        }

        .comparison-table td {
            color: var(--text);
        }

        .best-score {
            background: #d1fae5;
            font-weight: 600;
        }

        .comparison-section {
            margin-bottom: 3rem;
        }

        .comparison-section h2 {
            font-size: 1.5rem;
            margin-bottom: 1rem;
            color: var(--text);
        }

        .metric-explanations {
            background: var(--surface);
            padding: 1rem 1.5rem;
            border-radius: 8px;
            margin-top: 1rem;
            font-size: 0.9rem;
        }

        .metric-explanations h3 {
            font-size: 1rem;
            margin: 0 0 0.75rem 0;
            color: var(--text);
        }

        .metric-explanations ul {
            margin: 0;
            padding-left: 1.5rem;
            color: var(--text-muted);
        }

        .metric-explanations li {
            margin-bottom: 0.5rem;
        }

        .metric-explanations strong {
            color: var(--text);
        }

        /* Trial Tabs */
        .trial-tabs {
            display: flex;
            gap: 0.25rem;
            border-bottom: 2px solid var(--border);
            margin-bottom: 1rem;
        }

        .trial-tab {
            padding: 0.5rem 1rem;
            background: var(--surface);
            border: 1px solid var(--border);
            border-bottom: none;
            border-radius: 4px 4px 0 0;
            cursor: pointer;
            font-size: 0.9rem;
            color: var(--text-muted);
            transition: all 0.2s;
        }

        .trial-tab:hover {
            background: white;
            color: var(--text);
        }

        .trial-tab.active {
            background: white;
            color: var(--primary);
            border-color: var(--primary);
            border-bottom: 2px solid white;
            margin-bottom: -2px;
        }

        .trial-tab-contents {
            position: relative;
        }

        .trial-tab-content {
            display: none;
        }

        .trial-tab-content.active {
            display: block;
        }

        /* Clickable Links */
        .clickable-link {
            color: var(--primary);
            text-decoration: none;
            cursor: pointer;
            transition: all 0.2s;
        }

        .clickable-link:hover {
            color: #1e40af;
            text-decoration: underline;
        }

        .clickable-link:active {
            color: #1e3a8a;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Agent Benchmark Run Results</h1>
        <div class="subtitle">Run: """,
        escape(timestamp),
        """</div>

        <div class="tabs">""",
    ]

    # Calculate cross-agent comparison metrics
    agent_comparison_data = []
    all_category_scores: dict[str, dict[str, float]] = {}

    for agent_name, agent_tasks in sorted(agent_results.items()):
        total_tasks = len(agent_tasks)
        avg_score = sum(t.score for t in agent_tasks) / total_tasks if total_tasks > 0 else 0.0
        std_devs = [t.std_dev for t in agent_tasks]
        mean_std_dev = sum(std_devs) / len(std_devs) if std_devs else 0.0
        consistent_tasks = sum(1 for t in agent_tasks if t.std_dev < 10.0)
        consistency_percentage = (consistent_tasks / len(agent_tasks) * 100) if agent_tasks else 0.0

        # Calculate timing statistics
        agent_durations = [t.mean_agent_duration_seconds for t in agent_tasks if t.mean_agent_duration_seconds > 0]
        mean_agent_time = sum(agent_durations) / len(agent_durations) if agent_durations else 0.0
        median_agent_durations = [
            t.median_agent_duration_seconds for t in agent_tasks if t.median_agent_duration_seconds > 0
        ]
        median_agent_time = sum(median_agent_durations) / len(median_agent_durations) if median_agent_durations else 0.0

        agent_comparison_data.append(
            {
                "name": agent_name,
                "avg_score": avg_score,
                "mean_std_dev": mean_std_dev,
                "consistency_percentage": consistency_percentage,
                "total_tasks": total_tasks,
                "mean_agent_time_minutes": mean_agent_time / 60,
                "median_agent_time_minutes": median_agent_time / 60,
            }
        )

        # Calculate per-category scores for this agent
        if "categories" in all_dimensions:
            category_metrics = _calculate_metrics_by_dimension(agent_tasks, "categories")
            for category, metrics in category_metrics.items():
                if category not in all_category_scores:
                    all_category_scores[category] = {}
                all_category_scores[category][agent_name] = metrics["avg"]  # type: ignore[index]

    # Generate Overview tab button (first tab, active by default)
    html_parts.append("""
            <button class="tab active" onclick="switchTab(event, 'overview')">Overview</button>""")

    # Generate Tasks tab button
    html_parts.append("""
            <button class="tab" onclick="switchTab(event, 'tasks')">Tasks</button>""")

    # Generate tabs for each agent (none active since Overview is first)
    for agent_name in sorted(agent_results.keys()):
        formatted_agent_name = _format_agent_name(agent_name)
        html_parts.extend(
            [
                """
            <button class="tab" onclick="switchTab(event, '""",
                escape(agent_name),
                """')">""",
                escape(formatted_agent_name),
                """</button>""",
            ]
        )

    html_parts.append("""
        </div>""")

    # Generate Overview tab content (active by default)
    html_parts.append("""

        <div class="tab-content active" id="tab-overview">
            <h1>Agent Comparison</h1>

            <!-- Overall Performance Table -->
            <div class="comparison-section">
                <h2>Overall Performance</h2>
                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>Agent</th>
                            <th>Avg Score</th>
                            <th>Mean Variability</th>
                            <th>Consistency Rate</th>
                            <th>Avg Agent Time</th>
                            <th>Tasks Run</th>
                        </tr>
                    </thead>
                    <tbody>""")

    # Find best performer for each metric
    best_avg = max((a["avg_score"] for a in agent_comparison_data), default=0)
    best_variability = min((a["mean_std_dev"] for a in agent_comparison_data), default=float("inf"))
    best_consistency = max((a["consistency_percentage"] for a in agent_comparison_data), default=0)
    best_time = min(
        (a["mean_agent_time_minutes"] for a in agent_comparison_data if a["mean_agent_time_minutes"] > 0),
        default=float("inf"),
    )

    for agent_data in agent_comparison_data:
        agent_formatted = _format_agent_name(agent_data["name"])
        agent_id = escape(agent_data["name"])

        # Determine best score classes
        avg_class = " best-score" if agent_data["avg_score"] == best_avg else ""
        var_class = " best-score" if agent_data["mean_std_dev"] == best_variability else ""
        cons_class = " best-score" if agent_data["consistency_percentage"] == best_consistency else ""
        time_class = " best-score" if agent_data["mean_agent_time_minutes"] == best_time and best_time > 0 else ""

        html_parts.append(f"""
                        <tr>
                            <td><a href="#" class="clickable-link" onclick="event.preventDefault(); navigateToTab('{agent_id}');">{escape(agent_formatted)}</a></td>
                            <td class="{avg_class}">{agent_data["avg_score"]:.1f}%</td>
                            <td class="{var_class}">±{agent_data["mean_std_dev"]:.1f}%</td>
                            <td class="{cons_class}">{agent_data["consistency_percentage"]:.0f}%</td>
                            <td class="{time_class}">{agent_data["mean_agent_time_minutes"]:.1f}m</td>
                            <td>{agent_data["total_tasks"]}</td>
                        </tr>""")

    html_parts.append("""
                    </tbody>
                </table>
                <div class="metric-explanations">
                    <h3>Metric Descriptions</h3>
                    <ul>
                        <li><strong>Avg Score:</strong> Average score across all tasks (0-100%). Higher is better.</li>
                        <li><strong>Mean Variability:</strong> Average trial-to-trial variation across tasks. Lower values indicate more consistent and reliable task completion.</li>
                        <li><strong>Consistency Rate:</strong> Percentage of tasks where the agent showed low variability across trials of the same task (&lt;10% std dev). Higher is better.</li>
                    </ul>
                </div>
            </div>""")

    # Category Comparison Table
    if all_category_scores:
        sorted_agents = sorted(agent_results.keys())

        html_parts.append("""

            <!-- Category Comparison Table -->
            <div class="comparison-section">
                <h2>Performance by Category</h2>
                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>Category</th>""")

        for agent_name in sorted_agents:
            agent_formatted = _format_agent_name(agent_name)
            agent_id = escape(agent_name)
            html_parts.append(f"""
                            <th><a href="#" class="clickable-link" onclick="event.preventDefault(); navigateToTab('{agent_id}');" style="color: inherit; font-weight: 600;">{escape(agent_formatted)}</a></th>""")

        html_parts.append("""
                        </tr>
                    </thead>
                    <tbody>""")

        for category in sorted(all_category_scores.keys()):
            # Find best score for this category
            category_scores = all_category_scores[category]
            best_category_score = max(category_scores.values()) if category_scores else 0

            html_parts.append(f"""
                        <tr>
                            <td>{escape(_format_name(category))}</td>""")

            for agent_name in sorted_agents:
                score = category_scores.get(agent_name, 0)
                score_class = " best-score" if score == best_category_score and score > 0 else ""
                score_display = f"{score:.1f}%" if score > 0 else "—"
                html_parts.append(f"""
                            <td class="{score_class}">{score_display}</td>""")

            html_parts.append("""
                        </tr>""")

        html_parts.append("""
                    </tbody>
                </table>
            </div>""")

    # Consolidated Reports Section
    html_parts.append("""

            <!-- Consolidated Reports -->
            <div class="comparison-section">
                <h2>Identified Areas for Improvement</h2>""")

    for agent_name in sorted(agent_results.keys()):
        if agent_name in consolidated_reports:
            agent_formatted = _format_agent_name(agent_name)
            agent_id = escape(agent_name)
            overview_consolidated_id = f"overview-consolidated-{agent_id}"

            # Store report content for rendering
            markdown_content[overview_consolidated_id] = consolidated_reports[agent_name]

            html_parts.append(f"""
                <div class="collapsible-header" onclick="toggleCollapsible(this)">
                    <h3>{escape(agent_formatted)} - Consolidated Report</h3>
                    <span class="collapsible-icon">▶</span>
                </div>
                <div class="collapsible-content">
                    <div id='{overview_consolidated_id}' class="markdown-content tall"></div>
                </div>""")

    html_parts.append("""
            </div>
        </div>""")

    # Generate Tasks tab content
    # Group tasks by task_name and collect metadata
    tasks_catalog: dict[str, dict[str, Any]] = {}
    for result in results:
        if result.task_name not in tasks_catalog:
            tasks_catalog[result.task_name] = {
                "instructions": result.instructions,
                "task_yaml_data": result.task_yaml_data,
                "agent_scores": [],
            }
        tasks_catalog[result.task_name]["agent_scores"].append({"agent": result.agent_name, "score": result.score})

    html_parts.append("""

        <div class="tab-content" id="tab-tasks">
            <h1>Benchmark Tasks</h1>
            <div class="subtitle">All tasks that agents are evaluated on</div>

            <div style="background: var(--surface); padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem; line-height: 1.8;">
                <p style="margin: 0 0 1rem 0;">
                    Each task represents a real-world scenario where AI agents are given natural language instructions
                    and must autonomously complete the objective using available tools and capabilities. Tasks range from
                    creating CLI applications to document processing and data extraction workflows.
                </p>
                <p style="margin: 0 0 1rem 0;">
                    Agents execute in isolated Docker containers with pre-configured dependencies and are evaluated through
                    a combination of semantic tests (using a specialized LLM "audit" agent to assess quality) and some deterministic tests where appropriate.
                    Overall scores range from 0-100%, with higher scores indicating better task completion.
                    Each task is run multiple times (trials) to measure consistency and reliability.
                </p>
            </div>

            <div class="task-list">""")

    # Sort tasks by name
    for task_name in sorted(tasks_catalog.keys()):
        task_data = tasks_catalog[task_name]
        task_info = task_data["task_yaml_data"].get("task_info", {})
        formatted_task_name = _format_name(task_name)
        task_id = f"task-catalog-{escape(task_name)}"
        instructions_id = f"instructions-{task_id}"

        # Store instructions for markdown rendering
        markdown_content[instructions_id] = task_data["instructions"]

        # Calculate statistics
        agent_scores = task_data["agent_scores"]
        num_agents = len(agent_scores)
        avg_score = sum(s["score"] for s in agent_scores) / num_agents if num_agents > 0 else 0.0

        # Get difficulty and categories
        difficulty = task_info.get("difficulty", "unknown")
        categories = task_info.get("categories", [])
        categories_str = ", ".join(_format_name(cat) for cat in categories) if categories else "None"

        html_parts.append(f"""
                <div class="task-item" id="{task_id}">
                    <div class="task-item-header" onclick="toggleTaskDetails(this)">
                        <div class="task-name">{escape(formatted_task_name)}</div>
                        <div style="display: flex; align-items: center; gap: 1rem;">
                            <span style="color: var(--text-muted); font-size: 0.9rem;">{num_agents} agent(s) | Avg: {avg_score:.1f}%</span>
                        </div>
                    </div>
                    <div class="task-details">
                        <div class="task-detail-section">
                            <h4 style="margin: 0 0 0.5rem 0;">Metadata</h4>
                            <div style="display: grid; grid-template-columns: auto 1fr; gap: 0.5rem 1rem; font-size: 0.9rem;">
                                <strong>Difficulty:</strong>
                                <span>{escape(difficulty.title())}</span>
                                <strong>Categories:</strong>
                                <span>{escape(categories_str)}</span>
                            </div>
                        </div>
                        <div class="task-detail-section">
                            <div class="nested-collapsible-header" onclick="toggleNestedCollapsible(this)">
                                <h4>Task Instructions</h4>
                                <span class="collapsible-icon">▶</span>
                            </div>
                            <div class="nested-collapsible-content">
                                <div id='{instructions_id}' class="markdown-content"></div>
                            </div>
                        </div>
                        <div class="task-detail-section">
                            <div class="nested-collapsible-header" onclick="toggleNestedCollapsible(this)">
                                <h4>Agent Performance</h4>
                                <span class="collapsible-icon">▶</span>
                            </div>
                            <div class="nested-collapsible-content">
                                <table style="width: 100%; border-collapse: collapse;">
                                    <thead>
                                        <tr style="border-bottom: 1px solid var(--border);">
                                            <th style="padding: 0.5rem; text-align: left; font-weight: 600;">Agent</th>
                                            <th style="padding: 0.5rem; text-align: right; font-weight: 600;">Score</th>
                                        </tr>
                                    </thead>
                                    <tbody>""")

        # Sort agents by score descending
        sorted_agents = sorted(agent_scores, key=lambda x: x["score"], reverse=True)
        for agent_score in sorted_agents:
            agent_formatted = _format_agent_name(agent_score["agent"])
            agent_id = escape(agent_score["agent"])
            score = agent_score["score"]
            # Create deep link ID to the specific task within the agent's tab
            agent_task_id = f"task-{agent_id}-{escape(task_name)}"
            html_parts.append(f"""
                                        <tr style="border-bottom: 1px solid var(--border);">
                                            <td style="padding: 0.5rem;"><a href="#" class="clickable-link" onclick="event.preventDefault(); navigateToTabAndElement('{agent_id}', '{agent_task_id}');">{escape(agent_formatted)}</a></td>
                                            <td style="padding: 0.5rem; text-align: right; font-weight: 600;">{score:.1f}%</td>
                                        </tr>""")

        html_parts.append("""
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>""")

    html_parts.append("""
            </div>
        </div>""")

    # Generate content for each agent (none active since Overview is first)
    for agent_name, agent_tasks in sorted(agent_results.items()):
        agent_id = escape(agent_name)

        # Calculate overall stats
        total_tasks = len(agent_tasks)
        avg_score = sum(t.score for t in agent_tasks) / total_tasks if total_tasks > 0 else 0.0

        # Calculate consistency metrics
        std_devs = [t.std_dev for t in agent_tasks]
        mean_std_dev = sum(std_devs) / len(std_devs) if std_devs else 0.0

        # Consistency score: % of tasks with std_dev < 10%
        consistent_tasks = sum(1 for t in agent_tasks if t.std_dev < 10.0)
        consistency_percentage = (consistent_tasks / len(agent_tasks) * 100) if agent_tasks else 0.0

        # Calculate timing metrics
        agent_durations = [t.mean_agent_duration_seconds for t in agent_tasks if t.mean_agent_duration_seconds > 0]
        avg_agent_time_minutes = (sum(agent_durations) / len(agent_durations) / 60) if agent_durations else 0.0

        html_parts.extend(
            [
                """
        <div class="tab-content" id="tab-""",
                agent_id,
                """">
            <div class="overall-score">""",
                f"{avg_score:.1f}%",
                """</div>
            <div class="task-count">Average score across """,
                str(total_tasks),
                """ task(s)</div>

            <div class="consistency-metrics">
                <div class="consistency-item">
                    <span class="metric-label">Mean Variability</span>
                    <span class="metric-value">±""",
                f"{mean_std_dev:.1f}%",
                """</span>
                    <span class="metric-explanation">Average trial-to-trial variation (lower is better)</span>
                </div>
                <div class="consistency-item">
                    <span class="metric-label">Consistency Rate</span>
                    <span class="metric-value">""",
                f"{consistency_percentage:.0f}%",
                """</span>
                    <span class="metric-explanation">Tasks with &lt;10% variation (higher is better)</span>
                </div>
                <div class="consistency-item">
                    <span class="metric-label">Average Time</span>
                    <span class="metric-value">""",
                f"{avg_agent_time_minutes:.1f}m",
                """</span>
                    <span class="metric-explanation">Mean agent execution time per task</span>
                </div>
            </div>

            <div class="metrics-section">
                <h2>Metrics Breakdown</h2>
                <div class="metrics-grid">""",
            ]
        )

        # Generate metric cards for each dimension
        for dimension in sorted(all_dimensions.keys()):
            metrics = _calculate_metrics_by_dimension(agent_tasks, dimension)

            # Format dimension name for display
            dimension_display = dimension.replace("_", " ").title()

            html_parts.extend(
                [
                    """
                    <div class="metric-card">
                        <h3>By """,
                    dimension_display,
                    """</h3>""",
                ]
            )

            for value in sorted(metrics.keys()):
                value_metrics = metrics[value]
                html_parts.extend(
                    [
                        """
                        <div class="metric-item">
                            <span class="metric-label">""",
                        escape(value.title()),
                        """</span>
                            <div class="metric-value">
                                <span class="metric-score">""",
                        f"{value_metrics['avg']:.1f}%",  # type: ignore[index]
                        """</span>
                                <span class="metric-count">(""",
                        str(value_metrics["count"]),  # type: ignore[index]
                        """ tasks)</span>
                            </div>
                        </div>""",
                    ]
                )

            html_parts.append("""
                    </div>""")

        html_parts.append("""
                </div>
            </div>""")

        # Add consolidated report if it exists
        if agent_name in consolidated_reports:
            consolidated_id = f"consolidated-{agent_id}"
            report_content = consolidated_reports[agent_name]
            markdown_content[consolidated_id] = report_content
            html_parts.extend(
                [
                    """

            <div class="collapsible-header" onclick="toggleCollapsible(this)">
                <h3>Identified Areas for Improvement</h3>
                <span class="collapsible-icon">▶</span>
            </div>
            <div class="collapsible-content">
                <div id='""",
                    consolidated_id,
                    """' class="markdown-content tall"></div>
            </div>""",
                ]
            )

        # Add individual task reports
        html_parts.append("""

            <div class="task-list">
                <h2>Individual Task Reports</h2>""")

        # Sort tasks by score (ascending)
        sorted_tasks = sorted(agent_tasks, key=lambda t: t.score)

        for task in sorted_tasks:
            task_id = f"{agent_id}-{escape(task.task_name)}"
            score_class = (
                "score-perfect" if task.score == 100.0 else "score-good" if task.score >= 50.0 else "score-poor"
            )

            formatted_task_name = _format_name(task.task_name)
            task_catalog_id = f"task-catalog-{escape(task.task_name)}"

            # Format score display with trial statistics
            if task.num_trials > 1:
                score_display = f"{task.score:.1f}% ± {task.std_dev:.1f}% ({task.min_score:.1f}%-{task.max_score:.1f}%)"
            else:
                score_display = f"{task.score:.1f}%"

            html_parts.extend(
                [
                    """
                <div class="task-item" id="task-""",
                    task_id,
                    """">
                    <div class="task-item-header" onclick="toggleTaskDetails(this)">
                        <div class="task-name"><a href="#" class="clickable-link" onclick="event.preventDefault(); event.stopPropagation(); navigateToTabAndElement('tasks', '""",
                    task_catalog_id,
                    """');">""",
                    escape(formatted_task_name),
                    """</a></div>
                        <div class="task-score """,
                    score_class,
                    """">""",
                    score_display,
                    """</div>
                    </div>
                    <div class="task-details">
                        <div class="task-detail-section">""",
                ]
            )

            # Task instructions (nested collapsible)
            instructions_id = f"instructions-{task_id}"
            markdown_content[instructions_id] = task.instructions
            html_parts.extend(
                [
                    """
                            <div class="nested-collapsible-header" onclick="toggleNestedCollapsible(this)">
                                <h4>Task Instructions</h4>
                                <span class="collapsible-icon">▶</span>
                            </div>
                            <div class="nested-collapsible-content">
                                <div id='""",
                    instructions_id,
                    """' class="markdown-content"></div>
                            </div>""",
                ]
            )

            # Trial scores breakdown (if multi-trial)
            if task.num_trials > 1 and task.trial_scores:
                trial_scores_html = "<ul>"
                for idx, score in enumerate(task.trial_scores, 1):
                    # Get timing data for this trial
                    timing_str = ""
                    if task.trials and idx <= len(task.trials):
                        trial = task.trials[idx - 1]
                        agent_duration = trial.get("agent_duration_seconds")
                        test_duration = trial.get("test_duration_seconds")
                        if agent_duration is not None and test_duration is not None:
                            agent_min = agent_duration / 60
                            test_min = test_duration / 60
                            timing_str = f" (Agent: {agent_min:.1f}min, Test: {test_min:.1f}min)"
                    trial_scores_html += f"<li>Trial {idx}: {score:.1f}%{timing_str}</li>"
                trial_scores_html += "</ul>"
                trial_scores_html += (
                    f"<p><strong>Summary:</strong> {task.num_perfect_trials}/{task.num_trials} perfect trials</p>"
                )

                html_parts.extend(
                    [
                        """
                            <div class="nested-collapsible-header" onclick="toggleNestedCollapsible(this)">
                                <h4>Trial Breakdown (""",
                        str(task.num_trials),
                        """ trials)</h4>
                                <span class="collapsible-icon">▶</span>
                            </div>
                            <div class="nested-collapsible-content">
                                """,
                        trial_scores_html,
                        """
                            </div>""",
                    ]
                )

            # Test output - with tabs for multiple trials
            html_parts.append("""
                        </div>
                        <div class="task-detail-section">""")

            if task.num_trials > 1 and task.trials:
                # Multiple trials - show tabbed interface
                html_parts.append("""
                            <div class="nested-collapsible-header" onclick="toggleNestedCollapsible(this)">
                                <h4>Test Outputs by Trial</h4>
                                <span class="collapsible-icon">▶</span>
                            </div>
                            <div class="nested-collapsible-content">
                                <div class="trial-tabs">""")

                # Generate tabs for each trial
                for idx, trial in enumerate(task.trials):
                    trial_num = trial.get("trial_number", idx + 1)
                    active_class = "active" if idx == 0 else ""
                    tab_id = f"{task_id}-trial-{trial_num}"

                    html_parts.append(
                        f"""
                                    <button class="trial-tab {active_class}" onclick="switchTrialTab(event, '{tab_id}')">
                                        Trial {trial_num}
                                    </button>"""
                    )

                html_parts.append("""
                                </div>
                                <div class="trial-tab-contents">""")

                # Generate content for each trial
                for idx, trial in enumerate(task.trials):
                    trial_num = trial.get("trial_number", idx + 1)
                    active_class = "active" if idx == 0 else ""
                    tab_id = f"{task_id}-trial-{trial_num}"
                    trial_metadata = trial.get("metadata", {})
                    trial_metadata_json = json.dumps(trial_metadata, indent=2)

                    html_parts.append(
                        f"""
                                    <div class="trial-tab-content {active_class}" id="{tab_id}">
                                        <pre class="json-content">{escape(trial_metadata_json)}</pre>
                                    </div>"""
                    )

                html_parts.append("""
                                </div>
                            </div>""")
            else:
                # Single trial - show as before
                metadata_json = json.dumps(task.metadata, indent=2)
                html_parts.extend(
                    [
                        """
                            <div class="nested-collapsible-header" onclick="toggleNestedCollapsible(this)">
                                <h4>Test Output</h4>
                                <span class="collapsible-icon">▶</span>
                            </div>
                            <div class="nested-collapsible-content">
                                <pre class="json-content">""",
                        escape(metadata_json),
                        """</pre>
                            </div>""",
                    ]
                )

            html_parts.append("""
                        </div>""")

            # Failure reports - with tabs for trials
            # Check if any trial has a failure report
            if task.trials is not None:
                trials_with_reports = [trial for trial in task.trials if trial.get("failure_report")]

                if trials_with_reports:
                    # Show tabbed interface for failure reports
                    html_parts.append("""
                        <div class="task-detail-section">
                            <div class="nested-collapsible-header" onclick="toggleNestedCollapsible(this)">
                                <h4>Failure Analysis by Trial</h4>
                                <span class="collapsible-icon">▶</span>
                            </div>
                            <div class="nested-collapsible-content">""")

                    if len(trials_with_reports) > 1:
                        # Multiple trials - show tabs
                        html_parts.append("""
                                <div class="trial-tabs">""")

                        for idx, trial in enumerate(trials_with_reports):
                            trial_num = trial.get("trial_number", idx + 1)
                            active_class = "active" if idx == 0 else ""
                            tab_id = f"{task_id}-failure-trial-{trial_num}"

                            html_parts.append(
                                f"""
                                    <button class="trial-tab {active_class}" onclick="switchTrialTab(event, '{tab_id}')">
                                        Trial {trial_num}
                                    </button>"""
                            )

                        html_parts.append("""
                                </div>
                                <div class="trial-tab-contents">""")

                        # Generate content for each trial
                        for idx, trial in enumerate(trials_with_reports):
                            trial_num = trial.get("trial_number", idx + 1)
                            active_class = "active" if idx == 0 else ""
                            tab_id = f"{task_id}-failure-trial-{trial_num}"
                            failure_id = f"failure-{tab_id}"
                            markdown_content[failure_id] = trial.get("failure_report", "")

                            html_parts.append(
                                f"""
                                    <div class="trial-tab-content {active_class}" id="{tab_id}">
                                        <div id='{failure_id}' class="markdown-content tall"></div>
                                    </div>"""
                            )

                        html_parts.append("""
                                </div>""")
                    else:
                        # Single trial - no tabs needed
                        trial = trials_with_reports[0]
                        failure_id = f"failure-{task_id}"
                        markdown_content[failure_id] = trial.get("failure_report", "")
                        html_parts.append(
                            f"""
                                <div id='{failure_id}' class="markdown-content tall"></div>"""
                        )

                    html_parts.append("""
                            </div>
                        </div>""")

            html_parts.append("""
                    </div>
                </div>""")

        html_parts.append("""
            </div>
        </div>""")

    # Serialize markdown content for JavaScript
    markdown_content_json = json.dumps(markdown_content)

    # Add JavaScript section
    javascript_section = f"""
    </div>

    <script>
        function switchTab(event, agentName) {{
            // Hide all tab contents
            const tabContents = document.querySelectorAll('.tab-content');
            tabContents.forEach(content => {{
                content.classList.remove('active');
            }});

            // Remove active class from all tabs
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => {{
                tab.classList.remove('active');
            }});

            // Show selected tab content
            document.getElementById('tab-' + agentName).classList.add('active');

            // Add active class to clicked tab
            event.currentTarget.classList.add('active');
        }}

        function toggleCollapsible(header) {{
            header.classList.toggle('expanded');
            const content = header.nextElementSibling;
            if (content) {{
                content.classList.toggle('open');
            }}
        }}

        function toggleTaskDetails(header) {{
            const details = header.nextElementSibling;
            if (details) {{
                details.classList.toggle('open');
            }}
        }}

        function toggleNestedCollapsible(header) {{
            header.classList.toggle('expanded');
            const content = header.nextElementSibling;
            if (content) {{
                content.classList.toggle('open');
            }}
        }}

        function switchTrialTab(event, trialId) {{
            // Get the parent trial-tabs container
            const tabsContainer = event.currentTarget.parentElement;
            const contentsContainer = tabsContainer.nextElementSibling;

            // Hide all tab contents in this container
            const tabContents = contentsContainer.querySelectorAll('.trial-tab-content');
            tabContents.forEach(content => {{
                content.classList.remove('active');
            }});

            // Remove active class from all tabs in this container
            const tabs = tabsContainer.querySelectorAll('.trial-tab');
            tabs.forEach(tab => {{
                tab.classList.remove('active');
            }});

            // Show selected tab content
            document.getElementById(trialId).classList.add('active');

            // Add active class to clicked tab
            event.currentTarget.classList.add('active');
        }}

        // Navigate to a specific tab (agent or tasks)
        function navigateToTab(tabName) {{
            // Hide all tab contents
            const tabContents = document.querySelectorAll('.tab-content');
            tabContents.forEach(content => {{
                content.classList.remove('active');
            }});

            // Remove active class from all tabs
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => {{
                tab.classList.remove('active');
            }});

            // Show selected tab content
            const targetContent = document.getElementById('tab-' + tabName);
            if (targetContent) {{
                targetContent.classList.add('active');
            }}

            // Add active class to the corresponding tab button
            const tabButtons = document.querySelectorAll('.tab');
            tabButtons.forEach(tab => {{
                if (tab.textContent.trim() === tabName || tab.onclick?.toString().includes(tabName)) {{
                    tab.classList.add('active');
                }}
            }});
        }}

        // Navigate to a specific element and optionally expand it
        function navigateToElement(elementId, shouldExpand = true) {{
            const element = document.getElementById(elementId);
            if (element) {{
                console.log('Navigating to element:', elementId);

                // If the element is a task-item or inside one, expand its details
                if (shouldExpand) {{
                    // Check if this is a task-item or find the closest task-item parent
                    const taskItem = element.classList.contains('task-item') ? element : element.closest('.task-item');
                    if (taskItem) {{
                        console.log('Found task-item:', taskItem.id);
                        const taskDetails = taskItem.querySelector('.task-details');
                        if (taskDetails) {{
                            console.log('Found task-details, open:', taskDetails.classList.contains('open'));
                            if (!taskDetails.classList.contains('open')) {{
                                taskDetails.classList.add('open');
                                console.log('Expanded task-details');
                            }}
                        }} else {{
                            console.log('No task-details found in task-item');
                        }}
                    }} else {{
                        console.log('No task-item found');
                    }}

                    // Also handle generic collapsible sections
                    let parent = element.closest('.task-details');
                    if (parent && !parent.classList.contains('open')) {{
                        parent.classList.add('open');
                    }}
                }}

                // Small delay to let expansion animation complete before scrolling
                setTimeout(() => {{
                    // Scroll to the element with smooth behavior
                    element.scrollIntoView({{ behavior: 'smooth', block: 'start' }});

                    // Optional: Add a brief highlight effect
                    element.style.transition = 'background-color 0.5s';
                    element.style.backgroundColor = '#fef3c7';
                    setTimeout(() => {{
                        element.style.backgroundColor = '';
                    }}, 1500);
                }}, 50);
            }} else {{
                console.error('Element not found:', elementId);
            }}
        }}

        // Combined navigation: switch tab and navigate to element
        function navigateToTabAndElement(tabName, elementId) {{
            navigateToTab(tabName);
            // Delay to ensure tab content is visible before scrolling
            setTimeout(() => {{
                const element = document.getElementById(elementId);
                if (!element) {{
                    console.error('Element not found:', elementId);
                    return;
                }}
                navigateToElement(elementId);
            }}, 200);
        }}

        // Render all markdown content after page loads
        window.addEventListener('load', function() {{
            const markdownContent = {markdown_content_json};

            console.log('Rendering', Object.keys(markdownContent).length, 'markdown sections...');

            let successCount = 0;
            let failCount = 0;

            for (const [elementId, content] of Object.entries(markdownContent)) {{
                const elem = document.getElementById(elementId);
                if (elem) {{
                    try {{
                        if (typeof marked !== 'undefined' && marked.parse) {{
                            elem.innerHTML = marked.parse(content);
                            successCount++;
                        }} else {{
                            console.error('marked library not available');
                            elem.innerHTML = '<p class="no-report">Error: Markdown renderer not loaded</p>';
                            failCount++;
                        }}
                    }} catch (error) {{
                        console.error('Failed to render markdown for', elementId, ':', error);
                        elem.innerHTML = '<p class="no-report">Error rendering content</p>';
                        failCount++;
                    }}
                }} else {{
                    console.warn('Element not found:', elementId);
                    failCount++;
                }}
            }}

            console.log('Markdown rendering complete:', successCount, 'succeeded,', failCount, 'failed');
        }});
    </script>
</body>
</html>"""
    html_parts.append(javascript_section)

    return "".join(html_parts)


def create_html_report(benchmarks_output_dir: Path, tasks_directory: Path, output_path: Path | None = None) -> None:
    """
    Create an HTML report for benchmark runs.

    Args:
        benchmarks_output_dir: Directory containing benchmark run outputs
        tasks_directory: Directory containing task definitions
        output_path: Optional path to write HTML report (default: benchmark_report.html in benchmarks_output_dir)
    """
    if not benchmarks_output_dir.exists():
        raise FileNotFoundError(f"Benchmarks output directory not found: {benchmarks_output_dir}")

    if not tasks_directory.exists():
        raise FileNotFoundError(f"Tasks directory not found: {tasks_directory}")

    # Load results
    results = _load_task_results(benchmarks_output_dir, tasks_directory)

    if not results:
        raise ValueError(f"No task results found in {benchmarks_output_dir}")

    # Generate HTML
    html_content = _generate_html(results, benchmarks_output_dir)

    # Write to file
    if output_path is None:
        output_path = benchmarks_output_dir / "benchmark_report.html"

    output_path.write_text(html_content)
    print(f"Report generated: {output_path}")
