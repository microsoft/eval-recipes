# Copyright (c) Microsoft. All rights reserved.

"""Generate an interactive HTML dashboard for benchmark evaluation runs."""

from collections import defaultdict
from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TaskResult:
    """Results for a single task run."""

    task_name: str
    agent_name: str
    score: float
    instructions: str
    metadata: dict
    task_yaml_data: dict
    failure_report: str | None


def _load_task_results(benchmarks_output_dir: Path, tasks_directory: Path) -> list[TaskResult]:
    """
    Load all task results from benchmark output directory.

    Args:
        benchmarks_output_dir: Directory containing benchmark run outputs
        tasks_directory: Directory containing task definitions

    Returns:
        List of TaskResult objects
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

        # Load test results
        test_results_path = run_dir / "test_results.json"
        if not test_results_path.exists():
            continue

        with test_results_path.open() as f:
            test_data = json.load(f)

        score = test_data.get("score", 0.0)
        metadata = test_data.get("metadata", {})
        instructions = metadata.get("instructions", "No instructions available")

        # Load task.yaml to get all task configuration
        task_yaml_data = {}
        task_yaml_path = task_dir_path / "task.yaml"
        if task_yaml_path.exists():
            with task_yaml_path.open() as f:
                task_yaml_data = yaml.safe_load(f) or {}

        # Load failure report if it exists and score is not 100%
        failure_report = None
        if score < 100.0:
            failure_report_path = run_dir / "FAILURE_REPORT.md"
            if failure_report_path.exists():
                failure_report = failure_report_path.read_text()

        results.append(
            TaskResult(
                task_name=task_name,
                agent_name=agent_name,
                score=score,
                instructions=instructions,
                metadata=metadata,
                task_yaml_data=task_yaml_data,
                failure_report=failure_report,
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

    for result in results:
        task_info = result.task_yaml_data.get("task_info", {})
        for key, value in task_info.items():
            # Only include string and boolean fields (categorical data)
            if isinstance(value, (str, bool)):
                dimensions[key].add(str(value))

    return dict(dimensions)


def _calculate_metrics_by_dimension(results: list[TaskResult], dimension: str) -> dict[str, dict[str, float | int]]:
    """
    Calculate metrics grouped by a specific dimension from task_info.

    Args:
        results: List of TaskResult objects
        dimension: The dimension key from task_info (e.g., "difficulty", "non_deterministic_evals")

    Returns:
        Dictionary mapping dimension values to metrics (avg, count, perfect)
    """

    metrics: dict[str, dict[str, Any]] = {}

    for result in results:
        task_info = result.task_yaml_data.get("task_info", {})
        value = str(task_info.get(dimension, "unknown"))

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
    """
    Format agent name for display.

    Wrapper around _format_name for backwards compatibility.
    """
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
            margin-bottom: 2rem;
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

    # Generate tabs for each agent
    for idx, agent_name in enumerate(sorted(agent_results.keys())):
        active_class = "active" if idx == 0 else ""
        formatted_agent_name = _format_agent_name(agent_name)
        html_parts.extend(
            [
                """
            <button class="tab """,
                active_class,
                """" onclick="switchTab(event, '""",
                escape(agent_name),
                """')">""",
                escape(formatted_agent_name),
                """</button>""",
            ]
        )

    html_parts.append("""
        </div>""")

    # Generate content for each agent
    for idx, (agent_name, agent_tasks) in enumerate(sorted(agent_results.items())):
        active_class = "active" if idx == 0 else ""
        agent_id = escape(agent_name)

        # Calculate overall stats
        total_tasks = len(agent_tasks)
        avg_score = sum(t.score for t in agent_tasks) / total_tasks if total_tasks > 0 else 0.0

        html_parts.extend(
            [
                """
        <div class="tab-content """,
                active_class,
                """" id="tab-""",
                agent_id,
                """">
            <div class="overall-score">""",
                f"{avg_score:.1f}%",
                """</div>
            <div class="task-count">Average score across """,
                str(total_tasks),
                """ task(s)</div>

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
                <h3>High Level Failure Analysis</h3>
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

        for task_idx, task in enumerate(sorted_tasks):
            task_id = f"{agent_id}-task-{task_idx}"
            score_class = (
                "score-perfect" if task.score == 100.0 else "score-good" if task.score >= 50.0 else "score-poor"
            )

            formatted_task_name = _format_name(task.task_name)
            html_parts.extend(
                [
                    """
                <div class="task-item">
                    <div class="task-item-header" onclick="toggleTaskDetails(this)">
                        <div class="task-name">""",
                    escape(formatted_task_name),
                    """</div>
                        <div class="task-score """,
                    score_class,
                    """">""",
                    f"{task.score:.1f}%",
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

            # Test output (nested collapsible)
            metadata_json = json.dumps(task.metadata, indent=2)
            html_parts.extend(
                [
                    """
                        </div>
                        <div class="task-detail-section">
                            <div class="nested-collapsible-header" onclick="toggleNestedCollapsible(this)">
                                <h4>Test Output</h4>
                                <span class="collapsible-icon">▶</span>
                            </div>
                            <div class="nested-collapsible-content">
                                <pre class="json-content">""",
                    escape(metadata_json),
                    """</pre>
                            </div>
                        </div>""",
                ]
            )

            # Failure report (nested collapsible, only if exists)
            if task.failure_report:
                failure_id = f"failure-{task_id}"
                markdown_content[failure_id] = task.failure_report
                html_parts.extend(
                    [
                        """
                        <div class="task-detail-section">
                            <div class="nested-collapsible-header" onclick="toggleNestedCollapsible(this)">
                                <h4>Failure Analysis</h4>
                                <span class="collapsible-icon">▶</span>
                            </div>
                            <div class="nested-collapsible-content">
                                <div id='""",
                        failure_id,
                        """' class="markdown-content tall"></div>
                            </div>
                        </div>""",
                    ]
                )

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
