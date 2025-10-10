# Copyright (c) Microsoft. All rights reserved.

"""Generate an interactive HTML dashboard for benchmark evaluation runs."""

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path

import yaml


@dataclass
class TaskResult:
    """Results for a single task run."""

    task_name: str
    agent_name: str
    score: float
    instructions: str
    difficulty: str
    non_deterministic_evals: bool
    metadata: dict
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
        # Find the task name by checking which task directory exists
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

        # Load task.yaml to get difficulty and non_deterministic_evals
        task_yaml_path = task_dir_path / "task.yaml"
        difficulty = "unknown"
        non_deterministic_evals = False
        if task_yaml_path.exists():
            with task_yaml_path.open() as f:
                task_config = yaml.safe_load(f)
                task_info = task_config.get("task_info", {})
                difficulty = task_info.get("difficulty", "unknown")
                non_deterministic_evals = task_info.get("non_deterministic_evals", False)

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
                difficulty=difficulty,
                non_deterministic_evals=non_deterministic_evals,
                metadata=metadata,
                failure_report=failure_report,
            )
        )

    return results


def _generate_html(results: list[TaskResult], benchmarks_output_dir: Path) -> str:
    """Generate HTML report from task results."""
    # Load consolidated report if it exists
    consolidated_report_path = benchmarks_output_dir / "CONSOLIDATED_REPORT.md"
    consolidated_report = None
    if consolidated_report_path.exists():
        consolidated_report = consolidated_report_path.read_text()

    # Calculate summary statistics
    total_tasks = len(results)
    perfect_tasks = sum(1 for r in results if r.score == 100.0)
    avg_score = sum(r.score for r in results) / total_tasks if total_tasks > 0 else 0.0

    # Calculate stats by difficulty
    difficulty_stats = {}
    for result in results:
        if result.difficulty not in difficulty_stats:
            difficulty_stats[result.difficulty] = {"scores": [], "count": 0, "perfect": 0}
        difficulty_stats[result.difficulty]["scores"].append(result.score)
        difficulty_stats[result.difficulty]["count"] += 1
        if result.score == 100.0:
            difficulty_stats[result.difficulty]["perfect"] += 1

    for diff in difficulty_stats:
        scores = difficulty_stats[diff]["scores"]
        difficulty_stats[diff]["avg"] = sum(scores) / len(scores) if scores else 0.0

    # Calculate stats by determinism
    deterministic_scores = [r.score for r in results if not r.non_deterministic_evals]
    non_deterministic_scores = [r.score for r in results if r.non_deterministic_evals]

    det_avg = sum(deterministic_scores) / len(deterministic_scores) if deterministic_scores else 0.0
    non_det_avg = sum(non_deterministic_scores) / len(non_deterministic_scores) if non_deterministic_scores else 0.0

    # Sort results by score (ascending) then by task name
    sorted_results = sorted(results, key=lambda r: (r.score, r.task_name))

    # Generate HTML
    html_parts = [
        """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benchmark Report</title>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
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
            max-width: 1200px;
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

        .summary-inline {
            display: flex;
            align-items: center;
            gap: 2rem;
            padding: 0.75rem 0;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }

        .summary-stat {
            display: flex;
            align-items: baseline;
            gap: 0.5rem;
        }

        .summary-stat .label {
            font-size: 0.85rem;
            color: var(--text-muted);
            font-weight: 500;
        }

        .summary-stat .value {
            font-size: 1.75rem;
            font-weight: 600;
            color: var(--text);
        }

        .breakdown-table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            border: 1px solid var(--border);
            border-radius: 6px;
            overflow: hidden;
            font-size: 0.9rem;
        }

        .breakdown-table th {
            background: var(--surface);
            padding: 0.5rem 0.75rem;
            text-align: left;
            font-weight: 600;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.025em;
            color: var(--text-muted);
            border-bottom: 1px solid var(--border);
        }

        .breakdown-table td {
            padding: 0.5rem 0.75rem;
            border-bottom: 1px solid var(--border);
        }

        .breakdown-table tbody tr:last-child td {
            border-bottom: none;
        }

        .breakdown-table tbody tr:hover {
            background: var(--surface);
        }

        .breakdown-table .score-col {
            font-weight: 600;
            font-size: 1.1rem;
        }

        .breakdown-table .secondary-col {
            color: var(--text-muted);
            font-size: 0.85rem;
        }

        .task-list {
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }

        .task-item {
            border-bottom: 1px solid var(--border);
            background: white;
        }

        .task-item:last-child {
            border-bottom: none;
        }

        .task-header {
            padding: 1rem 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
            transition: background 0.2s;
        }

        .task-header:hover {
            background: var(--surface);
        }

        .task-info {
            flex: 1;
        }

        .task-name {
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 0.25rem;
        }

        .task-agent {
            color: var(--text-muted);
            font-size: 0.9rem;
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
            padding: 0 1.5rem 1.5rem 1.5rem;
            border-top: 1px solid var(--border);
            background: var(--surface);
        }

        .task-details.open {
            display: block;
        }

        .details-section {
            margin-top: 1rem;
        }

        .details-section h4 {
            margin: 0 0 0.5rem 0;
            font-size: 0.9rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .markdown-content {
            background: white;
            padding: 1rem;
            border-radius: 4px;
            border: 1px solid var(--border);
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

        .expand-icon {
            margin-left: 1rem;
            color: var(--text-muted);
            transition: transform 0.2s;
        }

        .task-header.expanded .expand-icon {
            transform: rotate(90deg);
        }

        .no-report {
            color: var(--text-muted);
            font-style: italic;
        }

        .breakdown-header {
            display: flex;
            align-items: center;
            cursor: pointer;
            margin-top: 1.5rem;
            margin-bottom: 0.75rem;
            padding: 0.5rem 0;
            user-select: none;
        }

        .breakdown-header:hover h2 {
            color: var(--primary);
        }

        .breakdown-header h2 {
            margin: 0;
            font-size: 1.5rem;
            transition: color 0.2s;
        }

        .breakdown-expand-icon {
            margin-left: 0.5rem;
            color: var(--text-muted);
            transition: transform 0.2s;
            font-size: 1.2rem;
        }

        .breakdown-header.expanded .breakdown-expand-icon {
            transform: rotate(90deg);
        }

        .breakdown-content {
            display: none;
            overflow: hidden;
        }

        .breakdown-content.open {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Benchmark Report</h1>
        <div class="subtitle">""",
        escape(str(benchmarks_output_dir.name)),
        """</div>

        <div class="summary-inline">
            <div class="summary-stat">
                <span class="label">Total Tasks:</span>
                <span class="value">""",
        str(total_tasks),
        """</span>
            </div>
            <div class="summary-stat">
                <span class="label">Perfect Scores:</span>
                <span class="value">""",
        str(perfect_tasks),
        """</span>
            </div>
            <div class="summary-stat">
                <span class="label">Average Score:</span>
                <span class="value">""",
        f"{avg_score:.1f}%",
        """</span>
            </div>
        </div>

        <div class="breakdown-header" onclick="toggleBreakdown(this)">
            <h2>Score Breakdown</h2>
            <span class="breakdown-expand-icon">▶</span>
        </div>

        <div class="breakdown-content">
            <table class="breakdown-table">
                <thead>
                    <tr>
                        <th>Category</th>
                        <th>Avg Score</th>
                        <th>Perfect / Total</th>
                    </tr>
                </thead>
                <tbody>""",
    ]

    # Add difficulty rows
    for difficulty in sorted(difficulty_stats.keys()):
        stats = difficulty_stats[difficulty]
        html_parts.extend(
            [
                """
                    <tr>
                        <td>""",
                escape(difficulty.title()),
                """</td>
                        <td class="score-col">""",
                f"{stats['avg']:.1f}%",
                """</td>
                        <td class="secondary-col">""",
                f"{stats['perfect']} / {stats['count']}",
                """</td>
                    </tr>""",
            ]
        )

    # Add deterministic/non-deterministic rows
    html_parts.extend(
        [
            """
                    <tr>
                        <td>Deterministic</td>
                        <td class="score-col">""",
            f"{det_avg:.1f}%",
            """</td>
                        <td class="secondary-col">""",
            f"{len(deterministic_scores)} tasks",
            """</td>
                    </tr>
                    <tr>
                        <td>Non-Deterministic</td>
                        <td class="score-col">""",
            f"{non_det_avg:.1f}%",
            """</td>
                        <td class="secondary-col">""",
            f"{len(non_deterministic_scores)} tasks",
            """</td>
                    </tr>
                </tbody>
            </table>
        </div>""",
        ]
    )

    # Add consolidated report section if it exists
    if consolidated_report:
        html_parts.extend(
            [
                """

        <div class="breakdown-header" onclick="toggleBreakdown(this)">
            <h2>Consolidated Failure Analysis</h2>
            <span class="breakdown-expand-icon">▶</span>
        </div>

        <div class="breakdown-content">
            <div id="consolidated-report" class="markdown-content tall"></div>
        </div>
        <script>
            document.getElementById('consolidated-report').innerHTML = marked.parse(""",
                json.dumps(consolidated_report),
                """);
        </script>""",
            ]
        )

    html_parts.extend(
        [
            """

        <h2 style="margin-top: 1.5rem; margin-bottom: 0.75rem; font-size: 1.5rem;">All Tasks</h2>
        <div class="task-list">""",
        ]
    )

    html_parts.append("")  # Empty addition to match the list structure
    html_parts = html_parts[:-1]  # Remove the empty one we just added

    # Generate task items
    for idx, result in enumerate(sorted_results):
        score_class = (
            "score-perfect" if result.score == 100.0 else "score-good" if result.score >= 50.0 else "score-poor"
        )
        task_id = f"task-{idx}"

        html_parts.extend(
            [
                """
            <div class="task-item">
                <div class="task-header" onclick="toggleTask(this, '""",
                task_id,
                """')">
                    <div class="task-info">
                        <div class="task-name">""",
                escape(result.task_name),
                """</div>
                        <div class="task-agent">Agent: """,
                escape(result.agent_name),
                """</div>
                    </div>
                    <div class="task-score """,
                score_class,
                """">""",
                f"{result.score:.1f}%",
                """</div>
                    <span class="expand-icon">▶</span>
                </div>""",
            ]
        )

        # Add details section for all tasks
        html_parts.extend(
            [
                """
                <div class="task-details">
                    <div class="details-section">
                        <h4>Task Instructions</h4>
                        <div id="instructions-""",
                task_id,
                """" class="markdown-content"></div>
                    </div>
                    <script>
                        document.getElementById('instructions-""",
                task_id,
                """').innerHTML = marked.parse(""",
                json.dumps(result.instructions),
                """);
                    </script>""",
            ]
        )

        # Add failure report section only for failed tasks
        if result.score < 100.0:
            if result.failure_report:
                html_parts.extend(
                    [
                        """
                    <div class="details-section">
                        <h4>Failure Analysis</h4>
                        <div id="report-""",
                        task_id,
                        """" class="markdown-content tall"></div>
                    </div>
                    <script>
                        document.getElementById('report-""",
                        task_id,
                        """').innerHTML = marked.parse(""",
                        json.dumps(result.failure_report),
                        """);
                    </script>""",
                    ]
                )
            else:
                html_parts.append("""
                    <div class="details-section">
                        <p class="no-report">No failure report available</p>
                    </div>""")

        html_parts.append("""
                </div>
            </div>""")

    html_parts.append(
        """
        </div>
    </div>

    <script>
        function toggleTask(header, taskId) {
            header.classList.toggle('expanded');
            const details = header.nextElementSibling;
            if (details) {
                details.classList.toggle('open');
            }
        }

        function toggleBreakdown(header) {
            header.classList.toggle('expanded');
            const content = header.nextElementSibling;
            if (content) {
                content.classList.toggle('open');
            }
        }
    </script>
</body>
</html>"""
    )

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
