# Copyright (c) Microsoft. All rights reserved.

"""HTML report generation for score-based benchmark results."""

from collections import defaultdict
from html import escape
import json
from pathlib import Path
from typing import Any

from eval_recipes.benchmarking.schemas import AgentMetrics, BenchmarkManifest, TaskMetrics


def create_html_report(
    manifest: BenchmarkManifest,
    results_dir: Path,
    output_path: Path | None = None,
) -> Path:
    """Generate HTML report from benchmark manifest.

    Args:
        manifest: Complete benchmark results
        results_dir: Directory containing results (for reading markdown files)
        output_path: Optional custom output path. Defaults to results_dir/report.html

    Returns:
        Path to generated HTML report
    """
    if output_path is None:
        output_path = results_dir / "report.html"

    html_content = _generate_html(manifest, results_dir)
    output_path.write_text(html_content, encoding="utf-8")

    return output_path


def _format_name(name: str) -> str:
    """Format a snake_case name for display."""
    name_lower = name.lower()
    special_names = {
        "gh_cli": "GitHub CLI",
        "amplifier_v1": "Amplifier Claude",
        "amplifier_v2": "Amplifier",
        "amplifier_v2_aoai": "Amplifier AOAI",
        "amplifier_v2_toolkit": "Amplifier Toolkit",
        "dev-local": "Amplifier gpt-5.1-codex-high",
    }
    if name_lower in special_names:
        return special_names[name_lower]

    words = name.split("_")
    formatted_words = []
    for word in words:
        word_lower = word.lower()
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


def _get_score_class(score: float) -> str:
    """Return CSS class for score coloring."""
    if score == 100.0:
        return "score-perfect"
    elif score >= 50.0:
        return "score-good"
    else:
        return "score-poor"


def _read_markdown_file(results_dir: Path, relative_path: str | None) -> str | None:
    """Read markdown file from relative path."""
    if not relative_path:
        return None
    full_path = results_dir / relative_path
    if full_path.exists():
        return full_path.read_text(encoding="utf-8")
    return None


def _extract_categorical_dimensions(manifest: BenchmarkManifest) -> dict[str, set[str]]:
    """Extract all categorical dimensions from task_info."""
    dimensions: dict[str, set[str]] = defaultdict(set)

    for agent in manifest.agents:
        for task in agent.tasks:
            task_info = task.task_info
            # Process difficulty
            dimensions["difficulty"].add(task_info.difficulty)
            # Process categories
            for cat in task_info.categories:
                dimensions["categories"].add(cat)

    return dict(dimensions)


def _calculate_metrics_by_dimension(
    tasks: list[TaskMetrics],
    dimension: str,
) -> dict[str, dict[str, float | int]]:
    """Calculate metrics grouped by a specific dimension."""
    metrics: dict[str, dict[str, Any]] = {}

    for task in tasks:
        task_info = task.task_info

        # Get dimension values
        if dimension == "difficulty":
            values = [task_info.difficulty]
        elif dimension == "categories":
            values = task_info.categories if task_info.categories else ["unknown"]
        else:
            values = ["unknown"]

        for value in values:
            if value not in metrics:
                metrics[value] = {"scores": [], "count": 0, "perfect": 0}
            metrics[value]["scores"].append(task.mean_score)
            metrics[value]["count"] += 1
            if task.mean_score == 100.0:
                metrics[value]["perfect"] += 1

    # Calculate averages
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


def _generate_html(manifest: BenchmarkManifest, results_dir: Path) -> str:
    """Generate HTML report from benchmark manifest."""
    # Collect markdown content for JavaScript rendering
    markdown_content: dict[str, str] = {}

    # Extract dimensions
    all_dimensions = _extract_categorical_dimensions(manifest)

    # Build agent comparison data
    agent_comparison_data = []
    for agent in manifest.agents:
        agent_comparison_data.append(
            {
                "name": agent.agent_id,
                "avg_score": agent.mean_score,
                "mean_std_dev": agent.variability,
                "consistency_percentage": agent.consistency_rate,
                "total_tasks": agent.num_unique_tasks,
                "mean_agent_time_minutes": (agent.mean_agent_duration_seconds or 0) / 60,
            }
        )

    # Build tasks catalog
    tasks_catalog: dict[str, dict[str, Any]] = {}
    for agent in manifest.agents:
        for task in agent.tasks:
            if task.task_name not in tasks_catalog:
                tasks_catalog[task.task_name] = {
                    "instructions": task.instructions,
                    "task_info": task.task_info,
                    "agent_scores": [],
                }
            tasks_catalog[task.task_name]["agent_scores"].append(
                {
                    "agent": agent.agent_id,
                    "score": task.mean_score,
                }
            )

    # Generate HTML
    html_parts = [_generate_html_head(manifest.benchmark_timestamp)]

    # Generate tabs
    html_parts.append(_generate_tabs_header(manifest.agents))

    # Generate Overview tab
    html_parts.append(
        _generate_overview_tab(
            manifest=manifest,
            agent_comparison_data=agent_comparison_data,
            results_dir=results_dir,
            markdown_content=markdown_content,
        )
    )

    # Generate Tasks tab
    html_parts.append(
        _generate_tasks_tab(
            tasks_catalog=tasks_catalog,
            markdown_content=markdown_content,
        )
    )

    # Generate Agent tabs
    for agent in sorted(manifest.agents, key=lambda a: a.agent_id):
        html_parts.append(
            _generate_agent_tab(
                agent=agent,
                all_dimensions=all_dimensions,
                results_dir=results_dir,
                markdown_content=markdown_content,
            )
        )

    # Add closing and JavaScript
    html_parts.append(_generate_javascript(markdown_content))

    return "".join(html_parts)


def _generate_html_head(timestamp: str) -> str:
    """Generate HTML head and opening body/container."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent Benchmark Run Results</title>
    <script src="https://cdn.jsdelivr.net/npm/marked@9.1.6/marked.min.js" defer></script>
    <style>
        :root {{
            --primary: #2563eb;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --background: #ffffff;
            --surface: #f3f4f6;
            --text: #1f2937;
            --text-muted: #6b7280;
            --border: #e5e7eb;
        }}

        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            padding: 2rem;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--background);
            color: var(--text);
            line-height: 1.6;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        h1 {{
            margin-top: 0;
            margin-bottom: 0.5rem;
            font-size: 2rem;
            color: var(--text);
        }}

        .subtitle {{
            color: var(--text-muted);
            margin-bottom: 2rem;
            font-size: 0.9rem;
        }}

        /* Tab Styles */
        .tabs {{
            display: flex;
            gap: 0.5rem;
            border-bottom: 2px solid var(--border);
            margin-bottom: 2rem;
        }}

        .tab {{
            padding: 0.75rem 1.5rem;
            background: transparent;
            border: none;
            border-bottom: 3px solid transparent;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 500;
            color: var(--text-muted);
            transition: all 0.2s;
        }}

        .tab:hover {{
            color: var(--text);
            background: var(--surface);
        }}

        .tab.active {{
            color: var(--primary);
            border-bottom-color: var(--primary);
        }}

        .tab-content {{
            display: none;
        }}

        .tab-content.active {{
            display: block;
        }}

        /* Summary Stats */
        .overall-score {{
            font-size: 3rem;
            font-weight: 700;
            color: var(--text);
            margin-bottom: 0.5rem;
        }}

        .task-count {{
            color: var(--text-muted);
            font-size: 1.1rem;
            margin-bottom: 1rem;
        }}

        /* Consistency Metrics */
        .consistency-metrics {{
            display: flex;
            gap: 2rem;
            margin-bottom: 2rem;
            padding: 1rem 1.5rem;
            background: var(--surface);
            border-radius: 8px;
        }}

        .consistency-item {{
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }}

        .consistency-item .metric-label {{
            font-size: 0.85rem;
            color: var(--text-muted);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .consistency-item .metric-value {{
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--text);
        }}

        .consistency-item .metric-explanation {{
            font-size: 0.75rem;
            color: var(--text-muted);
            font-style: italic;
        }}

        /* Metrics Section */
        .metrics-section {{
            margin-bottom: 2rem;
        }}

        .metrics-section h2 {{
            font-size: 1.5rem;
            margin-bottom: 1rem;
            color: var(--text);
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }}

        .metric-card {{
            background: white;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
        }}

        .metric-card h3 {{
            margin: 0 0 0.75rem 0;
            font-size: 0.85rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
        }}

        .metric-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--border);
        }}

        .metric-item:last-child {{
            border-bottom: none;
        }}

        .metric-label {{
            font-weight: 500;
        }}

        .metric-value {{
            display: flex;
            align-items: baseline;
            gap: 0.5rem;
        }}

        .metric-score {{
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text);
        }}

        .metric-count {{
            font-size: 0.85rem;
            color: var(--text-muted);
        }}

        /* Collapsible Sections */
        .collapsible-header {{
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
        }}

        .collapsible-header:hover {{
            background: var(--surface);
        }}

        .collapsible-header h3 {{
            margin: 0;
            font-size: 1.1rem;
            color: var(--text);
        }}

        .collapsible-icon {{
            color: var(--text-muted);
            transition: transform 0.2s;
            font-size: 1.2rem;
        }}

        .collapsible-header.expanded .collapsible-icon {{
            transform: rotate(90deg);
        }}

        .collapsible-content {{
            display: none;
            margin-bottom: 1.5rem;
            padding: 1.5rem;
            background: white;
            border: 1px solid var(--border);
            border-radius: 8px;
            border-top-left-radius: 0;
            border-top-right-radius: 0;
            margin-top: -0.5rem;
        }}

        .collapsible-content.open {{
            display: block;
        }}

        /* Task List */
        .task-list {{
            margin-top: 2rem;
        }}

        .task-list h2 {{
            font-size: 1.5rem;
            margin-bottom: 1rem;
        }}

        .task-item {{
            background: white;
            border: 1px solid var(--border);
            border-radius: 8px;
            margin-bottom: 1rem;
            overflow: hidden;
        }}

        .task-item-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 1.5rem;
            cursor: pointer;
            transition: background 0.2s;
        }}

        .task-item-header:hover {{
            background: var(--surface);
        }}

        .task-name {{
            font-weight: 600;
            font-size: 1.1rem;
            color: var(--text);
        }}

        .task-score {{
            font-size: 1.5rem;
            font-weight: 600;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            min-width: 80px;
            text-align: center;
        }}

        .score-perfect {{
            background: #d1fae5;
            color: #065f46;
        }}

        .score-good {{
            background: #fed7aa;
            color: #92400e;
        }}

        .score-poor {{
            background: #fee2e2;
            color: #991b1b;
        }}

        .task-details {{
            display: none;
            padding: 1rem 1.5rem;
            border-top: 1px solid var(--border);
            background: var(--surface);
        }}

        .task-details.open {{
            display: block;
        }}

        .task-detail-section {{
            margin-bottom: 1rem;
        }}

        .task-detail-section:last-child {{
            margin-bottom: 0;
        }}

        /* Nested Collapsible */
        .nested-collapsible-header {{
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
        }}

        .nested-collapsible-header:hover {{
            background: var(--surface);
        }}

        .nested-collapsible-header h4 {{
            margin: 0;
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--text);
        }}

        .nested-collapsible-content {{
            display: none;
            padding: 1rem;
            background: white;
            border: 1px solid var(--border);
            border-radius: 6px;
            border-top-left-radius: 0;
            border-top-right-radius: 0;
            margin-top: -0.25rem;
            margin-bottom: 0.5rem;
        }}

        .nested-collapsible-content.open {{
            display: block;
        }}

        /* Markdown Content */
        .markdown-content {{
            max-height: 400px;
            overflow-y: auto;
            font-size: 0.9rem;
            line-height: 1.6;
        }}

        .markdown-content.tall {{
            max-height: 600px;
        }}

        .markdown-content pre {{
            background: var(--surface);
            padding: 0.75rem;
            border-radius: 4px;
            overflow-x: auto;
        }}

        .markdown-content code {{
            background: var(--surface);
            padding: 0.2em 0.4em;
            border-radius: 3px;
            font-size: 0.85em;
        }}

        .markdown-content pre code {{
            background: transparent;
            padding: 0;
        }}

        .markdown-content h1, .markdown-content h2, .markdown-content h3 {{
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }}

        .markdown-content h1 {{ font-size: 1.5rem; }}
        .markdown-content h2 {{ font-size: 1.25rem; }}
        .markdown-content h3 {{ font-size: 1.1rem; }}

        .markdown-content ul, .markdown-content ol {{
            padding-left: 1.5rem;
        }}

        .markdown-content blockquote {{
            border-left: 3px solid var(--border);
            padding-left: 1rem;
            margin-left: 0;
            color: var(--text-muted);
        }}

        .markdown-content table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1rem 0;
        }}

        .markdown-content th, .markdown-content td {{
            border: 1px solid var(--border);
            padding: 0.5rem;
            text-align: left;
        }}

        .markdown-content th {{
            background: var(--surface);
        }}

        /* JSON Content */
        .json-content {{
            background: var(--surface);
            padding: 1rem;
            border-radius: 4px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.85rem;
            max-height: 400px;
            overflow-y: auto;
        }}

        .no-report {{
            color: var(--text-muted);
            font-style: italic;
            padding: 1rem;
        }}

        /* Comparison Tables */
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 2rem;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }}

        .comparison-table th,
        .comparison-table td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}

        .comparison-table th {{
            background: var(--surface);
            font-weight: 600;
            color: var(--text);
            user-select: none;
        }}

        .comparison-table tbody tr:hover {{
            background: var(--surface);
        }}

        .comparison-table td {{
            color: var(--text);
        }}

        .best-score {{
            background: #d1fae5;
            font-weight: 600;
        }}

        .comparison-section {{
            margin-bottom: 3rem;
        }}

        .comparison-section h2 {{
            font-size: 1.5rem;
            margin-bottom: 1rem;
            color: var(--text);
        }}

        .metric-explanations {{
            background: var(--surface);
            padding: 1rem 1.5rem;
            border-radius: 8px;
            margin-top: 1rem;
            font-size: 0.9rem;
        }}

        .metric-explanations h3 {{
            font-size: 1rem;
            margin: 0 0 0.75rem 0;
            color: var(--text);
        }}

        .metric-explanations ul {{
            margin: 0;
            padding-left: 1.5rem;
            color: var(--text-muted);
        }}

        .metric-explanations li {{
            margin-bottom: 0.5rem;
        }}

        .metric-explanations strong {{
            color: var(--text);
        }}

        /* Trial Tabs */
        .trial-tabs {{
            display: flex;
            gap: 0.25rem;
            border-bottom: 2px solid var(--border);
            margin-bottom: 1rem;
        }}

        .trial-tab {{
            padding: 0.5rem 1rem;
            background: var(--surface);
            border: 1px solid var(--border);
            border-bottom: none;
            border-radius: 4px 4px 0 0;
            cursor: pointer;
            font-size: 0.9rem;
            color: var(--text-muted);
            transition: all 0.2s;
        }}

        .trial-tab:hover {{
            background: white;
            color: var(--text);
        }}

        .trial-tab.active {{
            background: white;
            color: var(--primary);
            border-color: var(--primary);
            border-bottom: 2px solid white;
            margin-bottom: -2px;
        }}

        .trial-tab-contents {{
            position: relative;
        }}

        .trial-tab-content {{
            display: none;
        }}

        .trial-tab-content.active {{
            display: block;
        }}

        /* Clickable Links */
        .clickable-link {{
            color: var(--primary);
            text-decoration: none;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .clickable-link:hover {{
            color: #1e40af;
            text-decoration: underline;
        }}

        .clickable-link:active {{
            color: #1e3a8a;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Agent Benchmark Run Results</h1>
        <div class="subtitle">Run: {escape(timestamp)}</div>
"""


def _generate_tabs_header(agents: list[AgentMetrics]) -> str:
    """Generate the tab buttons."""
    html_parts = ['        <div class="tabs">']
    html_parts.append(
        '            <button class="tab active" onclick="switchTab(event, \'overview\')">Overview</button>'
    )
    html_parts.append('            <button class="tab" onclick="switchTab(event, \'tasks\')">Tasks</button>')

    for agent in sorted(agents, key=lambda a: a.agent_id):
        formatted_name = _format_name(agent.agent_id)
        html_parts.append(
            f'            <button class="tab" onclick="switchTab(event, \'{escape(agent.agent_id)}\')">'
            f"{escape(formatted_name)}</button>"
        )

    html_parts.append("        </div>")
    return "\n".join(html_parts)


def _generate_overview_tab(
    manifest: BenchmarkManifest,
    agent_comparison_data: list[dict[str, Any]],
    results_dir: Path,
    markdown_content: dict[str, str],
) -> str:
    """Generate the Overview tab content."""
    html_parts = [
        """
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
                    <tbody>"""
    ]

    # Find best performers
    best_avg = max((a["avg_score"] for a in agent_comparison_data), default=0)
    best_variability = min((a["mean_std_dev"] for a in agent_comparison_data), default=float("inf"))
    best_consistency = max((a["consistency_percentage"] for a in agent_comparison_data), default=0)
    best_time = min(
        (a["mean_agent_time_minutes"] for a in agent_comparison_data if a["mean_agent_time_minutes"] > 0),
        default=float("inf"),
    )

    for agent_data in agent_comparison_data:
        agent_formatted = _format_name(agent_data["name"])
        agent_id = escape(agent_data["name"])

        avg_class = " best-score" if agent_data["avg_score"] == best_avg else ""
        var_class = " best-score" if agent_data["mean_std_dev"] == best_variability else ""
        cons_class = " best-score" if agent_data["consistency_percentage"] == best_consistency else ""
        time_class = " best-score" if agent_data["mean_agent_time_minutes"] == best_time and best_time > 0 else ""

        html_parts.append(f"""
                        <tr>
                            <td><a href="#" class="clickable-link" onclick="event.preventDefault(); navigateToTab('{agent_id}');">{escape(agent_formatted)}</a></td>
                            <td class="{avg_class}">{agent_data["avg_score"]:.1f}%</td>
                            <td class="{var_class}">&plusmn;{agent_data["mean_std_dev"]:.1f}%</td>
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

    # Consolidated Reports Section
    html_parts.append("""

            <!-- Consolidated Reports -->
            <div class="comparison-section">
                <h2>Identified Areas for Improvement</h2>""")

    for agent in sorted(manifest.agents, key=lambda a: a.agent_id):
        if agent.full_report_path:
            report_content = _read_markdown_file(results_dir, agent.full_report_path)
            if report_content:
                agent_formatted = _format_name(agent.agent_id)
                overview_consolidated_id = f"overview-consolidated-{escape(agent.agent_id)}"
                markdown_content[overview_consolidated_id] = report_content

                html_parts.append(f"""
                <div class="collapsible-header" onclick="toggleCollapsible(this)">
                    <h3>{escape(agent_formatted)} - Consolidated Report</h3>
                    <span class="collapsible-icon">&#9654;</span>
                </div>
                <div class="collapsible-content">
                    <div id='{overview_consolidated_id}' class="markdown-content tall"></div>
                </div>""")

    html_parts.append("""
            </div>
        </div>""")

    return "".join(html_parts)


def _generate_tasks_tab(
    tasks_catalog: dict[str, dict[str, Any]],
    markdown_content: dict[str, str],
) -> str:
    """Generate the Tasks tab content."""
    html_parts = [
        """

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

            <div class="task-list">"""
    ]

    for task_name in sorted(tasks_catalog.keys()):
        task_data = tasks_catalog[task_name]
        task_info = task_data["task_info"]
        formatted_task_name = _format_name(task_name)
        task_id = f"task-catalog-{escape(task_name)}"
        instructions_id = f"instructions-{task_id}"

        markdown_content[instructions_id] = task_data["instructions"]

        agent_scores = task_data["agent_scores"]
        num_agents = len(agent_scores)
        avg_score = sum(s["score"] for s in agent_scores) / num_agents if num_agents > 0 else 0.0

        difficulty = task_info.difficulty
        categories = task_info.categories
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
                                <span class="collapsible-icon">&#9654;</span>
                            </div>
                            <div class="nested-collapsible-content">
                                <div id='{instructions_id}' class="markdown-content"></div>
                            </div>
                        </div>
                        <div class="task-detail-section">
                            <div class="nested-collapsible-header" onclick="toggleNestedCollapsible(this)">
                                <h4>Agent Performance</h4>
                                <span class="collapsible-icon">&#9654;</span>
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

        sorted_agents = sorted(agent_scores, key=lambda x: x["score"], reverse=True)
        for agent_score in sorted_agents:
            agent_formatted = _format_name(agent_score["agent"])
            agent_id = escape(agent_score["agent"])
            score = agent_score["score"]
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

    return "".join(html_parts)


def _generate_agent_tab(
    agent: AgentMetrics,
    all_dimensions: dict[str, set[str]],
    results_dir: Path,
    markdown_content: dict[str, str],
) -> str:
    """Generate tab content for a single agent."""
    agent_id = escape(agent.agent_id)
    avg_agent_time_minutes = (agent.mean_agent_duration_seconds or 0) / 60

    html_parts = [
        f"""
        <div class="tab-content" id="tab-{agent_id}">
            <div class="overall-score">{agent.mean_score:.1f}%</div>
            <div class="task-count">Average score across {agent.num_unique_tasks} task(s)</div>

            <div class="consistency-metrics">
                <div class="consistency-item">
                    <span class="metric-label">Mean Variability</span>
                    <span class="metric-value">&plusmn;{agent.variability:.1f}%</span>
                    <span class="metric-explanation">Average trial-to-trial variation (lower is better)</span>
                </div>
                <div class="consistency-item">
                    <span class="metric-label">Consistency Rate</span>
                    <span class="metric-value">{agent.consistency_rate:.0f}%</span>
                    <span class="metric-explanation">Tasks with &lt;10% variation (higher is better)</span>
                </div>
                <div class="consistency-item">
                    <span class="metric-label">Average Time</span>
                    <span class="metric-value">{avg_agent_time_minutes:.1f}m</span>
                    <span class="metric-explanation">Mean agent execution time per task</span>
                </div>
            </div>

            <div class="metrics-section">
                <h2>Metrics Breakdown</h2>
                <div class="metrics-grid">"""
    ]

    # Generate metric cards for each dimension
    for dimension in sorted(all_dimensions.keys()):
        metrics = _calculate_metrics_by_dimension(agent.tasks, dimension)
        dimension_display = dimension.replace("_", " ").title()

        html_parts.append(f"""
                    <div class="metric-card">
                        <h3>By {dimension_display}</h3>""")

        for value in sorted(metrics.keys()):
            value_metrics = metrics[value]
            html_parts.append(f"""
                        <div class="metric-item">
                            <span class="metric-label">{escape(value.title())}</span>
                            <div class="metric-value">
                                <span class="metric-score">{value_metrics["avg"]:.1f}%</span>
                                <span class="metric-count">({value_metrics["count"]} tasks)</span>
                            </div>
                        </div>""")

        html_parts.append("""
                    </div>""")

    html_parts.append("""
                </div>
            </div>""")

    # Add consolidated report if it exists
    if agent.full_report_path:
        report_content = _read_markdown_file(results_dir, agent.full_report_path)
        if report_content:
            consolidated_id = f"consolidated-{agent_id}"
            markdown_content[consolidated_id] = report_content
            html_parts.append(f"""

            <div class="collapsible-header" onclick="toggleCollapsible(this)">
                <h3>Identified Areas for Improvement</h3>
                <span class="collapsible-icon">&#9654;</span>
            </div>
            <div class="collapsible-content">
                <div id='{consolidated_id}' class="markdown-content tall"></div>
            </div>""")

    # Add individual task reports
    html_parts.append("""

            <div class="task-list">
                <h2>Individual Task Reports</h2>""")

    # Sort tasks by score (ascending)
    sorted_tasks = sorted(agent.tasks, key=lambda t: t.mean_score)

    for task in sorted_tasks:
        task_id = f"{agent_id}-{escape(task.task_name)}"
        score_class = _get_score_class(task.mean_score)
        formatted_task_name = _format_name(task.task_name)
        task_catalog_id = f"task-catalog-{escape(task.task_name)}"

        # Format score display with trial statistics
        if task.num_trials > 1:
            score_display = (
                f"{task.mean_score:.1f}% &plusmn; {task.std_dev:.1f}% ({task.min_score:.1f}%-{task.max_score:.1f}%)"
            )
        else:
            score_display = f"{task.mean_score:.1f}%"

        html_parts.append(f"""
                <div class="task-item" id="task-{task_id}">
                    <div class="task-item-header" onclick="toggleTaskDetails(this)">
                        <div class="task-name"><a href="#" class="clickable-link" onclick="event.preventDefault(); event.stopPropagation(); navigateToTabAndElement('tasks', '{task_catalog_id}');">{escape(formatted_task_name)}</a></div>
                        <div class="task-score {score_class}">{score_display}</div>
                    </div>
                    <div class="task-details">
                        <div class="task-detail-section">""")

        # Task instructions
        instructions_id = f"instructions-{task_id}"
        markdown_content[instructions_id] = task.instructions
        html_parts.append(f"""
                            <div class="nested-collapsible-header" onclick="toggleNestedCollapsible(this)">
                                <h4>Task Instructions</h4>
                                <span class="collapsible-icon">&#9654;</span>
                            </div>
                            <div class="nested-collapsible-content">
                                <div id='{instructions_id}' class="markdown-content"></div>
                            </div>""")

        # Trial scores breakdown
        if task.num_trials > 1:
            trial_scores_html = "<ul>"
            for trial in task.trials:
                timing_str = ""
                if trial.agent_duration_seconds is not None and trial.test_duration_seconds is not None:
                    agent_min = trial.agent_duration_seconds / 60
                    test_min = trial.test_duration_seconds / 60
                    timing_str = f" (Agent: {agent_min:.1f}min, Test: {test_min:.1f}min)"
                trial_scores_html += f"<li>Trial {trial.trial_number}: {trial.score:.1f}%{timing_str}</li>"
            trial_scores_html += "</ul>"
            trial_scores_html += (
                f"<p><strong>Summary:</strong> {task.num_perfect_trials}/{task.num_trials} perfect trials</p>"
            )

            html_parts.append(f"""
                            <div class="nested-collapsible-header" onclick="toggleNestedCollapsible(this)">
                                <h4>Trial Breakdown ({task.num_trials} trials)</h4>
                                <span class="collapsible-icon">&#9654;</span>
                            </div>
                            <div class="nested-collapsible-content">
                                {trial_scores_html}
                            </div>""")

        # Test output section
        html_parts.append("""
                        </div>
                        <div class="task-detail-section">""")

        if task.num_trials > 1:
            # Multiple trials - show tabbed interface
            html_parts.append("""
                            <div class="nested-collapsible-header" onclick="toggleNestedCollapsible(this)">
                                <h4>Test Outputs by Trial</h4>
                                <span class="collapsible-icon">&#9654;</span>
                            </div>
                            <div class="nested-collapsible-content">
                                <div class="trial-tabs">""")

            for idx, trial in enumerate(task.trials):
                active_class = "active" if idx == 0 else ""
                tab_id = f"{task_id}-trial-{trial.trial_number}"
                html_parts.append(f"""
                                    <button class="trial-tab {active_class}" onclick="switchTrialTab(event, '{tab_id}')">
                                        Trial {trial.trial_number}
                                    </button>""")

            html_parts.append("""
                                </div>
                                <div class="trial-tab-contents">""")

            for idx, trial in enumerate(task.trials):
                active_class = "active" if idx == 0 else ""
                tab_id = f"{task_id}-trial-{trial.trial_number}"
                rubric_json = json.dumps(trial.rubric, indent=2)

                html_parts.append(f"""
                                    <div class="trial-tab-content {active_class}" id="{tab_id}">
                                        <pre class="json-content">{escape(rubric_json)}</pre>
                                    </div>""")

            html_parts.append("""
                                </div>
                            </div>""")
        else:
            # Single trial
            rubric = task.trials[0].rubric if task.trials else {}
            rubric_json = json.dumps(rubric, indent=2)
            html_parts.append(f"""
                            <div class="nested-collapsible-header" onclick="toggleNestedCollapsible(this)">
                                <h4>Test Output</h4>
                                <span class="collapsible-icon">&#9654;</span>
                            </div>
                            <div class="nested-collapsible-content">
                                <pre class="json-content">{escape(rubric_json)}</pre>
                            </div>""")

        html_parts.append("""
                        </div>""")

        # Failure reports
        trials_with_reports = [t for t in task.trials if t.failure_report_path]
        if trials_with_reports:
            html_parts.append("""
                        <div class="task-detail-section">
                            <div class="nested-collapsible-header" onclick="toggleNestedCollapsible(this)">
                                <h4>Failure Analysis by Trial</h4>
                                <span class="collapsible-icon">&#9654;</span>
                            </div>
                            <div class="nested-collapsible-content">""")

            if len(trials_with_reports) > 1:
                # Multiple trials with reports - show tabs
                html_parts.append("""
                                <div class="trial-tabs">""")

                for idx, trial in enumerate(trials_with_reports):
                    active_class = "active" if idx == 0 else ""
                    tab_id = f"{task_id}-failure-trial-{trial.trial_number}"
                    html_parts.append(f"""
                                    <button class="trial-tab {active_class}" onclick="switchTrialTab(event, '{tab_id}')">
                                        Trial {trial.trial_number}
                                    </button>""")

                html_parts.append("""
                                </div>
                                <div class="trial-tab-contents">""")

                for idx, trial in enumerate(trials_with_reports):
                    active_class = "active" if idx == 0 else ""
                    tab_id = f"{task_id}-failure-trial-{trial.trial_number}"
                    failure_id = f"failure-{tab_id}"
                    failure_content = _read_markdown_file(results_dir, trial.failure_report_path)
                    if failure_content:
                        markdown_content[failure_id] = failure_content

                    html_parts.append(f"""
                                    <div class="trial-tab-content {active_class}" id="{tab_id}">
                                        <div id='{failure_id}' class="markdown-content tall"></div>
                                    </div>""")

                html_parts.append("""
                                </div>""")
            else:
                # Single trial with report
                trial = trials_with_reports[0]
                failure_id = f"failure-{task_id}"
                failure_content = _read_markdown_file(results_dir, trial.failure_report_path)
                if failure_content:
                    markdown_content[failure_id] = failure_content
                html_parts.append(f"""
                                <div id='{failure_id}' class="markdown-content tall"></div>""")

            html_parts.append("""
                            </div>
                        </div>""")

        html_parts.append("""
                    </div>
                </div>""")

    html_parts.append("""
            </div>
        </div>""")

    return "".join(html_parts)


def _generate_javascript(markdown_content: dict[str, str]) -> str:
    """Generate JavaScript section."""
    markdown_content_json = json.dumps(markdown_content)

    return f"""
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
