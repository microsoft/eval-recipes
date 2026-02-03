# Copyright (c) Microsoft. All rights reserved.

"""HTML report generation for comparison-based benchmark results."""

from html import escape
import json
from pathlib import Path
import re

from eval_recipes.benchmarking.schemas import ComparisonBenchmarkManifest, ComparisonTaskMetrics


def create_comparison_html_report(
    manifest: ComparisonBenchmarkManifest,
    results_dir: Path,
    output_path: Path | None = None,
) -> Path:
    """Generate HTML report from comparison benchmark manifest.

    Args:
        manifest: Complete comparison benchmark results
        results_dir: Directory containing results (for reading markdown files)
        output_path: Optional custom output path. Defaults to results_dir/comparison_report.html

    Returns:
        Path to generated HTML report
    """
    if output_path is None:
        output_path = results_dir / "comparison_report.html"

    html_content = _generate_html(manifest)
    output_path.write_text(html_content, encoding="utf-8")

    return output_path


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
        "openai_codex": "OpenAI Codex",
    }
    name_lower = name.lower()
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
    hue = 120 * (1 - ratio)  # 120 = green, 60 = yellow, 0 = red
    saturation = 75
    lightness = 38  # Darker for better white text contrast

    # Convert HSL to RGB
    c = (1 - abs(2 * lightness / 100 - 1)) * saturation / 100
    x = c * (1 - abs((hue / 60) % 2 - 1))
    m = lightness / 100 - c / 2

    if hue < 60:
        r, g, b = c, x, 0.0
    elif hue < 120:
        r, g, b = x, c, 0.0
    else:
        r, g, b = 0.0, c, x

    r_int = int((r + m) * 255)
    g_int = int((g + m) * 255)
    b_int = int((b + m) * 255)

    return f"#{r_int:02x}{g_int:02x}{b_int:02x}"


def _generate_html(manifest: ComparisonBenchmarkManifest) -> str:
    """Generate the complete HTML report."""
    markdown_content: dict[str, str] = {}

    html_parts = [_generate_html_head(manifest.benchmark_timestamp)]

    # Generate tabs: Methodology + one tab per agent combination
    html_parts.append(_generate_tabs_header(manifest))

    # Generate Methodology tab (active by default)
    html_parts.append(_generate_methodology_tab())

    # Generate agent combination tab (contains overview + task sections)
    html_parts.append(_generate_agent_combination_tab(manifest, markdown_content))

    # Add JavaScript
    html_parts.append(_generate_javascript(markdown_content))

    return "".join(html_parts)


def _generate_html_head(timestamp: str) -> str:
    """Generate HTML head and opening body/container."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comparison Benchmark Results</title>
    <script src="https://cdn.jsdelivr.net/npm/marked@9.1.6/marked.min.js" defer></script>
    <style>
        :root {{
            --primary: #2563eb;
            --success: #10b981;
            --warning: #f59e0b;
            --background: #ffffff;
            --surface: #f3f4f6;
            --text: #1f2937;
            --text-muted: #6b7280;
            --border: #e5e7eb;
        }}

        * {{ box-sizing: border-box; }}

        body {{
            margin: 0;
            padding: 2rem;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--background);
            color: var(--text);
            line-height: 1.6;
        }}

        .container {{ max-width: 1400px; margin: 0 auto; }}

        h1 {{ margin-top: 0; margin-bottom: 0.5rem; font-size: 2rem; }}
        .subtitle {{ color: var(--text-muted); margin-bottom: 2rem; font-size: 0.9rem; }}

        /* Tabs */
        .tabs {{
            display: flex;
            gap: 0.5rem;
            border-bottom: 2px solid var(--border);
            margin-bottom: 2rem;
            flex-wrap: wrap;
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

        .tab:hover {{ color: var(--text); background: var(--surface); }}
        .tab.active {{ color: var(--primary); border-bottom-color: var(--primary); }}

        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}

        /* Task Section (collapsible within combination tab) */
        .task-section {{
            background: white;
            border: 1px solid var(--border);
            border-radius: 8px;
            margin-bottom: 1rem;
            overflow: hidden;
        }}

        .task-header {{
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 1rem 1.5rem;
            cursor: pointer;
            transition: background 0.2s;
        }}

        .task-header:hover {{ background: var(--surface); }}

        .task-name {{ font-weight: 600; font-size: 1.1rem; }}

        .expand-icon {{
            margin-left: auto;
            color: var(--text-muted);
            transition: transform 0.2s;
        }}

        .task-header.expanded .expand-icon {{ transform: rotate(90deg); }}

        /* Category Pills */
        .category-pill {{
            display: inline-block;
            padding: 0.2rem 0.6rem;
            background: #e0e7ff;
            color: #3730a3;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 500;
        }}

        /* Agent Score Badges */
        .agent-score {{
            display: inline-block;
            padding: 0.2rem 0.6rem;
            color: white;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 500;
        }}

        /* Task Content */
        .task-content {{
            display: none;
            padding: 0 1.5rem 1.5rem;
            border-top: 1px solid var(--border);
            background: var(--surface);
        }}

        .task-content.open {{ display: block; }}

        /* Subsections */
        .subsection {{
            margin-top: 1rem;
        }}

        .subsection-header {{
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
        }}

        .subsection-header:hover {{ background: #fafafa; }}

        .subsection-header.expanded .expand-icon {{ transform: rotate(90deg); }}

        .subsection-content {{
            display: none;
            padding: 1rem;
            background: white;
            border: 1px solid var(--border);
            border-top: none;
            border-radius: 0 0 6px 6px;
        }}

        .subsection-content.open {{ display: block; }}

        /* Nested Collapsibles (for reasoning) */
        .nested-collapsible-header {{
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
        }}

        .nested-collapsible-header:hover {{ background: #e9ecef; }}
        .nested-collapsible-header.expanded .expand-icon {{ transform: rotate(90deg); }}

        .nested-collapsible-content {{
            display: none;
            padding: 0.75rem;
            background: white;
            border: 1px solid var(--border);
            border-top: none;
            border-radius: 0 0 4px 4px;
            margin-top: -0.25rem;
            margin-bottom: 0.5rem;
        }}

        .nested-collapsible-content.open {{ display: block; }}

        /* Average Rank Summary */
        .avg-rank-summary {{
            padding: 0.25rem 0 0.5rem 0;
            margin-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
        }}

        .avg-rank-summary h4 {{
            margin: 0 0 0.5rem 0;
            font-size: 0.9rem;
            color: var(--text);
        }}

        .avg-rank-list {{
            margin: 0;
            padding-left: 1.5rem;
            font-size: 0.9rem;
        }}

        .avg-rank-list li {{
            margin-bottom: 0.25rem;
        }}

        .avg-rank-list .agent-name {{
            font-style: italic;
        }}

        .agreement-line {{
            margin: 0 0 0.25rem 0;
            font-size: 0.9rem;
        }}

        .agreement-note {{
            margin: 0 0 0.5rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
            font-size: 0.75rem;
            color: var(--text-muted);
        }}

        .section-heading {{
            margin: 0.75rem 0 0.25rem 0;
            font-size: 1rem;
            font-weight: 600;
            color: var(--text);
        }}

        /* Markdown Content */
        .markdown-content {{
            max-height: 400px;
            overflow-y: auto;
            font-size: 0.9rem;
            line-height: 1.6;
        }}

        .markdown-content pre {{
            background: var(--surface);
            padding: 0.75rem;
            border-radius: 4px;
            overflow-x: auto;
        }}

        /* Rankings Table */
        .rankings-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}

        .rankings-table th,
        .rankings-table td {{
            padding: 0.5rem 0.75rem;
            text-align: center;
            border: 1px solid var(--border);
        }}

        .rankings-table th {{
            background: var(--surface);
            font-weight: 600;
        }}

        .rankings-table td:first-child {{
            text-align: left;
            font-weight: 500;
        }}

        .rank-1 {{ background: #d1fae5; color: #065f46; font-weight: 600; }}
        .rank-2 {{ background: #fef3c7; color: #92400e; }}
        .rank-3 {{ background: #fee2e2; color: #991b1b; }}

        /* Overview Section */
        .overview-section {{
            background: white;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }}

        .overview-section h2 {{
            margin: 0 0 1rem 0;
            font-size: 1.25rem;
            color: var(--text);
        }}

        .overview-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1rem;
        }}

        .overview-metric {{
            background: var(--surface);
            border-radius: 6px;
            padding: 1rem;
        }}

        .overview-metric h3 {{
            margin: 0 0 0.5rem 0;
            font-size: 0.9rem;
            color: var(--text-muted);
            font-weight: 600;
        }}

        .overview-metric-list {{
            margin: 0;
            padding: 0;
            list-style: none;
        }}

        .overview-metric-list li {{
            display: flex;
            justify-content: space-between;
            padding: 0.25rem 0;
            font-size: 0.9rem;
        }}

        .overview-metric-list .agent-name {{
            font-weight: 500;
        }}

        .overview-metric-list .value {{
            color: var(--text-muted);
        }}

        .overview-single-value {{
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--text);
        }}

        .overview-single-note {{
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }}

        /* Methodology Tab Styles */
        .methodology-container {{
            width: 100%;
        }}

        .methodology-section {{
            background: white;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }}

        .methodology-section h2 {{
            margin: 0 0 1rem 0;
            font-size: 1.25rem;
            color: var(--text);
        }}

        .methodology-section p {{
            margin: 0 0 0.75rem 0;
            line-height: 1.7;
        }}

        .methodology-section ul {{
            margin: 0;
            padding-left: 1.5rem;
        }}

        .methodology-section li {{
            margin-bottom: 0.5rem;
            line-height: 1.6;
        }}

        .methodology-section code {{
            background: var(--surface);
            padding: 0.15rem 0.4rem;
            border-radius: 3px;
            font-size: 0.9em;
        }}

        .workflow-diagram {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            margin-top: 1.5rem;
            flex-wrap: wrap;
        }}

        .workflow-step {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            background: var(--surface);
            padding: 1rem 1.25rem;
            border-radius: 8px;
            border: 1px solid var(--border);
        }}

        .step-number {{
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
        }}

        .step-content {{
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }}

        .step-content strong {{
            font-size: 0.95rem;
        }}

        .step-content span {{
            font-size: 0.8rem;
            color: var(--text-muted);
        }}

        .workflow-arrow {{
            font-size: 1.5rem;
            color: var(--text-muted);
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1rem;
        }}

        .metric-card {{
            background: var(--surface);
            border-radius: 6px;
            padding: 1rem;
        }}

        .metric-card h3 {{
            margin: 0 0 0.5rem 0;
            font-size: 1rem;
            color: var(--text);
        }}

        .metric-card p {{
            margin: 0 0 0.5rem 0;
            font-size: 0.9rem;
        }}

        .metric-example {{
            font-size: 0.8rem;
            color: var(--text-muted);
            font-style: italic;
            padding: 0.5rem;
            background: white;
            border-radius: 4px;
            margin-top: 0.5rem;
        }}

        .metric-formula {{
            font-family: 'Times New Roman', serif;
            font-size: 1.1rem;
            text-align: center;
            padding: 0.75rem;
            background: white;
            border-radius: 4px;
            margin: 0.5rem 0;
        }}

        .metric-note {{
            font-size: 0.75rem;
            color: var(--text-muted);
            text-align: center;
            margin: 0 0 0.75rem 0;
        }}

        .metric-interpretation {{
            font-size: 0.85rem;
        }}

        .interp-row {{
            display: flex;
            gap: 0.75rem;
            padding: 0.35rem 0;
            border-bottom: 1px solid var(--border);
        }}

        .interp-row:last-child {{
            border-bottom: none;
        }}

        .interp-value {{
            font-weight: 600;
            min-width: 5.5rem;
            font-family: monospace;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Comparison Benchmark Results</h1>
        <div class="subtitle">Run: {escape(timestamp)}</div>

        <div class="tabs">"""


def _generate_tabs_header(manifest: ComparisonBenchmarkManifest) -> str:
    """Generate tab buttons: Methodology + agent combination."""
    # Format agent combination name (sorted alphabetically)
    formatted_names = " vs ".join(_format_name(name) for name in sorted(manifest.agent_ids))
    return f"""
            <button class="tab active" onclick="switchTab(event, 'methodology')">Methodology</button>
            <button class="tab" onclick="switchTab(event, 'combo-0')">{escape(formatted_names)}</button>
        </div>
"""


def _generate_methodology_tab() -> str:
    """Generate the Methodology tab with explanations."""
    return """
        <div class="tab-content active" id="tab-methodology">
            <div class="methodology-container">
                <div class="methodology-section">
                    <h2>Overview</h2>
                    <p>
                        This report presents results from <strong>comparison-based evaluation</strong>,
                        where multiple AI agents are evaluated on the same tasks and ranked relative to each other.
                        Rather than assigning absolute scores, a judge agent (an LLM agent with shell access) determines which agent performed better
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
        </div>
"""


def _generate_agent_combination_tab(
    manifest: ComparisonBenchmarkManifest,
    markdown_content: dict[str, str],
) -> str:
    """Generate the agent combination tab with overview and collapsible task sections."""
    html_parts = []
    overview = manifest.overview
    num_tasks = len(manifest.tasks)

    html_parts.append("""
        <div class="tab-content" id="tab-combo-0">
            <div class="overview-section">
                <h2>Overview</h2>
                <div class="overview-grid">""")

    # Average Rank (sorted by rank, best first)
    html_parts.append("""
                    <div class="overview-metric">
                        <h3 title="Mean rank across all tasks and comparison runs. Lower is better.">Average Rank</h3>
                        <ul class="overview-metric-list">""")
    sorted_avg_ranks = sorted(overview.agent_avg_rank.items(), key=lambda x: x[1])
    for agent_id, avg_rank in sorted_avg_ranks:
        html_parts.append(f"""
                            <li><span class="agent-name">{escape(_format_name(agent_id))}</span><span class="value">{avg_rank:.2f}</span></li>""")
    html_parts.append("""
                        </ul>
                    </div>""")

    # Win Rate (sorted by win rate, highest first)
    html_parts.append("""
                    <div class="overview-metric">
                        <h3 title="Percentage of comparison runs where the agent was ranked #1, across all tasks.">Win Rate over all Comparisons</h3>
                        <ul class="overview-metric-list">""")
    sorted_win_rates = sorted(overview.agent_win_rate.items(), key=lambda x: x[1], reverse=True)
    for agent_id, win_rate in sorted_win_rates:
        html_parts.append(f"""
                            <li><span class="agent-name">{escape(_format_name(agent_id))}</span><span class="value">{win_rate:.0f}%</span></li>""")
    html_parts.append("""
                        </ul>
                    </div>""")

    # Task Wins (sorted by wins, highest first)
    html_parts.append("""
                    <div class="overview-metric">
                        <h3 title="Number of tasks where the agent had the best (lowest) average rank.">Number of Overall Task Wins</h3>
                        <ul class="overview-metric-list">""")
    # Ensure all agents appear even if they have 0 wins
    task_wins_with_zeros = {agent_id: overview.agent_task_wins.get(agent_id, 0) for agent_id in manifest.agent_ids}
    sorted_task_wins = sorted(task_wins_with_zeros.items(), key=lambda x: x[1], reverse=True)
    for agent_id, wins in sorted_task_wins:
        html_parts.append(f"""
                            <li><span class="agent-name">{escape(_format_name(agent_id))}</span><span class="value">{wins}/{num_tasks}</span></li>""")
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
                        <div class="overview-single-note">Mean Kendall's W across tasks. An agreement value of &gt;=0.33 typically means the result is significant. Smaller values may indicate either the grader is not capable of accurately ranking or the two outputs are very similar.</div>
                    </div>""")

    html_parts.append("""
                </div>""")

    # Final Analysis section (if available)
    if manifest.final_analysis_report:
        final_analysis_id = "final-analysis-combo-0"
        markdown_content[final_analysis_id] = manifest.final_analysis_report

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

    # Generate collapsible task sections
    for task in sorted(manifest.tasks, key=lambda t: t.task_name):
        html_parts.append(_generate_task_section(task, markdown_content))

    html_parts.append("""
        </div>
""")

    return "".join(html_parts)


def _generate_task_section(
    task: ComparisonTaskMetrics,
    markdown_content: dict[str, str],
) -> str:
    """Generate a collapsible task section within the agent combination tab."""
    task_id = task.task_name.replace("-", "_")
    formatted_task_name = _format_name(task.task_name)
    num_agents = len(task.agent_ranks)
    num_comparison_runs = len(task.trials)

    html_parts = []

    # Task header with categories and agent score badges
    html_parts.append(f"""
            <div class="task-section">
                <div class="task-header" onclick="toggleTaskSection(this)">
                    <span class="task-name">{escape(formatted_task_name)}</span>""")

    # Category pills
    for category in task.task_info.categories:
        html_parts.append(f"""
                    <span class="category-pill">{escape(category)}</span>""")

    # Agent score badges with color gradient (sorted by avg rank, best first)
    agent_ranks_sorted = sorted(task.agent_avg_rank.items(), key=lambda x: x[1])
    for agent_id, avg_rank in agent_ranks_sorted:
        color = _get_rank_color(avg_rank, num_agents)
        formatted_agent = _format_name(agent_id)
        html_parts.append(f"""
                    <span class="agent-score" style="background: {color};">{escape(formatted_agent)}: {avg_rank:.2f}</span>""")

    html_parts.append("""
                    <span class="expand-icon">&#9654;</span>
                </div>
                <div class="task-content">
                    <h3 class="section-heading">Summary Metrics</h3>""")

    # Average rank summary
    html_parts.append("""
                    <div class="avg-rank-summary">
                        <h4>Average Ranking across comparisons (lower is better)</h4>
                        <ul class="avg-rank-list">""")
    for agent_id, avg_rank in agent_ranks_sorted:
        formatted_agent = _format_name(agent_id)
        html_parts.append(f"""
                            <li><span class="agent-name">{escape(formatted_agent)}</span>: {avg_rank:.2f}</li>""")
    html_parts.append("""
                        </ul>
                    </div>""")

    # Win rate summary
    html_parts.append("""
                    <div class="avg-rank-summary">
                        <h4>Win Rate (percentage of comparisons ranked #1)</h4>
                        <ul class="avg-rank-list">""")
    sorted_win_rates = sorted(task.agent_win_rate.items(), key=lambda x: x[1], reverse=True)
    for agent_id, win_rate in sorted_win_rates:
        total_runs = len(task.agent_ranks.get(agent_id, []))
        wins = sum(1 for r in task.agent_ranks.get(agent_id, []) if r == 1)
        formatted_agent = _format_name(agent_id)
        html_parts.append(f"""
                            <li><span class="agent-name">{escape(formatted_agent)}</span>: {win_rate:.0f}% ({wins}/{total_runs})</li>""")
    html_parts.append("""
                        </ul>
                    </div>""")

    # Kendall's W
    w_display = f"{task.agreement_kendalls_w:.2f}" if task.agreement_kendalls_w is not None else "N/A"
    html_parts.append(f"""
                    <p class="agreement-line"><strong>Agreement (<a href="https://en.wikipedia.org/wiki/Kendall%27s_W" target="_blank">Kendall's W</a>):</strong> {w_display}</p>
                    <p class="agreement-note"><em>An agreement value of &gt;=0.33 typically means the result is significant. Smaller values may indicate either the grader is not capable of accurately ranking or the two outputs are very similar.</em></p>""")

    # Aggregate Analysis subsection (if available)
    if task.aggregate_analysis:
        aggregate_id = f"aggregate-{task_id}"
        markdown_content[aggregate_id] = task.aggregate_analysis

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

    # Task Instructions subsection
    if task.task_instructions:
        instructions_id = f"instructions-{task_id}"
        markdown_content[instructions_id] = task.task_instructions

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

    for i in range(num_comparison_runs):
        html_parts.append(f"""
                                        <th>Comparison {i + 1}</th>""")

    html_parts.append("""
                                    </tr>
                                </thead>
                                <tbody>""")

    # Sort agents for consistent ordering
    for agent_id in sorted(task.agent_ranks.keys()):
        ranks = task.agent_ranks[agent_id]
        formatted_agent = _format_name(agent_id)
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

    for i, trial in enumerate(task.trials):
        reasoning_id = f"reasoning-{task_id}-{i}"
        # Format agent names in reasoning
        formatted_reasoning = trial.reasoning
        for agent_id in task.agent_ranks:
            formatted_agent = _format_name(agent_id)
            formatted_reasoning = re.sub(
                rf"\b{re.escape(agent_id)}\b",
                formatted_agent,
                formatted_reasoning,
                flags=re.IGNORECASE,
            )
        markdown_content[reasoning_id] = formatted_reasoning

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
    difficulty = task.task_info.difficulty
    categories_str = (
        ", ".join(_format_name(cat) for cat in task.task_info.categories) if task.task_info.categories else "None"
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
                                <span>{escape(difficulty.title())}</span>
                                <strong>Categories:</strong>
                                <span>{escape(categories_str)}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>""")

    return "".join(html_parts)


def _generate_javascript(markdown_content: dict[str, str]) -> str:
    """Generate JavaScript section."""
    markdown_json = json.dumps(markdown_content)
    return f"""
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
</html>"""
