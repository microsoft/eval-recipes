# Copyright (c) Microsoft. All rights reserved

import asyncio
import os
from pathlib import Path
from typing import Literal, cast

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from eval_recipes.benchmarking.harness import Harness

load_dotenv()


@click.command()
@click.option(
    "--agents-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=lambda: Path(__file__).parents[1] / "data" / "agents",
    help="Directory containing agent configurations",
)
@click.option(
    "--tasks-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=lambda: Path(__file__).parents[1] / "data" / "tasks",
    help="Directory containing task definitions",
)
@click.option(
    "--runs-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=lambda: Path(__file__).parents[1] / ".benchmark_results",
    help="Directory to store benchmark run results",
)
@click.option(
    "--agent-filter",
    "agent_filters",
    multiple=True,
    default=("name=claude_code",),
    help="Filter agents by field. Format: field=value or field=value1,value2 or field!=value (negation). "
    "Can specify multiple times (AND logic). Examples: name=claude_code or name!=old_agent",
)
@click.option(
    "--task-filter",
    "task_filters",
    multiple=True,
    default=("name=cpsc_recall_monitor,arxiv_conclusion_extraction",),
    help="Filter tasks by field. Format: field=value or field.nested=value1,value2 or field!=value (negation). "
    "Can specify multiple times (AND logic). Examples: difficulty=medium or name!=sec_10q_extractor",
)
@click.option(
    "--generate-reports",
    is_flag=True,
    default=True,
    help="Generate failure reports for each task, a consolidated summary report, and an HTML report",
)
@click.option(
    "--max-parallel-trials",
    type=int,
    default=1,
    help="Maximum number of trials to run in parallel",
)
@click.option(
    "--num-trials",
    type=int,
    default=1,
    help="Number of times to run each task",
)
@click.option(
    "--continuation-provider",
    type=click.Choice(["openai", "azure_openai", "none"], case_sensitive=False),
    default="openai",
    help="LLM provider for agent continuation ('none' to disable)",
)
@click.option(
    "--continuation-model",
    type=click.Choice(["gpt-5", "gpt-5.1"], case_sensitive=False),
    default="gpt-5",
    help="Model to use for agent continuation decisions",
)
@click.option(
    "--report-score-threshold",
    type=float,
    default=85.0,
    help="Minimum score threshold to skip report generation (reports generated for scores below this)",
)
def main(
    agents_dir: Path,
    tasks_dir: Path,
    runs_dir: Path,
    agent_filters: tuple[str, ...],
    task_filters: tuple[str, ...],
    generate_reports: bool,
    max_parallel_trials: int,
    num_trials: int,
    continuation_provider: str,
    continuation_model: str,
    report_score_threshold: float,
) -> None:
    """Run benchmarks for LLM agents."""
    harness = Harness(
        agents_dir=agents_dir,
        tasks_dir=tasks_dir,
        runs_dir=runs_dir,
        environment={
            "ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"],
            "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
            "GITHUB_TOKEN": os.environ.get("GITHUB_TOKEN", ""),
            "AZURE_OPENAI_ENDPOINT": os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
            "AZURE_OPENAI_VERSION": os.environ.get("AZURE_OPENAI_VERSION", ""),
        },
        agent_filters=list(agent_filters) if agent_filters else None,
        task_filters=list(task_filters) if task_filters else None,
        max_parallel_trials=max_parallel_trials,
        num_trials=num_trials,
        continuation_provider=cast(Literal["openai", "azure_openai", "none"], continuation_provider),
        continuation_model=cast(Literal["gpt-5", "gpt-5.1"], continuation_model),
        report_score_threshold=report_score_threshold,
    )
    asyncio.run(harness.run(generate_reports=generate_reports))

    console = Console()
    console.print()
    console.print(
        Panel(
            "Any of the files generated in the benchmarking run may contain secrets that were used during the evaluation run. "
            "[bold red]NEVER[/bold red] commit these files to source control without first checking for exposed secrets.",
            title="[yellow]⚠ Security Warning[/yellow]",
            border_style="yellow",
        )
    )


if __name__ == "__main__":
    main()
