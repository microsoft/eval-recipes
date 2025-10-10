# Copyright (c) Microsoft. All rights reserved

import asyncio
import os
from pathlib import Path

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
    default=lambda: Path(__file__).parents[1] / "data" / "benchmarking" / "runs",
    help="Directory to store benchmark run results",
)
@click.option(
    "--agent-filter",
    "agent_filters",
    multiple=True,
    default=(),
    help="Filter agents by field. Format: field=value or field=value1,value2. "
    "Can specify multiple times (AND logic). Examples: name=claude_code",
)
@click.option(
    "--task-filter",
    "task_filters",
    multiple=True,
    default=None,
    help="Filter tasks by field. Format: field=value or field.nested=value1,value2. "
    "Can specify multiple times (AND logic). Examples: difficulty=medium, "
    "name=email_drafting,arxiv_conclusion_extraction, task_info.non_deterministic_evals=true. ",
)
@click.option(
    "--generate-reports",
    is_flag=True,
    default=True,
    help="Generate failure reports for each task, a consolidated summary report, and an HTML report",
)
@click.option(
    "--max-parallel-tasks",
    type=int,
    default=5,
    help="Maximum number of tasks to run in parallel",
)
def main(
    agents_dir: Path,
    tasks_dir: Path,
    runs_dir: Path,
    agent_filters: tuple[str, ...],
    task_filters: tuple[str, ...],
    generate_reports: bool,
    max_parallel_tasks: int,
) -> None:
    """Run benchmarks for LLM agents."""
    harness = Harness(
        agents_dir=agents_dir,
        tasks_dir=tasks_dir,
        runs_dir=runs_dir,
        environment={
            "ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"],
            "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
        },
        agent_filters=list(agent_filters) if agent_filters else None,
        task_filters=list(task_filters) if task_filters else None,
        max_parallel_tasks=max_parallel_tasks,
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
