# Copyright (c) Microsoft. All rights reserved

import asyncio
from datetime import UTC, datetime
import os
from pathlib import Path
from typing import Literal, cast

import click
from dotenv import load_dotenv
from loguru import logger

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
    default=None,
    help="Run directory. If not provided, creates a new timestamped dir. If provided and exists, resumes.",
)
@click.option(
    "--agent-filter",
    "agent_filters",
    multiple=True,
    default=("name=amplifier_v1,amplifier_v2_toolkit,claude_code,openai_codex,gh_cli",),
    help="Filter agents by field. Format: field=value or field!=value. Can specify multiple times.",
)
@click.option(
    "--task-filter",
    "task_filters",
    multiple=True,
    default=("name=name!=sec_10q_extractor",),
    help="Filter tasks by field. Format: field=value or field!=value. Can specify multiple times.",
)
@click.option(
    "--max-parallel-trials",
    type=int,
    default=20,
    help="Maximum number of trials to run in parallel",
)
@click.option(
    "--num-trials",
    type=int,
    default=5,
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
    runs_dir: Path | None,
    agent_filters: tuple[str, ...],
    task_filters: tuple[str, ...],
    max_parallel_trials: int,
    num_trials: int,
    continuation_provider: str,
    continuation_model: str,
    report_score_threshold: float,
) -> None:
    if runs_dir is None:
        base_dir = Path(__file__).parents[1] / ".benchmark_results"
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
        runs_dir = base_dir / timestamp
        logger.info(f"Starting new run: {runs_dir}")
    else:
        if runs_dir.exists() and (runs_dir / "jobs.db").exists():
            logger.info(f"Resuming existing run: {runs_dir}")
        else:
            logger.info(f"Starting new run: {runs_dir}")

    harness = Harness(
        runs_dir=runs_dir,
        agents_dir=agents_dir,
        tasks_dir=tasks_dir,
        environment={
            "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
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
    asyncio.run(harness.run())


if __name__ == "__main__":
    main()
