# Copyright (c) Microsoft. All rights reserved

import asyncio
from datetime import UTC, datetime
import os
from pathlib import Path
from typing import Literal, cast

import click
from dotenv import load_dotenv
from loguru import logger
import yaml

from eval_recipes.benchmarking.harness import Harness
from eval_recipes.benchmarking.schemas import ScoreRunSpec

load_dotenv()


@click.command()
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=lambda: Path(__file__).parents[1] / "data" / "eval-setups" / "score-default.yaml",
    help="Path to YAML config file with run definition",
)
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
    "--max-parallel-trials",
    type=int,
    default=20,
    help="Maximum number of trials to run in parallel",
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
    config_file: Path,
    agents_dir: Path,
    tasks_dir: Path,
    runs_dir: Path | None,
    max_parallel_trials: int,
    continuation_provider: str,
    continuation_model: str,
    report_score_threshold: float,
) -> None:
    # Load run definition from config file
    with config_file.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_definition = ScoreRunSpec.model_validate(config)
    logger.info(f"Loaded run definition from {config_file}")

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

    runs_dir.mkdir(parents=True, exist_ok=True)
    log_file = runs_dir / "benchmark_run_job.log"
    logger.add(log_file, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}")
    logger.info(f"Logging to {log_file}")

    harness = Harness(
        agents_dir=agents_dir,
        tasks_dir=tasks_dir,
        run_definition=run_definition,
        runs_dir=runs_dir,
        environment={
            "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
            "GITHUB_TOKEN": os.environ.get("GITHUB_TOKEN", ""),
            "AZURE_OPENAI_ENDPOINT": os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
            "AZURE_OPENAI_VERSION": os.environ.get("AZURE_OPENAI_VERSION", ""),
        },
        max_parallel_trials=max_parallel_trials,
        continuation_provider=cast(Literal["openai", "azure_openai", "none"], continuation_provider),
        continuation_model=cast(Literal["gpt-5", "gpt-5.1"], continuation_model),
        report_score_threshold=report_score_threshold,
    )
    asyncio.run(harness.run())


if __name__ == "__main__":
    main()
