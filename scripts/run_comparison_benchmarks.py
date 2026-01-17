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

from eval_recipes.benchmarking.create_comparison_html_report import create_comparison_html_report
from eval_recipes.benchmarking.harness_comparison import ComparisonHarness
from eval_recipes.benchmarking.schemas import ComparisonTaskSpec

load_dotenv()


@click.command()
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=lambda: Path(__file__).parents[1] / "data" / "eval-setups" / "comparison-default.yaml",
    help="Path to YAML config file with comparison specs",
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
    help="Run directory. If not provided, creates a new timestamped dir.",
)
@click.option(
    "--max-parallel",
    type=int,
    default=11,
    help="Maximum number of trials to run in parallel",
)
@click.option(
    "--comparison-runs",
    type=int,
    default=7,
    help="Number of times to run semantic_test_comparison per trial set (for uncertainty measurement)",
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
def main(
    config_file: Path,
    agents_dir: Path,
    tasks_dir: Path,
    runs_dir: Path | None,
    max_parallel: int,
    comparison_runs: int,
    continuation_provider: str,
    continuation_model: str,
) -> None:
    """Run comparison benchmarks with explicit task-to-agents associations.

    The config file should be a YAML file with the following format:

    comparisons:
      - task: task_name
        agents:
          - agent_name_1
          - agent_name_2
          - ...
    - task: task_name
        agents:
          - agent_name_1
          - agent_name_2
          - ...
    """
    # Load comparison specs from config file
    with config_file.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if "comparisons" not in config:
        raise click.ClickException("Config file must contain a 'comparisons' key")

    comparison_specs = []
    for spec_data in config["comparisons"]:
        if "task" not in spec_data or "agents" not in spec_data:
            raise click.ClickException("Each comparison spec must have 'task' and 'agents' keys")
        comparison_specs.append(
            ComparisonTaskSpec(
                task_name=spec_data["task"],
                agent_names=spec_data["agents"],
            )
        )

    if not comparison_specs:
        raise click.ClickException("No comparison specs found in config file")

    # Set up runs directory
    if runs_dir is None:
        base_dir = Path(__file__).parents[1] / ".comparison_results"
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
        runs_dir = base_dir / timestamp
        logger.info(f"Starting new comparison run: {runs_dir}")
    else:
        logger.info(f"Using runs directory: {runs_dir}")

    runs_dir.mkdir(parents=True, exist_ok=True)
    log_file = runs_dir / "comparison_run.log"
    logger.add(log_file, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}")
    logger.info(f"Logging to {log_file}")
    logger.info(f"Loaded {len(comparison_specs)} comparison spec(s) from {config_file}")

    harness = ComparisonHarness(
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
        max_parallel=max_parallel,
        comparison_runs=comparison_runs,
        continuation_provider=cast(Literal["openai", "azure_openai", "none"], continuation_provider),
        continuation_model=cast(Literal["gpt-5", "gpt-5.1"], continuation_model),
    )

    results_path = asyncio.run(harness.run(comparison_specs=comparison_specs))
    logger.info(f"Comparison benchmark complete. Results written to: {results_path}")

    # Generate HTML report
    report_path = create_comparison_html_report(runs_dir, tasks_dir)
    logger.info(f"HTML report generated: {report_path}")


if __name__ == "__main__":
    main()
