# Copyright (c) Microsoft. All rights reserved

import asyncio
from datetime import datetime
import os
from pathlib import Path
import sys

import click
from dotenv import load_dotenv
from loguru import logger

from eval_recipes.benchmarking.loaders import load_agents, load_benchmark, load_tasks
from eval_recipes.benchmarking.pipelines.comparison_pipeline import ComparisonPipeline
from eval_recipes.benchmarking.pipelines.score_pipeline import ScorePipeline

load_dotenv()


@click.command()
@click.option(
    "--benchmark",
    "benchmark_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    default=lambda: Path(__file__).parents[1] / "data" / "benchmarks" / "full_benchmark.yaml",
    help="Path to benchmark definition YAML file",
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
    "--output-dir",
    type=click.Path(path_type=Path),
    default=lambda: Path.cwd() / ".benchmark_results" / datetime.now().strftime("%Y%m%d_%H%M%S"),
    help="Directory to store benchmark results",
)
@click.option(
    "--max-parallel",
    type=int,
    default=12,
    help="Maximum number of parallel jobs",
)
def main(
    benchmark_path: Path,
    agents_dir: Path,
    tasks_dir: Path,
    output_dir: Path,
    max_parallel: int,
) -> None:
    # Ensure output directory exists before setting up file logging
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging to both stderr and file
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(output_dir / "benchmark.log", level="INFO", encoding="utf-8")

    agents = load_agents(agents_dir)
    tasks = load_tasks(tasks_dir)
    benchmark = load_benchmark(benchmark_path)

    logger.info(f"Agents: {list(agents.keys())}")
    logger.info(f"Tasks: {list(tasks.keys())}")

    environment = {
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
        "GITHUB_TOKEN": os.environ.get("GITHUB_TOKEN", ""),
        "AZURE_OPENAI_ENDPOINT": os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        "AZURE_OPENAI_VERSION": os.environ.get("AZURE_OPENAI_VERSION", ""),
    }

    if benchmark.score_benchmark and benchmark.score_benchmark.score_benchmarks:
        logger.info("Running score pipeline...")
        pipeline = ScorePipeline(
            benchmark=benchmark.score_benchmark,
            agents=agents,
            tasks=tasks,
            output_dir=output_dir,
            max_parallel=max_parallel,
            environment=environment,
        )
        results = asyncio.run(pipeline.run())
        logger.info(f"Completed {len(results)} score job(s)")

    if benchmark.comparison_benchmark and benchmark.comparison_benchmark.comparison_benchmarks:
        logger.info("Running comparison pipeline...")
        comparison_pipeline = ComparisonPipeline(
            benchmark=benchmark.comparison_benchmark,
            agents=agents,
            tasks=tasks,
            output_dir=output_dir,
            max_parallel=max_parallel,
            environment=environment,
        )
        comparison_results = asyncio.run(comparison_pipeline.run())
        logger.info(f"Completed {len(comparison_results)} comparison job(s)")


if __name__ == "__main__":
    main()
