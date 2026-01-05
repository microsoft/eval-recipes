# Copyright (c) Microsoft. All rights reserved.

"""
Script to set up OpenAI FrontierScience benchmark tasks by downloading from HuggingFace.

This script:
1. Fetches task data from the HuggingFace datasets API
2. Samples n tasks with a fixed seed for reproducibility
3. Creates eval-recipes benchmark tasks for each sampled task
"""

import json
from pathlib import Path
import random
import shutil
import urllib.request

import click

HUGGINGFACE_API_URL = "https://datasets-server.huggingface.co/rows"
DATASET_NAME = "openai/frontierscience"
SPLIT = "test"

INSTRUCTIONS_TEMPLATE = """\
This is a FrontierScience evaluation task. Do NOT look up the answer.

{problem}

Put your final answer in ANSWER.txt"""

TEST_PY_TEMPLATE = '''
"""
Test script for FrontierScience task: {task_id}

Uses semantic test to compare ANSWER.txt against the expected answer.
"""

import asyncio
import sys
from pathlib import Path

import click

from eval_recipes.benchmarking.semantic_test import semantic_test
from eval_recipes.benchmarking.test_utils import (
    get_instructions_from_file_or_default,
    get_test_id_from_env_or_default,
    write_test_result,
)

EXPECTED_ANSWER = """{expected_answer}"""

STEPS = f"""
1. Read the file ANSWER.txt in /project/ directory
2. Compare the agent's answer to the expected answer below
3. Consider mathematical equivalence - different forms of the same answer should be considered correct

Expected answer:
{{EXPECTED_ANSWER}}
"""

RUBRIC = {{
    "is_correct": "str - Is the answer in ANSWER.txt correct or mathematically equivalent to the expected answer? Explain briefly.",
    "score": "float - 100 if correct, 0 if incorrect or ANSWER.txt is missing.",
}}


@click.command()
@click.option(
    "--test-id",
    default=lambda: get_test_id_from_env_or_default("dev"),
    help="Test ID for result file naming",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=lambda: Path(__file__).parents[0],
    help="Directory to write result file",
)
@click.option(
    "--instructions-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to instructions file",
)
def main(test_id: str, output_dir: Path, instructions_file: Path | None) -> int:
    """Test FrontierScience task by comparing ANSWER.txt to expected answer."""
    return asyncio.run(async_main(test_id, output_dir, instructions_file))


async def async_main(test_id: str, output_dir: Path, instructions_file: Path | None) -> int:
    """Async implementation of the test."""
    metadata: dict = {{}}

    instructions = get_instructions_from_file_or_default(instructions_file=instructions_file)
    metadata["instructions"] = instructions

    result = await semantic_test(
        steps=STEPS,
        rubric=RUBRIC,
        context=instructions,
        working_dir=Path("/project"),
    )

    metadata["semantic_test_result"] = result.metadata
    write_test_result(output_dir, test_id, result.score, metadata)

    print(f"Score: {{result.score:.1f}}/100")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

TASK_YAML_TEMPLATE = """\
required_env_vars:
  - ANTHROPIC_API_KEY
task_info:
  difficulty: hard
  non_deterministic_evals: true
  categories:
    - openai-frontier-science
"""


def fetch_all_rows() -> list[dict]:
    """Fetch all rows from the FrontierScience dataset."""
    all_rows = []
    offset = 0
    page_size = 100

    print(f"Fetching data from HuggingFace dataset: {DATASET_NAME}...")

    while True:
        url = f"{HUGGINGFACE_API_URL}?dataset={DATASET_NAME}&config=default&split={SPLIT}&offset={offset}&length={page_size}"
        request = urllib.request.Request(url)

        with urllib.request.urlopen(request) as response:
            data = json.loads(response.read().decode())

        rows = data.get("rows", [])
        if not rows:
            break

        for row in rows:
            all_rows.append(row["row"])

        print(f"  Fetched {len(all_rows)} rows...")

        if len(rows) < page_size:
            break

        offset += page_size

    print(f"Found {len(all_rows)} total tasks")
    return all_rows


def generate_instructions(row: dict) -> str:
    """Generate instructions.txt content for a task."""
    # Unescape backslashes that come escaped from the dataset
    problem = row["problem"].replace("\\\\", "\\")
    return INSTRUCTIONS_TEMPLATE.format(problem=problem)


def generate_test_py(task_id: str, row: dict) -> str:
    """Generate test.py content for a task."""
    # Unescape backslashes from dataset, then escape for Python triple-quoted string
    expected_answer = row["answer"].replace("\\\\", "\\")
    # Escape triple quotes if they appear in the answer
    expected_answer = expected_answer.replace('"""', '\\"\\"\\"')
    return TEST_PY_TEMPLATE.format(
        task_id=task_id,
        expected_answer=expected_answer,
    )


def create_task(task_id: str, row: dict, output_dir: Path) -> None:
    """Create a benchmark task directory with all required files."""
    task_dir = output_dir / f"frontier-science-{task_id}"
    task_dir.mkdir(parents=True, exist_ok=True)

    instructions_file = task_dir / "instructions.txt"
    instructions_file.write_text(generate_instructions(row))

    test_file = task_dir / "test.py"
    test_file.write_text(generate_test_py(task_id, row))

    yaml_file = task_dir / "task.yaml"
    yaml_file.write_text(TASK_YAML_TEMPLATE)

    print(f"  Created task: {task_dir.name}")


def clean_existing_tasks(output_dir: Path) -> int:
    """Remove existing FrontierScience tasks from the output directory."""
    removed_count = 0
    if not output_dir.exists():
        return removed_count

    for task_dir in output_dir.iterdir():
        if task_dir.is_dir() and task_dir.name.startswith("frontier-science-"):
            shutil.rmtree(task_dir)
            removed_count += 1

    return removed_count


@click.command()
@click.option(
    "--num-tasks",
    type=int,
    default=10,
    help="Number of tasks to sample (default: 10)",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed for consistent sampling (default: 42)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path(__file__).parents[2] / "data" / "tasks",
    help="Directory to create tasks in (default: data/tasks)",
)
@click.option(
    "--clean",
    is_flag=True,
    default=False,
    help="Remove existing FrontierScience tasks before creating new ones",
)
def main(num_tasks: int, seed: int, output_dir: Path, clean: bool) -> None:
    """Set up FrontierScience benchmark tasks by downloading from HuggingFace."""
    if clean:
        removed = clean_existing_tasks(output_dir)
        if removed > 0:
            print(f"Removed {removed} existing FrontierScience task(s)")

    all_rows = fetch_all_rows()

    if num_tasks > len(all_rows):
        print(f"Warning: Requested {num_tasks} tasks but only {len(all_rows)} available. Using all.")
        num_tasks = len(all_rows)

    random.seed(seed)
    sampled_rows = random.sample(all_rows, num_tasks)

    sampled_ids = [row["task_group_id"] for row in sampled_rows]
    print(f"\nSampled {num_tasks} tasks with seed={seed}:")
    for task_id in sampled_ids:
        print(f"  - {task_id}")

    print(f"\nCreating tasks in {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    for row in sampled_rows:
        task_id = row["task_group_id"]
        create_task(task_id, row, output_dir)

    print(f"\nDone! Created {num_tasks} FrontierScience benchmark tasks.")
    print(f"Seed used: {seed}")
    print(f"Task IDs: {', '.join(sampled_ids)}")


if __name__ == "__main__":
    main()
