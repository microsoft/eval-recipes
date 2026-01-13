# Copyright (c) Microsoft. All rights reserved.

"""
Script to set up ARC-AGI-2 benchmark tasks by downloading evaluation data from GitHub.

This script:
1. Fetches available task IDs from the ARC-AGI-2 repository
2. Samples n task IDs with a fixed seed for reproducibility
3. Creates eval-recipes benchmark tasks for each sampled task
"""

import json
from pathlib import Path
import random
import shutil
import urllib.request

import click

GITHUB_API_URL = "https://api.github.com/repos/arcprize/ARC-AGI-2/contents/data/evaluation"
RAW_CONTENT_URL = "https://raw.githubusercontent.com/arcprize/ARC-AGI-2/main/data/evaluation"

INSTRUCTIONS_TEMPLATE = """\
## Overview

This is an ARC-AGI-2 evaluation task you will be solving. Do NOT look up the answer.

A "grid" is a rectangular matrix (list of lists) of integers between 0 and 9 (inclusive).
The smallest possible grid size is 1x1 and the largest is 30x30.

Your goal is to construct the output grid(s) corresponding to the test input grid(s).
"Constructing the output grid" involves picking the height and width of the output grid,
then filling each cell in the grid with a symbol (integer between 0 and 9).
Only exact solutions (all cells match the expected answer) are considered correct.

## Training Examples

Analyze these input/output pairs to understand the transformation pattern:

{training_examples}

## Test Input(s)

Apply the pattern you identified to produce the output for each test input.

{test_inputs}

## Output Format

For each test input i, create a file ANSWER_i.json containing the output grid as a JSON array.
{answer_files_list}

Each file should contain ONLY the grid (a list of lists of integers 0-9), e.g.:
[[0, 1, 2], [3, 4, 5]]
"""

TEST_PY_TEMPLATE = '''\
"""
Test script for ARC-AGI-2 task: {task_id}

Checks ANSWER_i.json files against expected outputs.
Score is the percentage of correct answers (exact grid matches only).
"""

import json
import sys
from pathlib import Path

import click

from eval_recipes.benchmarking.test_utils import (
    get_test_id_from_env_or_default,
    write_test_result,
)

EXPECTED_OUTPUTS = {expected_outputs_json}


def check_answer(expected: list[list[int]], answer_file: Path) -> tuple[float, str]:
    """Check if an answer file matches the expected output exactly."""
    if not answer_file.exists():
        return 0.0, f"File {{answer_file.name}} not found"

    try:
        actual = json.loads(answer_file.read_text())
    except json.JSONDecodeError as e:
        return 0.0, f"Invalid JSON in {{answer_file.name}}: {{e}}"

    if not isinstance(actual, list):
        return 0.0, f"{{answer_file.name}} must contain a list of lists"

    if actual == expected:
        return 100.0, f"{{answer_file.name}} is correct"
    else:
        return 0.0, f"{{answer_file.name}} does not match expected output"


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
def main(test_id: str, output_dir: Path) -> int:
    """Test ARC-AGI-2 task by checking ANSWER_i.json files against expected outputs."""
    metadata: dict = {{}}
    results: list[dict] = []
    total_score = 0.0

    project_dir = Path("/project")

    for i, expected in enumerate(EXPECTED_OUTPUTS):
        answer_file = project_dir / f"ANSWER_{{i}}.json"
        score, message = check_answer(expected, answer_file)
        total_score += score
        results.append({{"test_index": i, "score": score, "message": message}})
        print(f"Test {{i}}: {{message}} ({{score:.0f}}/100)")

    num_tests = len(EXPECTED_OUTPUTS)
    final_score = total_score / num_tests if num_tests > 0 else 0.0

    metadata["results"] = results
    metadata["num_tests"] = num_tests
    metadata["tests_passed"] = sum(1 for r in results if r["score"] == 100.0)

    write_test_result(output_dir, test_id, final_score, metadata)

    print(f"\\nFinal score: {{final_score:.1f}}/100 ({{metadata['tests_passed']}}/{{num_tests}} tests passed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

TASK_YAML_TEMPLATE = """\
timeout: 3600
task_info:
  difficulty: hard
  non_deterministic_evals: false
  categories:
    - arc-agi-2
"""


def fetch_task_ids() -> list[str]:
    """Fetch available task IDs from the ARC-AGI-2 GitHub repository."""
    print(f"Fetching task list from {GITHUB_API_URL}...")

    request = urllib.request.Request(GITHUB_API_URL)

    with urllib.request.urlopen(request) as response:
        data = json.loads(response.read().decode())

    task_ids = []
    for item in data:
        if item["type"] == "file" and item["name"].endswith(".json"):
            task_id = item["name"].replace(".json", "")
            task_ids.append(task_id)

    print(f"Found {len(task_ids)} evaluation tasks")
    return sorted(task_ids)


def download_task(task_id: str) -> dict:
    """Download a task JSON from GitHub."""
    url = f"{RAW_CONTENT_URL}/{task_id}.json"
    print(f"  Downloading {task_id}...")

    request = urllib.request.Request(url)
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode())


def format_grid(grid: list[list[int]]) -> str:
    """Format a grid as JSON for instructions."""
    return json.dumps(grid)


def generate_instructions(task_data: dict) -> str:
    """Generate instructions.txt content for a task."""
    training_examples = []
    for i, example in enumerate(task_data["train"]):
        example_text = f"""### Example {i + 1}

Input:
{format_grid(example["input"])}

Output:
{format_grid(example["output"])}"""
        training_examples.append(example_text)

    test_inputs = []
    for i, test in enumerate(task_data["test"]):
        test_text = f"""### Test {i}

Input:
{format_grid(test["input"])}"""
        test_inputs.append(test_text)

    num_tests = len(task_data["test"])
    answer_files_list = "\n".join(f"- ANSWER_{i}.json for Test {i}" for i in range(num_tests))

    return INSTRUCTIONS_TEMPLATE.format(
        training_examples="\n\n".join(training_examples),
        test_inputs="\n\n".join(test_inputs),
        answer_files_list=answer_files_list,
    )


def generate_test_py(task_id: str, task_data: dict) -> str:
    """Generate test.py content for a task."""
    expected_outputs = [test["output"] for test in task_data["test"]]
    expected_outputs_json = json.dumps(expected_outputs)

    return TEST_PY_TEMPLATE.format(
        task_id=task_id,
        expected_outputs_json=expected_outputs_json,
    )


def create_task(task_id: str, task_data: dict, output_dir: Path) -> None:
    """Create a benchmark task directory with all required files."""
    task_dir = output_dir / f"arc-agi-2-{task_id}"
    task_dir.mkdir(parents=True, exist_ok=True)

    instructions_file = task_dir / "instructions.txt"
    instructions_file.write_text(generate_instructions(task_data))

    test_file = task_dir / "test.py"
    test_file.write_text(generate_test_py(task_id, task_data))

    yaml_file = task_dir / "task.yaml"
    yaml_file.write_text(TASK_YAML_TEMPLATE)

    print(f"  Created task: {task_dir.name}")


def clean_existing_tasks(output_dir: Path) -> int:
    """Remove existing ARC-AGI-2 tasks from the output directory."""
    removed_count = 0
    if not output_dir.exists():
        return removed_count

    for task_dir in output_dir.iterdir():
        if task_dir.is_dir() and task_dir.name.startswith("arc-agi-2-"):
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
    help="Remove existing ARC-AGI-2 tasks before creating new ones",
)
def main(num_tasks: int, seed: int, output_dir: Path, clean: bool) -> None:
    """Set up ARC-AGI-2 benchmark tasks by downloading from GitHub."""
    if clean:
        removed = clean_existing_tasks(output_dir)
        if removed > 0:
            print(f"Removed {removed} existing ARC-AGI-2 task(s)")

    task_ids = fetch_task_ids()

    if num_tasks > len(task_ids):
        print(f"Warning: Requested {num_tasks} tasks but only {len(task_ids)} available. Using all.")
        num_tasks = len(task_ids)

    random.seed(seed)
    sampled_ids = random.sample(task_ids, num_tasks)
    sampled_ids.sort()

    print(f"\nSampled {num_tasks} tasks with seed={seed}:")
    for task_id in sampled_ids:
        print(f"  - {task_id}")

    print(f"\nCreating tasks in {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    for task_id in sampled_ids:
        task_data = download_task(task_id)
        create_task(task_id, task_data, output_dir)

    print(f"\nDone! Created {num_tasks} ARC-AGI-2 benchmark tasks.")
    print(f"Seed used: {seed}")
    print(f"Task IDs: {', '.join(sampled_ids)}")


if __name__ == "__main__":
    main()
