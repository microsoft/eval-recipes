# Benchmarking

This module provides a benchmarking harness for evaluating AI agents within isolated Docker containers.


## Installation

```bash
# Install prerequisites below first.
# With uv (add to project dependencies, pinned to a release tag)
uv add "eval-recipes @ git+https://github.com/microsoft/eval-recipes@v0.0.32"

# With pip
pip install "git+https://github.com/microsoft/eval-recipes@v0.0.32"
```


## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Install [Docker Desktop](https://docs.docker.com/desktop/) for work on systems running Windows* or [Docker Engine](https://docs.docker.com/engine/install/ubuntu/) on setups like WSL 2.
  - After installing Docker Engine on WSL 2, ensure your user has docker permissions by running:
    - `sudo usermod -aG docker $USER`
    - `newgrp docker`
- The Claude Agent SDK which requires setting up [Claude Code](https://docs.claude.com/en/docs/claude-code/overview)
- [`ANTHROPIC_API_KEY`](https://platform.claude.com/docs/en/get-started) for the Claude Agent SDK.
- [`OPENAI_API_KEY`](https://platform.openai.com/api-keys) if using agent continuation (see parameters below, or running tasks that requires it as a dependency).

* All features may not currently work on Windows due to Claude Agent SDK limitations.


## Usage

1. Create agent definition(s). Examples are provided in [data/agents](../data/agents) and described further [below](#agent-definitions).
1. Create task definition(s). Examples are provided in [data/tasks](../data/tasks) and described further [below](#task-definitions).
1. Create a benchmark configuration. Examples are provided in [data/benchmarks](../data/benchmarks) and described further [below](#benchmark-configurations).

```python
import asyncio
from pathlib import Path

from eval_recipes.benchmarking.loaders import load_agents, load_benchmark, load_tasks
from eval_recipes.benchmarking.pipelines.score_pipeline import ScorePipeline

agents = load_agents(Path("data/agents"))
tasks = load_tasks(Path("data/tasks"))
benchmark = load_benchmark(Path("data/benchmarks/full_benchmark.yaml"))

pipeline = ScorePipeline(
    benchmark=benchmark.score_benchmark,
    agents=agents,
    tasks=tasks,
    output_dir=Path(".benchmark_results"),
)
asyncio.run(pipeline.run())
```

Or for comparison based evaluations:

```python
import asyncio
from pathlib import Path

from eval_recipes.benchmarking.loaders import load_agents, load_benchmark, load_tasks
from eval_recipes.benchmarking.pipelines.comparison_pipeline import ComparisonPipeline

agents = load_agents(Path("data/agents"))
tasks = load_tasks(Path("data/tasks"))
benchmark = load_benchmark(Path("data/benchmarks/full_benchmark.yaml"))

pipeline = ComparisonPipeline(
    benchmark=benchmark.comparison_benchmark,
    agents=agents,
    tasks=tasks,
    output_dir=Path(".comparison_results"),
)
asyncio.run(pipeline.run())
```


## Agent Definitions

Each agent is a subdirectory containing an `agent.yaml` file and optional supporting files:

```
agent_id/                         # Agent directory
  agent.yaml                      # Agent configuration (required)
  data/                           # (Optional) Agent-specific data files used during installation or runtime
```

### `agent.yaml`

Configuration file for the agent. Contains all agent settings in a single file.

```yaml
# Required fields
id: my_agent                                   # Unique identifier for the agent
agent_name: my_agent                           # Display name (allows grouping agent variants)
command_template: >-                           # Liquid template for the command to start a task
  my-agent -p "{{task_instructions}}"

# Optional fields
command_template_continue: >-                  # Template for agent continuation when follow-up is needed
  my-agent -p "{{task_instructions}}" --continue

dockerfile_portion: |                          # Docker commands to install the agent
  RUN npm install -g my-agent
  RUN my-agent --version

installation_files:                            # Files to copy into container before agent installation
  - source: ./config                           # Relative to agent directory, or absolute path
    dest: /root/.config/my-agent               # Absolute path in container

runtime_files:                                 # Files to copy into container after agent installation
  - source: ./data
    dest: /project/data

agent_logs_paths:                              # Paths where the agent stores logs in the container
  - /root/.my-agent/logs

source_code_path: /opt/my-agent                # Location of agent's source code in container (if applicable)
```

### `command_template`

[Python Liquid](https://github.com/jg-rp/liquid) template for the command to run the agent. The `{{task_instructions}}` variable contains the task instructions.

### `command_template_continue`

Optional. Liquid template for continuing an agent session when follow-up is needed. The `{{task_instructions}}` variable contains the continuation prompt. This command must resume the previous conversation.

### `data/`

Optional. Directory containing agent-specific data files. Contents can be copied to the container using `installation_files` or `runtime_files` mappings.

### Defining Local Agents

For development and testing, you can create agent variants that use local source code instead of remote repositories. 

1. Add `source_code_path` pointing to where source will be in the container
2. Use `installation_files` to copy local source to the container:
   ```yaml
   installation_files:
     - source: /absolute/path/to/your/agent/source
       dest: /tmp/agent_source
   ```
3. Create `dockerfile_portion` that installs from the copied location
4. By default it ignores files in `.gitignore` if present (otherwise excludes `.git`, `.venv`, `__pycache__`, etc.)


## Task Definitions

Each task is a subdirectory containing the files needed to define the task and test the agent's solution.

```
task_id/
  task.yaml            # Task configuration (required)
  test.py              # Python script to test the agent's solution (required for score-based tasks)
  task_time_data/      # (Optional) Data files copied before agent runs
  test_time_data/      # (Optional) Data files copied before tests run
```

### `task.yaml`

Configuration file for the task.

```yaml
# Required
name: my_task                                          # Task name, usually matching directory name
task_info:
  difficulty: medium                                   # "easy", "medium", or "hard"
  non_deterministic_evals: true                        # Whether evals use LLMs (default: false)
  categories:                                          # Optional category tags
    - cli_tool
    - writing

# Task instructions (can also be in separate instructions.txt)
instructions: |
  Build a CLI tool that...

# Evaluation configuration
evaluation_configs:
  - type: score                                        # For score-based evaluation
    test_script: test.py                               # Path to test script (relative to task dir)
    test_command: uv run --no-project /project/test.py # Command to run tests

  - type: comparison                                   # For comparison-based evaluation
    guidelines: |                                      # Optional evaluation guidelines for the judge
      Focus on clarity and completeness of the response.

# Optional
timeout: 5400                                          # Timeout in seconds (default: 1800)
dockerfile_portion: |                                  # Docker commands to set up task environment
  RUN apt-get install -y some-dependency

task_time_files:                                       # Files copied before agent runs
  - source: ./task_time_data
    dest: /project

test_time_files:                                       # Files copied before tests run
  - source: ./test_time_data
    dest: /project
```

### `test.py`

Python script that evaluates the agent's work. The script runs inside the container after the agent completes. 
Uses `semantic_test()` for LLM-based evaluation (eval-recipes is by default available when the script runs) or direct file checks for deterministic tests.

The script must output a JSON to following path `{output_dir}/.eval_recipes_test_results_{test_id}.json` and the following format:

```json
{"score": 85.0, "metadata": {"field": "value"}}
```

The `write_test_result()` utility handles writing this file in the correct format. 
See the following complete example:

```python
import asyncio
import sys
from pathlib import Path

import click
from eval_recipes.benchmarking.evaluation.semantic_test import semantic_test
from eval_recipes.benchmarking.evaluation.test_utils import (
    get_instructions_from_file_or_default,
    get_test_id_from_env_or_default,
    write_test_result,
)

STEPS = """1. Explore the /project directory to understand what the agent created
2. Verify the required files exist
3. Check that the implementation meets the requirements"""

RUBRIC = {
    "files_exist": "bool - Do all required files exist?",
    "implementation_correct": "str - Assessment of the implementation",
    "score": "float - Overall score from 0-100",
}

@click.command()
@click.option("--test-id", default=lambda: get_test_id_from_env_or_default("dev"))
@click.option("--output-dir", type=click.Path(path_type=Path), default=lambda: Path(__file__).parents[0])
@click.option("--instructions-file", type=click.Path(path_type=Path), default=None)
def main(test_id: str, output_dir: Path, instructions_file: Path | None) -> int:
    return asyncio.run(run_test(test_id, output_dir, instructions_file))

async def run_test(test_id: str, output_dir: Path, instructions_file: Path | None) -> int:
    instructions = get_instructions_from_file_or_default(instructions_file)

    result = await semantic_test(
        steps=STEPS,
        rubric=RUBRIC,
        context=instructions,
        working_dir=Path("/project"),
    )

    write_test_result(output_dir, test_id, result.score, result.metadata)
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

### `task_time_data/`

Optional. Directory containing files the agent needs to complete the task (e.g., PDFs, source code archives). Contents are copied to `/project` before the agent runs.

### `test_time_data/`

Optional. Directory containing files needed only for testing (e.g., expected outputs, test inputs). Contents are copied to `/project` before tests run, but after the agent completes.


## Benchmark Configurations

YAML files that define which agents to run on which tasks. Agent and task names must match the `id` and `name` fields in agent and task definitions respectively.

### Combined Configuration

A single benchmark file can define both score-based and comparison-based evaluations:

```yaml
score_benchmark:
  benchmark_type: score
  continuation_provider: openai              # "openai", "azure_openai", or "none"
  continuation_model: gpt-5.1
  analysis_score_threshold: 85.0             # Skip failure analysis if score >= threshold
  score_benchmarks:
    - agent_id: claude_code
      task_names:
        - email_drafting
        - arxiv_paper_summarizer
      trials: 3                              # Number of trials per task
    - agent_id: gh_cli
      task_names:
        - email_drafting
      trials: 1

comparison_benchmark:
  benchmark_type: comparison
  continuation_provider: openai
  continuation_model: gpt-5.1
  comparison_runs: 7                         # Number of comparison runs per task
  comparison_benchmarks:
    - task_name: ppt-1
      agent_ids:
        - claude_code
        - gh_cli
        - openai_codex
    - task_name: ppt-2
      agent_ids:
        - claude_code
        - gh_cli
```


## Pipelines

Pipelines orchestrate the benchmark run: they build Docker images for each agent-task combination, run agents in isolated containers, execute test scripts to evaluate results, and generate HTML reports with scores and analysis.

### Score Pipeline

[Source](../eval_recipes/benchmarking/pipelines/score_pipeline.py)

```python
from eval_recipes.benchmarking.pipelines.score_pipeline import ScorePipeline

pipeline = ScorePipeline(
    benchmark=ScoreBenchmarkDefinition,    # Parsed benchmark configuration
    agents=dict[str, AgentDefinition],     # Loaded agent definitions
    tasks=dict[str, TaskDefinition],       # Loaded task definitions
    output_dir=Path,                       # Output directory (default: .benchmark_results)
    max_parallel=5,                        # Maximum concurrent trials
    environment=dict | None,               # Environment variables for containers
)

await pipeline.run()
```

### Comparison Pipeline

[Source](../eval_recipes/benchmarking/pipelines/comparison_pipeline.py)

```python
from eval_recipes.benchmarking.pipelines.comparison_pipeline import ComparisonPipeline

pipeline = ComparisonPipeline(
    benchmark=ComparisonBenchmarkDefinition,  # Parsed benchmark configuration
    agents=dict[str, AgentDefinition],        # Loaded agent definitions
    tasks=dict[str, TaskDefinition],          # Loaded task definitions
    output_dir=Path,                          # Output directory (default: .comparison_results)
    max_parallel=5,                           # Maximum concurrent jobs
    environment=dict | None,                  # Environment variables for containers
)

await pipeline.run()
```


## Semantic Test

LLM-based evaluation functions that use Claude to audit agent work. Used within `test.py` scripts.

### `semantic_test`

[Source](../eval_recipes/benchmarking/evaluation/semantic_test.py)

Audits a single agent's work against a rubric.

```python
from eval_recipes.benchmarking.evaluation.semantic_test import semantic_test

result = await semantic_test(
    steps="1. Explore /project\n2. Check if README exists\n3. Run the tests",
    rubric={
        "readme_exists": "bool - Does README.md exist?",
        "tests_pass": "bool - Do the tests pass?",
        "score": "float - Overall score 0-100",  # Required field
    },
    context="The agent was asked to create a CLI tool...",
    working_dir=Path("/project"),
)

# result.score: float (0-100)
# result.metadata: dict (other rubric fields)
```

### `semantic_test_comparison`

[Source](../eval_recipes/benchmarking/evaluation/semantic_test_comparison.py)

Compares multiple agents' work on the same task using blind evaluation. Directories are anonymized before comparison.

```python
from eval_recipes.benchmarking.evaluation.semantic_test_comparison import semantic_test_comparison

result = await semantic_test_comparison(
    original_task="Create a CLI tool that...",
    directories=[
        Path("/outputs/agent_1/project"),
        Path("/outputs/agent_2/project"),
        Path("/outputs/agent_3/project"),
    ],
    guidelines="Focus on code quality and completeness",  # Optional
    log_file=Path("comparison.log"),  # Optional
)

# result.reasoning: str (bullet-point analysis)
# result.rankings: list[int] (indices ordered best to worst)
# result.anonymous_to_index: dict (maps anonymized names to indices)
```


## Third-Party Benchmarks

Scripts for integrating third-party benchmarks are located in `scripts/third_party_benchmarks/`. These scripts download benchmark data from external sources and convert them into eval-recipes score-based task format.

### ARC-AGI-2

[ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2/tree/main) is a general artificial intelligence benchmark focused on abstract reasoning. Tasks involve analyzing input/output grid pairs to discover transformation patterns, then applying those patterns to new test inputs.
A set of sample tasks can be created using this script:

```bash
uv run scripts/third_party_benchmarks/setup_arc_agi_2.py --num-tasks 10 --seed 42 --output-dir data/tasks --clean
```

### OpenAI FrontierScience

[FrontierScience](https://huggingface.co/datasets/openai/frontierscience) is a benchmark that evaluates AI capabilities for expert-level scientific reasoning across physics, chemistry, and biology.

```bash
uv run scripts/third_party_benchmarks/setup_frontier_science.py --num-tasks 10 --seed 42 --output-dir data/tasks --clean
```

## Notes

- You may want to prune your Docker images and containers periodically to save space. Containers/images can hang around when runs are unexpectedly interrupted.
