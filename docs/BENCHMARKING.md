# Benchmarking

This module provides a benchmarking harness for evaluating AI agents within isolated Docker containers.


## Installation

```bash
# Install prerequisites below first.
# With uv (add to project dependencies, pinned to a release tag)
uv add "eval-recipes @ git+https://github.com/microsoft/eval-recipes@v0.0.31"

# With pip
pip install "git+https://github.com/microsoft/eval-recipes@v0.0.31"
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
1. Create a run configuration. Examples are provided in [data/eval-setups](../data/eval-setups) and described further [below](#run-configurations).

```python
import yaml
from eval_recipes.benchmarking.harness import Harness
from eval_recipes.benchmarking.schemas import ScoreRunSpec

with Path("score-default.yaml").open(encoding="utf-8") as f:
    run_definition = ScoreRunSpec(**yaml.safe_load(f))

harness = Harness(
    agents_dir=Path("data/agents"),
    tasks_dir=Path("data/tasks"),
    run_definition=run_definition,
)
asyncio.run(harness.run())
```

Comparison based evaluations currently are separate from score based evaluations.

```python
import yaml
from eval_recipes.benchmarking.harness_comparison import ComparisonHarness
from eval_recipes.benchmarking.schemas import ComparisonTaskSpec

with Path("comparison-default.yaml").open(encoding="utf-8") as f:
    data = yaml.safe_load(f)
    specs = [ComparisonTaskSpec(task_name=c["task"], agent_names=c["agents"]) for c in data["comparisons"]]

harness = ComparisonHarness(
    agents_dir=Path("data/agents"),
    tasks_dir=Path("data/tasks"),
)

asyncio.run(harness.run(specs))
```


## Agent Definitions

Each agent is a subdirectory containing the files needed to install and run the agent:

```
agent_id/                         # Agent directory
  agent.yaml                      # Agent configuration
  install.dockerfile              # Docker commands to install the agent
  command_template.txt            # Liquid template for the command to start a task with the agent
  command_template_continue.txt   # (Optional) Template for agent continuation when follow-up is needed. This command must continue from the previous session/conversation.
  data/                           # (Optional) Agent-specific data files used during installation or runtime
```

### `agent.yaml`

Configuration file for the agent. All fields are optional.

```yaml
# Environment variables passed into the container.
# Sourced from the harness `environment` parameter.
required_env_vars:
  - ANTHROPIC_API_KEY
  - OPENAI_API_KEY

# Absolute path to local source code for development.
local_source_path: /path/to/source
```

### `install.dockerfile`

Docker commands to install the agent. These are injected into the [base image](../eval_recipes/benchmarking/base.dockerfile).

```dockerfile
# Example: Install GitHub Copilot CLI
RUN npm install -g @github/copilot
RUN copilot --version
```

### `command_template.txt`

[Python Liquid](https://github.com/jg-rp/liquid) template for the command to run the agent. The `{{task_instructions}}` variable contains the task instructions from `instructions.txt`.

```
copilot -p "{{task_instructions}}" --allow-all-tools
```

### `command_template_continue.txt`

Optional. Liquid template for continuing an agent session when follow-up is needed. The `{{task_instructions}}` variable contains the continuation prompt. This command must resume the previous conversation.

```
copilot -p "{{task_instructions}}" --continue --allow-all-tools
```

### `data/`

Optional. Directory containing agent-specific data files. Contents are copied to `/project` in the container before the agent runs.

### Defining Local Agents

For development and testing, you can create agent variants that use local source code instead of remote repositories. 

1. Add `agent.yaml` with `local_source_path` pointing to your local source:
   ```yaml
   local_source_path: /absolute/path/to/your/agent/source
   ```
1. Create `install.dockerfile` that installs from `/tmp/agent_source/` (where source is automatically copied)
1. By default it ignores files in `.gitignore` if present (otherwise excludes `.git`, `.venv`, `__pycache__`, etc.)


## Task Definitions

Each task is a subdirectory containing the files needed to define the task and test the agent's solution.

```
task_id/
  task.yaml            # Task configuration (required)
  instructions.txt     # Instructions given to the agent (required)
  test.py              # Python script to test the agent's solution (required for score-based tasks)
  setup.dockerfile     # (Optional) Docker commands to set up the task environment
  task_time_data/      # (Optional) Data files copied before agent runs
  test_time_data/      # (Optional) Data files copied before tests run
```

### `task.yaml`

Configuration file for the task.

```yaml
# Required
task_info:
  difficulty: medium                                   # "easy", "medium", or "hard"
  non_deterministic_evals: true                        # Whether evals use LLMs or not (default: false)
  categories:                                          # Optional category tags
    - cli_tool
    - writing

# Optional
timeout: 5400                                          # Timeout in seconds (default: 1800)
required_env_vars:                                     # Environment variables needed by the task
  - ANTHROPIC_API_KEY
test_command: uv run --no-project /project/test.py     # Command to run tests (default shown)

# For comparison-based tasks
eval_type: comparison                                  # "score", "comparison", or "both" (default: "score")
comparison_eval:
  guidelines: |                                        # Evaluation guidelines for the comparison judge
    Focus on clarity and completeness of the response.
```

### `instructions.txt`

The task prompt given to the agent. This is passed to the agent via the `{{task_instructions}}` template variable in `command_template.txt`.

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
from eval_recipes.benchmarking.semantic_test import semantic_test
from eval_recipes.benchmarking.test_utils import (
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

### `setup.dockerfile`

Optional. Docker commands to set up task-specific dependencies. Injected into the [base image](../eval_recipes/benchmarking/base.dockerfile) after the agent installation.

```dockerfile
# Example: Install Ollama for embedding tasks
RUN curl -fsSL https://ollama.com/install.sh | sh
RUN nohup ollama serve > /dev/null 2>&1 & sleep 5 && ollama pull embeddinggemma:300m-qat-q8_0
```

### `task_time_data/`

Optional. Directory containing files the agent needs to complete the task (e.g., PDFs, source code archives). Contents are copied to `/project` before the agent runs.

### `test_time_data/`

Optional. Directory containing files needed only for testing (e.g., expected outputs, test inputs). Contents are copied to `/project` before tests run, but after the agent completes.


## Run Configurations

YAML files that define which agents to run on which tasks. Agent and task names must match directory names in `agents_dir` and `tasks_dir` passed to the harness.

### Score-Based Configuration

```yaml
type: score
definitions:
  - agent: claude_code
    trials: 3                        # Default trials for this agent's tasks
    tasks:
      - task: email_drafting
      - task: arxiv_paper_summarizer
        trials: 5                    # Override trials for this specific task
  - agent: gh_cli
    trials: 3
    tasks:
      - task: email_drafting
```

### Comparison-Based Configuration

Each comparison can include any number of agents (minimum 2). The harness runs each agent on the task and uses an agent judge to rank their outputs.

```yaml
comparisons:
  - task: ppt-1
    agents:
      - claude_code
      - gh_cli
      - openai_codex
  - task: ppt-2
    agents:
      - claude_code
      - gh_cli
```


## Harnesses

Harnesses orchestrate the benchmark run: they build Docker images for each agent-task combination, run agents in isolated containers, execute test scripts to evaluate results, and generate HTML reports with scores and analysis.

### Score Harness

[Source](../eval_recipes/benchmarking/harness.py)

```python
from eval_recipes.benchmarking.harness import Harness

harness = Harness(
    agents_dir=Path,                  # Directory containing agent definitions
    tasks_dir=Path,                   # Directory containing task definitions
    run_definition=ScoreRunSpec,      # Parsed run configuration
    runs_dir=Path | None,             # Output directory (default: .benchmark_results)
    environment=dict | None,          # Environment variables for containers
    max_parallel_trials=5,            # Maximum concurrent trials
    continuation_provider="none",     # "openai", "azure_openai", or "none"
    continuation_model="gpt-5",       # Model for continuation prompts
    eval_recipes_version="...",       # Version of eval-recipes in containers
    report_score_threshold=85.0,      # Generate failure reports below this score
)

await harness.run()
```

### Comparison Harness

[Source](../eval_recipes/benchmarking/harness_comparison.py)

```python
from eval_recipes.benchmarking.harness_comparison import ComparisonHarness

harness = ComparisonHarness(
    agents_dir=Path,                  # Directory containing agent definitions
    tasks_dir=Path,                   # Directory containing task definitions
    runs_dir=Path | None,             # Output directory (default: .comparison_results)
    environment=dict | None,          # Environment variables for containers
    max_parallel=5,                   # Maximum concurrent jobs
    comparison_runs=3,                # Number of comparison runs per task
    continuation_provider="none",     # "openai", "azure_openai", or "none"
    continuation_model="gpt-5",       # Model for continuation prompts
    eval_recipes_version="...",       # Version of eval-recipes in containers
    report_score_threshold=85.0,      # Score threshold for reports
)

await harness.run(comparison_specs)   # Pass list of ComparisonTaskSpec
```


## Semantic Test

LLM-based evaluation functions that use Claude to audit agent work. Used within `test.py` scripts.

### `semantic_test`

[Source](../eval_recipes/benchmarking/semantic_test.py)

Audits a single agent's work against a rubric.

```python
from eval_recipes.benchmarking.semantic_test import semantic_test

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

[Source](../eval_recipes/benchmarking/semantic_test_comparison.py)

Compares multiple agents' work on the same task using blind evaluation. Directories are anonymized before comparison.

```python
from eval_recipes.benchmarking.semantic_test_comparison import semantic_test_comparison

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
