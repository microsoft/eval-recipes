# Benchmarking

This module provides a benchmarking harness for evaluating AI agents on real-world tasks within isolated Docker containers.

The goal of the module is to produce a final report that details how well each agent performed on a variety of custom tasks.

The core of the module is a harness that uses agent definitions (defaults in `data/agents/`) 
and task definitions (defaults in `data/tasks/`) to run agents on tasks, each their own isolated Docker containers.
After each agent has been run on a task, the task's `test.py` script is executed to validate the agent's solution and produce a score.
For each agent-task pair that scores below a threshold, a report is generated that analyzes what went wrong.
Finally, these individual reports are rolled up into a final report for each agent.


## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Install [Docker Desktop](https://docs.docker.com/desktop/) for work on systems running Windows or [Docker Engine](https://docs.docker.com/engine/install/ubuntu/) on setups like WSL 2.
  - After installing Docker Engine on WSL 2, ensure your user has docker permissions by running:
    - `sudo usermod -aG docker $USER`
    - `newgrp docker`
- The Claude Agent SDK which requires setting up [Claude Code](https://docs.claude.com/en/docs/claude-code/overview)
- [`ANTHROPIC_API_KEY`](https://platform.claude.com/docs/en/get-started) for the Claude Agent SDK.
- [`OPENAI_API_KEY`](hhttps://platform.openai.com/api-keys) if using agent continuation (see parameters below, or running tasks that requires it as a dependency).

## Installation

```bash
# If you have make installed:
make install

# If not, you can manually setup the uv environment:
uv lock --upgrade && uv sync --all-extras --group dev
```

## Running Benchmarks

The benchmarking harness is available via the CLI script `scripts/run_benchmarks.py`.

### Basic Usage

```bash
# Make sure your .env file is setup according to .env.sample
uv run scripts/run_benchmarks.py --agent-filter name=claude_code --task-filter name=cpsc_recall_monitor,arxiv_conclusion_extraction,email_drafting --max-parallel-trials 6  --num-trials 2 --continuation-provider openai

# Command for a typical full benchmark run
# sec_10q_extractor is excluded due to most all agents failing at it.
uv run scripts/run_benchmarks.py --agent-filter name=amplifier_v1,amplifier_v2_toolkit,claude_code,gh_cli,openai_codex --task-filter name!=sec_10q_extractor --max-parallel-trials 20  --num-trials 5 --continuation-provider openai
```


## Creating a New Agent

Agents are defined in the `data/agents/` directory.
Each agent is a subdirectory containing the files needed to install and run the agent.
Included agents are located in [data/agents/](../data/agents/).

```
data/agents/your_agent_name/
agent.yaml                     # Agent configuration
install.dockerfile             # Docker commands to install the agent
command_template.txt           # Liquid template for the command to run the agent
command_template_continue.txt  # (Optional) Template for agent continuation when follow-up is needed. This command must continue from the previous session/conversation.
data/                          # (Optional) Agent-specific data files
```

See [data/agents/gh_cli/](../data/agents/gh_cli/) for an example agent definition.


## Using Local Agent Versions

For development and testing, create agent variants that use local source code instead of remote repositories. This enables rapid iteration on agent improvements while testing against benchmarks.

### Creating a Local Agent Variant

1. Create a new agent directory (e.g., `data/agents/your_agent_local/`)
2. Add `agent.yaml` with `local_source_path` pointing to your local source:
   ```yaml
   local_source_path: /absolute/path/to/your/agent/source
   required_env_vars:
     - API_KEY
   ```
3. Create `install.dockerfile` that installs from `/tmp/agent_source/` (where source is automatically copied)
4. Copy or create `command_template.txt`


### How It Works

1. Harness validates `local_source_path` exists
2. Collects files, respecting `.gitignore` if present (otherwise excludes `.git`, `.venv`, `__pycache__`, etc.)
3. Adds files to Docker build context as `agent_source/`
4. Copies `agent_source` to `/tmp/agent_source/` in container
5. Your `install.dockerfile` installs from there. This dockerfile should install the agent so that it is globally available. The commands will run in `/project/`, not where the agent's files are.
6. Image is rebuilt each run, capturing your latest changes

### Usage

```bash
# Run with local version
uv run scripts/run_benchmarks.py --agent-filter name=your_agent_local

# Compare local vs production
uv run scripts/run_benchmarks.py --agent-filter name=your_agent,your_agent_local
```

**Notes**: `local_source_path` must be absolute. Build time includes copying all source files. Images rebuild automatically to capture code changes.


## Creating a New Task

Tasks are defined in the `data/tasks/` directory. 
Each task is a subdirectory containing the files needed to define the task and test the agent's solution.
Included tasks are available at [data/tasks/](../data/tasks/).


### Task Directory Structure

A template task is available at **[data/_template_task/](../data/_template_task/)** that you can copy as a starting point. The `/create-benchmark-test` command can automate this for you.

```
data/tasks/your_task_name/
task.yaml            # Task configuration (required)
instructions.txt     # Instructions given to the agent (required)
test.py              # Python script to test the agent's solution (required)
setup.dockerfile     # (Optional) Docker commands to set up the task environment
data/                # (Optional) Directory containing test data files
```

### File Descriptions

#### `task.yaml` (Required)

Required fields:
- `task_info`: Object containing:
  - `difficulty`: One of `easy`, `medium`, or `hard`
  - `non_deterministic_evals`: Boolean indicating if test evaluations are non-deterministic (e.g., semantic tests using LLMs)

Optional task_info fields:
- `categories`: List of strings categorizing the task (e.g., `["cli", "automation"]`)

Optional fields:
- `required_env_vars`: List of environment variables required for the task (e.g., API keys for evaluation)
- `test_command`: Command to run the test script
- `timeout`: Timeout in seconds for agent execution

#### `setup.dockerfile` (Optional)

Contains Docker `RUN` commands to install any dependencies needed for the task or tests. This should be kept minimal, only include resources the agent would not be able to configure themselves.

#### `data/` (Optional)

Optional directory containing test data files that will be copied into the container and made available to `test.py`. This is useful for:
- Providing sample inputs for the agent to work with
- Supplying reference data for semantic tests to validate against
- Making tests more deterministic and reproducible

Examples: See `data/tasks/style_blender/data/` and `data/tasks/email_drafting/data/`

#### `instructions.txt` (Required)
Plain text instructions that will be passed to the agent. This describes what the agent should build or solve.

#### `test.py` (Required)
A Python test script that validates the agent's solution and outputs a score.

All test scripts must follow a standardized contract for integration with the harness. See **[data/_template_task/test.py](../data/_template_task/test.py)** for the complete contract specification, detailed documentation, and a reference implementation you can copy when creating new tasks.


### Semantic Tests

Many tasks use semantic tests. They are tests where an LLM "auditor" follows specific steps and evaluates outputs against a rubric. This is useful for complex tasks where deterministic validation is difficult.
See **[data/tasks/style_blender/test.py](../data/tasks/style_blender/test.py)** and **[data/tasks/email_drafting/test.py](../data/tasks/email_drafting/test.py)** for examples of these.


### Helper Commands for Task Creation

Two slash commands are available to help create high-quality benchmark tasks:
- [`/create-benchmark-test`](../.claude/commands/create-benchmark-test.md) - Guides you through creating a complete new benchmark task
- [`/create-semantic-tests`](../.claude/commands/create-semantic-tests.md) - Helps design semantic tests for a task


## Multi-Trial Evaluation

The harness supports running multiple trials of the same agent-task pair to measure consistency and reliability. Use `--num-trials N` to run each task N times.
Results are aggregated with statistics including mean, median, standard deviation, minimum, and maximum scores.
Each trial is stored in a separate subdirectory (`trial_1`, `trial_2`, etc.) within the results directory, and an `aggregated_results.json` file contains the statistical summary.


## Command-Line Options

The `scripts/run_benchmarks.py` script accepts the following options:

- `--agents-dir`: Path to agents directory
- `--tasks-dir`: Path to tasks directory
- `--runs-dir`: Output directory for results
- `--agent-filter`: Filter agents by field (format: `field=value`, `field!=value`, or `field=val1,val2` for multiple)
- `--task-filter`: Filter tasks by field (same format as agent-filter)
- `--generate-reports`: Generate failure analysis reports
- `--max-parallel-trials`: Maximum number of trials to run in parallel
- `--num-trials`: Number of trials per agent-task pair
- `--continuation-provider`: LLM provider for agent continuation - `openai`, `azure_openai`, or `none` to disable
- `--continuation-model`: Model to use for agent continuation decisions - `gpt-5` or `gpt-5.1`
- `--report-score-threshold`: Score threshold for generating failure reports


## Results

Results include detailed metrics:
- **Scores**: Task-specific scores from test scripts
- **Timing**: Agent execution duration and test execution duration
- **Reports**: Three types of reports are generated:
  - **Trial failure reports**: Individual analysis for each trial scoring below the threshold
  - **Consolidated reports**: Per-agent summary of all failures
  - **HTML reports**: Interactive dashboards with tabbed interface showing overview, task catalog, and per-agent detailed results

## Notes

- You may want to prune your Docker images and containers periodically to save space. Containers/images can hang around when runs are unexpectedly interrupted.
