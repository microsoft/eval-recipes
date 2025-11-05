# Benchmarking

This module provides a benchmarking harness for evaluating AI agents on real-world tasks within isolated Docker containers.

The goal of the module is to produce a final report that details how well each agent performed on a variety of custom tasks.

The core of the module is a harness that uses agent definitions (defaults in `data/agents/`) 
and task definitions (defaults in `data/tasks/`) to run agents on tasks, each their own isolated Docker containers.
After each agent has been run on a task, the task's `test.py` script is executed to validate the agent's solution and produce a score.
For each agent-task pair that scores below 85%, a report is generated that analyzes what went wrong.
Finally, these individual reports are rolled up into a final report for each agent.


## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Docker](https://docs.docker.com/engine/install/ubuntu/)
  - After installing, ensure your user has docker permissions by running:
    - `sudo usermod -aG docker $USER`
    - `newgrp docker`
- Claude Agent SDK which requires [Claude Code](https://docs.claude.com/en/docs/claude-code/overview)
- ANTHROPIC_API_KEY for the Claude Agent SDK.


## Running Benchmarks

The benchmarking harness is available via the CLI script `scripts/run_benchmarks.py`.

### Basic Usage

```bash
# Make sure your .env file is setup according to .env.sample
# Run all agents on all tasks, by default this will use the existing data/agents/ and data/tasks/ directories
uv run scripts/run_benchmarks.py --max-parallel-tasks 3

# You can also specify various filters
uv run scripts/run_benchmarks.py --agent-filter name=claude_code --task-filter name=your_task_name
```


## Creating a New Agent

Agents are defined in the `data/agents/` directory.
Each agent is a subdirectory containing the files needed to install and run the agent.
Included agents are located in [data/agents/](../data/agents/).

```
data/agents/your_agent_name/
agent.yaml            # Agent configuration
install.dockerfile    # Docker commands to install the agent
command_template.txt  # Liquid template for the command to run the agent
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

Optional fields:
- `required_env_vars`: List of environment variables required for the task (e.g., API keys for evaluation)
- `test_command`: Command to run the test script (default: `uv run --no-project /project/test.py`)

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

## Notes

- You may want to prune your Docker images and containers periodically to save space. Containers/images can hang around when runs are unexpectedly interrupted.
