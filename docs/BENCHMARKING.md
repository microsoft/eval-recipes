# Benchmarking

This module provides a benchmarking harness for evaluating AI agents on real-world tasks within isolated Docker containers.


## Running Benchmarks

The benchmarking harness is available via the CLI script `scripts/run_benchmarks.py`.

### Basic Usage

```bash
# The default agents/tasks require these environment variables
export ANTHROPIC_API_KEY=your_anthropic_key
export OPENAI_API_KEY=your_openai_key

# Run all agents on all tasks, by default this will use the existing data/agents/ and data/tasks/ directories
uv run scripts/run_benchmarks.py --max-parallel-tasks 3

# You can also specify various filters
uv run scripts/run_benchmarks.py --agent-filter name=claude_code
uv run scripts/run_benchmarks.py --task-filter name=email_drafting
uv run scripts/run_benchmarks.py \
  --agent-filter name=claude_code \
  --task-filter task_info.difficulty=easy
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


## Creating a New Task

Tasks are defined in the `data/tasks/` directory. 
Each task is a subdirectory containing the files needed to define the task and test the agent's solution.
Included tasks are available at [data/tasks/](../data/tasks/).


### Task Directory Structure

```
data/tasks/your_task_name/
task.yaml            # Task configuration (required)
instructions.txt     # Instructions given to the agent (required)
test.py              # Python script to test the agent's solution (required)
setup.dockerfile     # (Optional) Docker commands to set up the task environment
test_commands.sh     # (Optional) Bash script to run before test.py
```

### File Descriptions

#### `task.yaml` (Required)
YAML configuration file for the task.

Required fields:
- `task_info`: Object containing:
  - `difficulty`: One of `easy`, `medium`, or `hard`
  - `non_deterministic_evals`: Boolean indicating if test evaluations are non-deterministic

Optional fields:
- `required_env_vars`: List of environment variables required for the task (e.g., API keys for evaluation)

#### `setup.dockerfile` (Optional)
Contains Docker `RUN` commands to install any dependencies needed for the task or tests.

#### `test_commands.sh` (Optional)
A bash script that runs before `test.py`. This is useful for installing test dependencies using `uv add` or setting up test data or configuration.

The script is executed from `/project` in the container. If it exists, the harness will:
1. Copy it into the container
2. Execute it with `bash test_commands.sh`
3. Save output to `test_commands_output.log`
4. Then proceed to run `test.py`

#### `instructions.txt` (Required)
Plain text instructions that will be passed to the agent. This describes what the agent should build or solve.

#### `test.py` (Required)
A Python test script that validates the agent's solution and outputs a score.

All test scripts must follow a standardized contract for integration with the harness. See **[data/tasks/test_template.py](../data/tasks/test_template.py)** for the complete contract specification, detailed documentation, and a reference implementation you can copy when creating new tasks.
