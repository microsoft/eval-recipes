# Copyright (c) Microsoft. All rights reserved.

from pathlib import Path

from loguru import logger
from pydantic import ValidationError
import yaml

from eval_recipes.benchmarking.schemas import (
    AgentDefinition,
    BenchmarkDefinition,
    InstallationFileMapping,
    ScoreEvalConfig,
    TaskDefinition,
)


def load_agents(agents_dir: Path) -> dict[str, AgentDefinition]:
    """Load agent definitions from a directory.

    Recursively searches for agent.yaml files in subdirectories.
    Returns a dict keyed by agent id.

    Args:
        agents_dir: Path to the agents directory.

    Returns:
        Dictionary mapping agent id to AgentDefinition.
    """
    agents: dict[str, AgentDefinition] = {}

    if not agents_dir.exists():
        logger.warning(f"Agents directory {agents_dir} does not exist.")
        return agents

    for agent_yaml_path in agents_dir.rglob("agent.yaml"):
        agent_dir = agent_yaml_path.parent

        try:
            with agent_yaml_path.open(encoding="utf-8") as f:
                agent_data = yaml.safe_load(f) or {}

            agent = AgentDefinition(**agent_data)

            # Resolve relative paths in installation_files and runtime_files
            agent = agent.model_copy(
                update={
                    "installation_files": _resolve_installation_file_mappings(agent.installation_files, agent_dir),
                    "runtime_files": _resolve_installation_file_mappings(agent.runtime_files, agent_dir),
                }
            )

            if agent.id in agents:
                logger.warning(f"Duplicate agent id '{agent.id}' found at {agent_yaml_path}, skipping.")
                continue

            agents[agent.id] = agent
            logger.debug(f"Loaded agent '{agent.id}' from {agent_yaml_path}")

        except ValidationError as e:
            logger.warning(f"Invalid agent definition at {agent_yaml_path}: {e}")
        except yaml.YAMLError as e:
            logger.warning(f"Failed to parse YAML at {agent_yaml_path}: {e}")

    logger.info(f"Loaded {len(agents)} agent(s) from {agents_dir}")
    return agents


def load_tasks(tasks_dir: Path) -> dict[str, TaskDefinition]:
    """Load task definitions from a directory.

    Recursively searches for task.yaml files in subdirectories.
    Returns a dict keyed by task name.

    Args:
        tasks_dir: Path to the tasks directory.

    Returns:
        Dictionary mapping task name to TaskDefinition.
    """
    tasks: dict[str, TaskDefinition] = {}

    if not tasks_dir.exists():
        logger.warning(f"Tasks directory {tasks_dir} does not exist.")
        return tasks

    for task_yaml_path in tasks_dir.rglob("task.yaml"):
        task_dir = task_yaml_path.parent

        try:
            with task_yaml_path.open(encoding="utf-8") as f:
                task_data = yaml.safe_load(f) or {}

            task = TaskDefinition(**task_data)

            # Resolve relative paths in task_time_files and test_time_files
            resolved_task_time_data = _resolve_installation_file_mappings(task.task_time_files, task_dir)
            resolved_test_time_data = _resolve_installation_file_mappings(task.test_time_files, task_dir)

            # Resolve test_script paths in ScoreEvalConfig
            resolved_eval_configs = []
            for eval_config in task.evaluation_configs:
                if isinstance(eval_config, ScoreEvalConfig) and not eval_config.test_script.is_absolute():
                    eval_config = eval_config.model_copy(update={"test_script": task_dir / eval_config.test_script})
                resolved_eval_configs.append(eval_config)

            task = task.model_copy(
                update={
                    "task_time_files": resolved_task_time_data,
                    "test_time_files": resolved_test_time_data,
                    "evaluation_configs": resolved_eval_configs,
                }
            )

            if task.name in tasks:
                logger.warning(f"Duplicate task name '{task.name}' found at {task_yaml_path}, skipping.")
                continue

            tasks[task.name] = task
            logger.debug(f"Loaded task '{task.name}' from {task_yaml_path}")

        except ValidationError as e:
            logger.warning(f"Invalid task definition at {task_yaml_path}: {e}")
        except yaml.YAMLError as e:
            logger.warning(f"Failed to parse YAML at {task_yaml_path}: {e}")

    logger.info(f"Loaded {len(tasks)} task(s) from {tasks_dir}")
    return tasks


def load_benchmark(benchmark_file: Path) -> BenchmarkDefinition:
    """Load a benchmark definition from a YAML file.

    Args:
        benchmark_file: Path to the benchmark YAML file.

    Returns:
        BenchmarkDefinition parsed from the file.

    Raises:
        FileNotFoundError: If the benchmark file does not exist.
        ValidationError: If the benchmark definition is invalid.
        yaml.YAMLError: If the YAML is malformed.
    """
    if not benchmark_file.exists():
        raise FileNotFoundError(f"Benchmark file {benchmark_file} does not exist.")

    with benchmark_file.open(encoding="utf-8") as f:
        benchmark_data = yaml.safe_load(f) or {}
    return BenchmarkDefinition(**benchmark_data)


def _resolve_installation_file_mappings(
    mappings: list[InstallationFileMapping], base_dir: Path
) -> list[InstallationFileMapping]:
    """Resolve relative source paths in InstallationFileMapping to absolute paths."""
    resolved = []
    for mapping in mappings:
        if mapping.source.is_absolute():
            resolved.append(mapping)
        else:
            resolved.append(
                InstallationFileMapping(
                    source=base_dir / mapping.source,
                    dest=mapping.dest,
                )
            )
    return resolved
