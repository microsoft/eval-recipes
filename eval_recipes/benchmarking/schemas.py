# Copyright (c) Microsoft. All rights reserved.

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, field_validator


class AgentConfig(BaseModel):
    name: str
    required_env_vars: list[str] = []  # List of required environment variables
    agent_installation: str  # Docker commands to install the agent
    command_template: str  # Command Liquid template with placeholders like {{task_instructions}}


class TaskInfo(BaseModel):
    difficulty: Literal["easy", "medium", "hard"]
    non_deterministic_evals: bool = False  # Whether the task evaluations are non-deterministic


class TaskConfig(BaseModel):
    name: str
    required_env_vars: list[str] = []  # List of required environment variables
    task_installation: str  # Docker commands to install the task
    instructions: str  # Instructions text for the agent
    test_script: Path  # Path to test.py script
    test_command: str = "uv run --no-project /project/test.py"  # Command to run tests
    data_dir: Path | None = None  # Optional path to data directory
    timeout: int = 600  # Timeout in seconds
    task_info: TaskInfo


class TestResult(BaseModel):
    score: float
    metadata: dict[str, Any] = {}
    test_output: str  # Full output from test execution

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        """Clamp score to 0-100 range."""
        return max(0.0, min(100.0, v))
