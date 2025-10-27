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
    categories: list[str] = []  # Category tags for the task like "finance"


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


class TrialResult(BaseModel):
    trial_number: int
    score: float
    metadata: dict[str, Any] = {}
    test_output: str

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        """Clamp score to 0-100 range."""
        return max(0.0, min(100.0, v))


class AggregatedTaskResult(BaseModel):
    task_name: str
    agent_name: str
    num_trials: int
    trials: list[TrialResult]
    mean_score: float
    median_score: float
    std_dev: float
    min_score: float
    max_score: float
    num_perfect_trials: int  # How many trials got 100%

    @field_validator("mean_score", "median_score", "min_score", "max_score")
    @classmethod
    def validate_scores(cls, v: float) -> float:
        """Clamp scores to 0-100 range."""
        return max(0.0, min(100.0, v))

    @field_validator("std_dev")
    @classmethod
    def validate_std_dev(cls, v: float) -> float:
        """Ensure std_dev is non-negative."""
        return max(0.0, v)
