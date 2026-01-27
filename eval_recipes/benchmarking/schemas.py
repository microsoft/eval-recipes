# Copyright (c) Microsoft. All rights reserved.

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, field_validator, model_validator

# region Common Configuration Schemas


class AgentConfig(BaseModel):
    name: str
    required_env_vars: list[str] = []  # List of required environment variables
    agent_installation: str  # Docker commands to install the agent
    command_template: str  # Command Liquid template with placeholders like {{task_instructions}}
    command_template_continue: str | None = None  # Optional command Liquid template for continuing agent conversation
    data_dir: Path | None = None  # Optional path to agent data directory
    local_source_path: Path | None = None  # Optional path to local agent source code for development
    agent_log_hint: str | None = None  # Optional hint for where agent stores logs in the container


class TaskInfo(BaseModel):
    difficulty: Literal["easy", "medium", "hard"]
    non_deterministic_evals: bool = False  # Whether the task evaluations are non-deterministic
    categories: list[str] = []  # Category tags for the task like "finance"


class TaskConfig(BaseModel):
    name: str
    eval_type: Literal["score", "comparison", "both"]
    required_env_vars: list[str] = []  # List of required environment variables
    task_installation: str  # Docker commands to install the task
    instructions: str  # Instructions text for the agent
    task_time_data_dir: Path | None = None  # Optional path to task-time data directory (copied before agent runs)
    test_time_data_dir: Path | None = None  # Optional path to test-time data directory (copied before tests run)
    timeout: int = 1800  # Timeout in seconds
    task_info: TaskInfo

    # Optional evaluation configs (populated based on eval_type)
    score_eval: "ScoreEvalConfig | None" = None
    comparison_eval: "ComparisonEvalConfig | None" = None

    @model_validator(mode="after")
    def validate_eval_configs(self) -> "TaskConfig":
        """Ensure the appropriate eval config is present based on eval_type."""
        if self.eval_type == "score" and self.score_eval is None:
            raise ValueError("score_eval must be set when eval_type is 'score'")
        if self.eval_type == "comparison" and self.comparison_eval is None:
            raise ValueError("comparison_eval must be set when eval_type is 'comparison'")
        if self.eval_type == "both":
            if self.score_eval is None:
                raise ValueError("score_eval must be set when eval_type is 'both'")
            if self.comparison_eval is None:
                raise ValueError("comparison_eval must be set when eval_type is 'both'")
        return self


# endregion


# region Score-Based Schemas


class ScoreEvalConfig(BaseModel):
    """Configuration for score-based evaluation."""

    test_script: Path  # Path to test.py script
    test_command: str = "uv run --no-project /project/test.py"  # Command to run tests


class TrialResult(BaseModel):
    trial_number: int
    score: float
    metadata: dict[str, Any] = {}
    test_output: str
    agent_duration_seconds: float | None = None
    test_duration_seconds: float | None = None

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
    mean_agent_duration_seconds: float | None = None
    median_agent_duration_seconds: float | None = None
    mean_test_duration_seconds: float | None = None
    median_test_duration_seconds: float | None = None

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


class SemanticTestResult(BaseModel):
    """Result from semantic_test() which uses an LLM agent to audit another agent's work.

    Used by: eval_recipes.benchmarking.semantic_test.semantic_test()
    """

    score: float  # Scores must be between 0 and 100
    metadata: dict[str, Any]  # All other fields from the rubric

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        """Clamp score to 0-100 range."""
        return max(0.0, min(100.0, v))


# endregion


# region Comparison-Based Schemas


class ComparisonEvalConfig(BaseModel):
    """Configuration for comparison-based evaluation."""

    guidelines: str | None = None  # Task-specific evaluation guidelines (e.g., how to evaluate a PPT)


class ComparisonTaskSpec(BaseModel):
    """Specification for a comparison task with explicit agent associations.

    Designed for mixed usage:
    - Can be loaded from YAML files (task.yaml with 'agents' field)
    - Can be defined programmatically in Python
    """

    task_name: str  # Name of the task (maps to task directory)
    agent_names: list[str]  # Names of agents to compare (maps to agents_dir)


class ComparisonTaskConfig(BaseModel):
    """Fully resolved comparison task configuration with loaded configs."""

    task: TaskConfig
    agents: list[AgentConfig]
    guidelines: str | None = None


class ComparisonResult(BaseModel):
    """Result from semantic_test_comparison() which compares multiple agents' outputs on the same task.

    Used by: eval_recipes.benchmarking.semantic_test.semantic_test_comparison()
    """

    reasoning: str  # Qualitative analysis explaining the comparison (may contain anonymous agent names)
    rankings: list[int]  # Ordered list of directory indices, best to worst
    anonymous_to_index: dict[str, int]  # Maps anonymous names (e.g., "agent_eac8") to directory indices


class ComparisonRunResult(BaseModel):
    """Result of a single semantic_test_comparison run."""

    task_name: str
    comparison_run_num: int
    result: ComparisonResult  # From semantic_test_comparison
    agent_names: list[str]  # Agent names in order matching result.rankings indices


class ComparisonBenchmarkResults(BaseModel):
    """All results from a comparison benchmark run."""

    comparison_runs: list[ComparisonRunResult]


class AggregateReport(BaseModel):
    """LLM-generated summary of comparison results for a task."""

    task_name: str
    agent_names: list[str]
    analysis: str  # Plain text analysis explaining why rankings occurred


class AggregateReportResult(BaseModel):
    """Result from aggregate report job."""

    task_name: str
    comparison_folder_name: str
    report: AggregateReport


class FinalAggregateReport(BaseModel):
    """LLM-generated final summary across all tasks."""

    agent_names: list[str]
    analysis: str  # Paragraph + bullet points explaining overall results


# endregion


# region Score Run Definition Schemas


class ScoreTaskSpec(BaseModel):
    """Specification for a task within a score run definition."""

    task: str  # Task name (maps to task directory)
    trials: int | None = None  # Optional trial count override


class ScoreAgentSpec(BaseModel):
    """Specification for an agent and its tasks in a score run definition."""

    agent: str  # Agent name (maps to agent directory)
    trials: int | None = None  # Default trial count for this agent's tasks
    tasks: list[ScoreTaskSpec]


class ScoreRunSpec(BaseModel):
    """Run definition for score-based benchmarking."""

    type: Literal["score"] = "score"
    definitions: list[ScoreAgentSpec]


# endregion
