# Copyright (c) Microsoft. All rights reserved.

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, field_validator

# region Definitions


class InstallationFileMapping(BaseModel):
    """Maps a local directory to a container directory for file copying."""

    source: Path  # Local path (relative paths resolved from agent directory, or absolute paths)
    dest: str  # Absolute path in container


class AgentDefinition(BaseModel):
    id: str  # unique id for the agent, usually corresponding to the directory name
    agent_name: str  # Allows for grouping multiple versions of the same agent.
    dockerfile_portion: str = ""  # Docker commands to install the task
    installation_files: list[InstallationFileMapping] = []  # Files to copy into container before agent installation
    runtime_files: list[InstallationFileMapping] = []  # Files to copy into container after agent installation
    command_template: str  # Command to send to the agent to complete a task. Liquid template with a placeholder for {{task_instructions}}
    command_template_continue: str | None = (
        None  # Command to send to the agent to continue a conversation. Liquid template with a placeholder for {{task_instructions}}. (Optional)
    )
    agent_logs_paths: list[str] = []  # List of paths where the agent stores logs in the container
    source_code_path: str | None  # Location of the agent's source code in the container, if applicable


class TaskInfo(BaseModel):
    difficulty: Literal["easy", "medium", "hard"]
    non_deterministic_evals: bool = False  # Whether the task evaluations are non-deterministic
    categories: list[str] = []  # Category tags for the task like "finance"


class ScoreEvalConfig(BaseModel):
    type: Literal["score"] = "score"
    test_script: Path  # Path to test.py script
    test_command: str = "uv run --no-project /project/test.py"  # Command to run tests


class ComparisonEvalConfig(BaseModel):
    type: Literal["comparison"] = "comparison"
    guidelines: str | None = None  # Task-specific evaluation guidelines (e.g., how to evaluate a PPT)


EvaluationConfig = ScoreEvalConfig | ComparisonEvalConfig


class TaskDefinition(BaseModel):
    name: str  # Name of the task, usually corresponding to the directory name
    task_info: TaskInfo  # Metadata about the task
    evaluation_configs: list[EvaluationConfig] = []  # Evaluation configuration for the task
    dockerfile_portion: str = ""  # Docker commands to install the task
    instructions: str | None  # Task instructions for the agent (optional)
    task_time_files: list[
        InstallationFileMapping
    ] = []  # Files to copy into container before agent runs, specific to the task
    test_time_files: list[
        InstallationFileMapping
    ] = []  # Files to copy into container before evaluations run, specific to the task
    timeout: int = 1800  # Timeout in seconds for agent to complete the task


class ScoreBenchmarkAgentDefinition(BaseModel):
    agent_id: str
    task_names: list[str]
    trials: int = 1


class ScoreBenchmarkDefinition(BaseModel):
    benchmark_type: str = "score"
    continuation_provider: Literal["openai", "azure_openai", "none"] = "none"
    continuation_model: str = "gpt-5.1"
    score_benchmarks: list[ScoreBenchmarkAgentDefinition]
    analysis_score_threshold: float = 85.0  # Skip analysis if score >= threshold


class ComparisonBenchmarkAgentDefinition(BaseModel):
    task_name: str
    agent_ids: list[str]


class ComparisonBenchmarkDefinition(BaseModel):
    benchmark_type: str = "comparison"
    continuation_provider: Literal["openai", "azure_openai", "none"] = "none"
    continuation_model: str = "gpt-5.1"
    comparison_benchmarks: list[ComparisonBenchmarkAgentDefinition]
    comparison_runs: int = 7  # Number of comparison runs per task for consistency measurement


class BenchmarkDefinition(BaseModel):
    """Combined benchmark definition for running both score and comparison benchmarks."""

    score_benchmark: ScoreBenchmarkDefinition | None = None
    comparison_benchmark: ComparisonBenchmarkDefinition | None = None


# endregion

# region Common Job Schemas


class ExecuteAgentJobInput(BaseModel):
    agent: AgentDefinition
    task: TaskDefinition
    trial_number: int
    continuation_provider: Literal["openai", "azure_openai", "none"] = "none"
    continuation_model: str = "gpt-5.1"


class ExecuteAgentJobOutput(BaseModel):
    container_id: str  # Docker container ID for subsequent jobs
    image_tag: str  # Docker image tag (needed for cleanup)
    agent_console_log: str
    agent_duration_seconds: float
    continuation_occurred: bool = False
    continuation_prompt: str | None = None
    continuation_error: str | None = None


# endregion

# region Score Based Evaluation Schemas


class ExecuteEvaluationsJobInput(BaseModel):
    task: TaskDefinition
    trial_number: int
    agent_log_hint: str | None = None  # From AgentDefinition.agent_logs_paths


class EvaluateJobOutput(BaseModel):
    score: float
    rubric: dict[str, Any] = {}
    test_console_log: str
    test_duration_seconds: float


class TaskAnalysisJobInput(BaseModel):
    task: TaskDefinition
    trial_number: int
    analysis_score_threshold: float = 85.0


class TaskAnalysisJobOutput(BaseModel):
    valid_trial: bool  # False if failure was infrastructure-related (Docker, API, timeout)
    analysis_skipped: bool  # True if score >= threshold
    failure_report: str | None = None  # Markdown report (None if skipped)
    failure_category: str | None = None  # e.g., "agent_error", "infrastructure_error", "test_issue"


class TrialExecutionJobInput(BaseModel):
    """Combined input for executing a complete trial (agent execution, evaluation, and analysis)."""

    agent: AgentDefinition
    task: TaskDefinition
    trial_number: int
    continuation_provider: Literal["openai", "azure_openai", "none"] = "none"
    continuation_model: str = "gpt-5.1"
    agent_log_hint: str | None = None
    analysis_score_threshold: float = 85.0


class TrialExecutionJobOutput(BaseModel):
    # From ExecuteAgentJobOutput
    agent_console_log: str
    agent_duration_seconds: float
    continuation_occurred: bool = False
    continuation_prompt: str | None = None
    continuation_error: str | None = None

    # From EvaluateJobOutput
    score: float
    rubric: dict[str, Any] = {}
    test_console_log: str
    test_duration_seconds: float

    # From TaskAnalysisJobOutput
    valid_trial: bool
    analysis_skipped: bool
    failure_report: str | None = None
    failure_category: str | None = None


class FinalAnalysisJobInput(BaseModel):
    agent_id: str
    provider: Literal["openai", "azure_openai"] = "openai"
    model: str = "gpt-5.2"


class FinalAnalysisJobOutput(BaseModel):
    executive_summary_path: str  # Path to executive summary markdown
    full_report_path: str  # Path to full report markdown
    executive_summary: str  # Content for easy access
    full_report: str  # Content for easy access
    report_generated: bool  # False if no failure reports to analyze
    num_reports_analyzed: int


class AgentComparisonJobInput(BaseModel):
    provider: Literal["openai", "azure_openai"] = "openai"
    model: str = "gpt-5.2"


class AgentComparisonJobOutput(BaseModel):
    executive_summary: str  # Executive summary content
    full_report: str  # Full comparison report content
    executive_summary_path: str = ""  # Path to executive summary (empty if skipped)
    full_report_path: str = ""  # Path to full report (empty if skipped)
    num_agents_compared: int = 0  # Number of agents with reports


# endregion

# region Comparison Based Evaluation Schemas


class ExtractProjectJobInput(BaseModel):
    agent_id: str
    task_name: str
    trial_number: int


class ExtractProjectJobOutput(BaseModel):
    project_dir: str  # Absolute path to extracted project directory
    agent_id: str
    task_name: str


class ComparisonTrialJobInput(BaseModel):
    agent: AgentDefinition
    task: TaskDefinition
    trial_number: int = 1  # Usually 1 for comparisons
    continuation_provider: Literal["openai", "azure_openai", "none"] = "none"
    continuation_model: str = "gpt-5.1"


class ComparisonTrialJobOutput(BaseModel):
    # From ExecuteAgentJobOutput
    agent_console_log: str
    agent_duration_seconds: float
    continuation_occurred: bool = False
    continuation_prompt: str | None = None
    continuation_error: str | None = None

    # From ExtractProjectJobOutput
    project_dir: str  # Absolute path to extracted project directory
    agent_id: str
    task_name: str


class ComparisonResult(BaseModel):
    reasoning: str  # Qualitative analysis explaining the comparison (may contain anonymous agent names)
    rankings: list[int]  # Ordered list of directory indices, best to worst
    anonymous_to_index: dict[str, int]  # Maps anonymous names (e.g., "agent_eac8") to directory indices


class SemanticComparisonJobInput(BaseModel):
    task_name: str
    task_instructions: str
    comparison_run_number: int  # For multiple runs (consistency)
    guidelines: str | None = None


class SemanticComparisonJobOutput(BaseModel):
    task_name: str
    comparison_run_number: int
    reasoning: str
    rankings: dict[str, int]  # agent_id -> rank (1 = best)
    anonymous_to_agent_id: dict[str, str]  # Maps anon names to agent_ids


class ComparisonAggregationJobInput(BaseModel):
    task_name: str
    task_instructions: str
    provider: Literal["openai", "azure_openai"] = "openai"
    model: str = "gpt-5.2"


class ComparisonAggregationJobOutput(BaseModel):
    task_name: str
    analysis_report: str
    report_path: str  # Path to saved markdown file
    num_comparisons_analyzed: int


class ComparisonFinalAnalysisJobInput(BaseModel):
    provider: Literal["openai", "azure_openai"] = "openai"
    model: str = "gpt-5.2"


class ComparisonFinalAnalysisJobOutput(BaseModel):
    analysis_report: str
    report_path: str  # Path to COMPARISON_FINAL_REPORT.md
    num_tasks_analyzed: int


# endregion


# region Results Aggregation Schemas


class SemanticTestResult(BaseModel):
    score: float  # Scores must be between 0 and 100
    metadata: dict[str, Any]  # All other fields from the rubric

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        """Clamp score to 0-100 range."""
        return max(0.0, min(100.0, v))


class TrialMetrics(BaseModel):
    """Metrics for a single trial."""

    trial_number: int
    score: float
    agent_duration_seconds: float | None
    test_duration_seconds: float | None
    valid_trial: bool
    failure_category: str | None
    failure_report_path: str | None  # Relative path to FAILURE_REPORT.md
    rubric: dict[str, Any]
    logs: dict[str, str]  # {"build": "...", "agent": "...", "test": "..."}
    project_zip_path: str | None  # Relative path to project.zip


class TaskMetrics(BaseModel):
    task_name: str
    instructions: str
    task_info: TaskInfo  # difficulty, categories, etc.

    # Aggregated metrics (excluding invalid trials)
    num_trials: int
    num_valid_trials: int
    mean_score: float
    std_dev: float
    min_score: float
    max_score: float
    median_score: float
    num_perfect_trials: int  # score == 100

    mean_agent_duration_seconds: float | None
    mean_test_duration_seconds: float | None

    trials: list[TrialMetrics]


class AgentMetrics(BaseModel):
    agent_id: str
    agent_name: str

    # Aggregated metrics
    num_unique_tasks: int
    total_trials: int
    total_valid_trials: int

    mean_score: float  # Mean of task mean scores
    variability: float  # Mean of task std_devs
    consistency_rate: float  # % of tasks with std_dev < 10
    mean_agent_duration_seconds: float | None

    # Report paths (relative)
    executive_summary_path: str | None
    full_report_path: str | None

    tasks: list[TaskMetrics]


class AgentSummary(BaseModel):
    agent_id: str
    agent_name: str
    mean_score: float
    variability: float
    consistency_rate: float


class BenchmarkSummary(BaseModel):
    benchmark_timestamp: str
    total_agents: int
    total_tasks: int
    total_trials: int

    agent_summaries: list[AgentSummary]


class BenchmarkManifest(BaseModel):
    benchmark_timestamp: str
    benchmark_log_path: str

    # Agent comparison (if 2+ agents)
    comparison_executive_summary: str | None
    comparison_full_report: str | None
    comparison_executive_summary_path: str | None
    comparison_full_report_path: str | None

    agents: list[AgentMetrics]


class ResultsAggregationJobInput(BaseModel):
    include_project_zips: bool = True
    include_logs: bool = True


class ResultsAggregationJobOutput(BaseModel):
    manifest_path: str
    summary_path: str
    results_dir: str
    html_report_path: str | None = None


# endregion

# region Comparison Results Aggregation Schemas


class ComparisonTrialData(BaseModel):
    comparison_run_number: int
    rankings: dict[str, int]  # agent_id -> rank (1 = best)
    reasoning: str


class ComparisonTaskMetrics(BaseModel):
    task_name: str
    task_instructions: str
    task_info: TaskInfo  # difficulty, categories

    # Per-agent rankings across all comparison runs
    agent_ranks: dict[str, list[int]]  # agent_id -> [rank1, rank2, ...]

    # Computed metrics
    agent_avg_rank: dict[str, float]  # agent_id -> average rank
    agent_win_rate: dict[str, float]  # agent_id -> win rate (0-100)
    agreement_kendalls_w: float | None  # Kendall's W for this task

    # LLM analysis from ComparisonAggregationJob
    aggregate_analysis: str
    aggregate_analysis_path: str | None

    # Trial details
    trials: list[ComparisonTrialData]

    # Project zips (agent_id -> relative path)
    project_zip_paths: dict[str, str]


class ComparisonOverviewMetrics(BaseModel):
    agent_avg_rank: dict[str, float]  # Overall average rank
    agent_win_rate: dict[str, float]  # Overall win rate (0-100)
    agent_task_wins: dict[str, int]  # Number of tasks won
    task_ties: int  # Number of tied tasks
    mean_kendalls_w: float | None


class ComparisonBenchmarkManifest(BaseModel):
    benchmark_timestamp: str
    benchmark_log_path: str | None

    # Participating agents
    agent_ids: list[str]

    # Overview metrics
    overview: ComparisonOverviewMetrics

    # Final analysis
    final_analysis_report: str | None
    final_analysis_report_path: str | None

    # Per-task data
    tasks: list[ComparisonTaskMetrics]


class ComparisonBenchmarkSummary(BaseModel):
    benchmark_timestamp: str
    num_tasks: int
    num_comparison_runs_per_task: int
    agent_ids: list[str]

    # Key metrics
    agent_avg_rank: dict[str, float]
    agent_win_rate: dict[str, float]
    agent_task_wins: dict[str, int]
    mean_kendalls_w: float | None


class ComparisonResultsAggregationJobInput(BaseModel):
    include_project_zips: bool = True


class ComparisonResultsAggregationJobOutput(BaseModel):
    manifest_path: str
    summary_path: str
    results_dir: str
    html_report_path: str | None = None


# endregion
