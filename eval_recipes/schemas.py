# Copyright (c) Microsoft. All rights reserved.


from typing import Any, Literal, Protocol, runtime_checkable

from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.responses import ResponseInputParam
from pydantic import BaseModel, Field


class EvaluationOutput(BaseModel):
    eval_name: str  # Name of the evaluation
    applicable: bool  # Was the evaluation applicable
    score: float  # Score from 0 to 100
    feedback: str | None = (
        None  # Feedback that can be used as part of another system to self-improve the response.
    )
    metadata: dict[
        str, Any
    ] = {}  # Any additional metadata the evaluation may generate.


# region Eval Configurations


class BaseEvaluationConfig(BaseModel):
    provider: Literal["openai", "azure_openai"] = "openai"
    model: Literal["gpt-5", "gpt-5-mini", "gpt-5-nano", "o3", "o4-mini"] = "gpt-5"


class GuidanceEvaluationConfig(BaseEvaluationConfig):
    capability_manifest: str | None = Field(
        default=None,
        description="""A description of the assistant's capabilities.
If not provided, it will be re-computed each time so it is highly recommended to pre-compute it.""",
    )
    in_scope_probability_threshold: float = Field(default=50)


class ToolEvaluationConfig(BaseEvaluationConfig):
    tool_thresholds: dict[str, float] = Field(
        default={},
        description="A dictionary mapping tool names to the threshold probabilities indicated that the tool should be called.",
    )


class ClaimVerifierConfig(BaseEvaluationConfig):
    claim_extraction_model: Literal[
        "gpt-5", "gpt-5-mini", "gpt-5-nano", "o3", "o4-mini"
    ] = Field(default="gpt-5-mini")
    verification_model: Literal[
        "gpt-5", "gpt-5-mini", "gpt-5-nano", "o3", "o4-mini"
    ] = Field(default="gpt-5")
    verification_reasoning_effort: Literal["minimal", "low", "medium", "high"] = Field(
        default="high"
    )
    max_line_length: int = Field(default=200)
    max_concurrency: int = Field(default=1)
    ignore_tool_names: list[str] = Field(
        default=[],
        description="A list of tool names to ignore as source context.",
    )


# endregion


@runtime_checkable
class EvaluatorProtocol(Protocol):
    """Protocol for custom evaluator classes."""

    def __init__(self, config: BaseEvaluationConfig | None = None) -> None:
        """Initialize the evaluator with an optional configuration.
        If config is not provided, it should be instantiated with defaults.
        """
        ...

    async def evaluate(
        self, messages: ResponseInputParam, tools: list[ChatCompletionToolParam]
    ) -> EvaluationOutput:
        """Evaluate messages and tools, returning an EvaluationOutput."""
        ...
