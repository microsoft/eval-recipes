# Copyright (c) Microsoft. All rights reserved.


from typing import Any, Literal, Protocol, runtime_checkable

from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.responses import ResponseInputParam
from pydantic import BaseModel


class EvaluationOutput(BaseModel):
    eval_name: str  # Name of the evaluation
    applicable: bool  # Was the evaluation applicable
    score: float  # Score from 0 to 100
    feedback: str | None = None  # Feedback that can be used as part of another system to self-improve the response.
    metadata: dict[str, Any] = {}  # Any additional metadata the evaluation may generate.


# region Eval Configurations


class BaseEvaluatorConfig(BaseModel):
    provider: Literal["openai", "azure_openai"] = "openai"
    model: Literal["gpt-5", "gpt-5-mini", "gpt-5-nano", "o3", "o4-mini"] = "gpt-5"


# endregion


@runtime_checkable
class EvaluatorProtocol(Protocol):
    """Protocol for custom evaluator classes."""

    def __init__(self, config: BaseEvaluatorConfig | None = None) -> None:
        """Initialize the evaluator with an optional configuration.
        If config is not provided, it should be instantiated with defaults.
        """
        ...

    async def evaluate(self, messages: ResponseInputParam, tools: list[ChatCompletionToolParam]) -> EvaluationOutput:
        """Evaluate messages and tools, returning an EvaluationOutput."""
        ...
