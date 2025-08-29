# Copyright (c) Microsoft. All rights reserved.

from typing import Literal

from liquid import render
from loguru import logger
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.responses import EasyInputMessageParam, ResponseInputParam
from pydantic import BaseModel, Field

from eval_recipes.evaluations.guidance.prompts import (
    CAPABILITY_MANIFEST_SYSTEM_PROMPT,
    CAPABILITY_MANIFEST_USER_PROMPT,
    EVAL_SYSTEM_PROMPT,
    EVAL_USER_PROMPT,
    SCOPE_SYSTEM_PROMPT,
    SCOPE_USER_PROMPT,
)
from eval_recipes.schemas import BaseEvaluatorConfig, EvaluationOutput
from eval_recipes.utils.llm import create_client
from eval_recipes.utils.responses_conversion import format_full_history

SCORE_MIN = 0
SCORE_MAX = 100


class GuidanceEvaluatorConfig(BaseEvaluatorConfig):
    capability_manifest: str | None = Field(
        default=None,
        description="""A description of the assistant's capabilities.
If not provided, it will be re-computed each time so it is highly recommended to pre-compute it.""",
    )
    in_scope_probability_threshold: float = Field(default=50)


class InputGuidanceEval(BaseModel):
    conversation_history_beginning_turn: str
    conversation_history_full: str
    tools: list[ChatCompletionToolParam] = []


# This is for Structured Outputs.
class IsInScopeReasoning(BaseModel):
    capabilities_analysis: str = Field(
        description="Detailed reasoning about the capabilities available to accomplish the user's task"
    )
    sufficient_capabilities: str = Field(
        description="""Detailed reasoning about whether the available capabilities are sufficient to accomplish the user's task. \
Be sure to always include the capabilities that could be included to more efficiently or effectively accomplish the task."""
    )
    in_scope_reasoning: str = Field(
        description="Detailed reasoning about whether the assistant could accomplish the user's task with the available capabilities to a high degree"
    )
    is_in_scope_probability: float = Field(
        description=f"""Probability from {SCORE_MIN} to {SCORE_MAX} that the user's request can be accomplished \
to a high degree by the assistant given the capabilities it has"""
    )


# This is for Structured Outputs.
class GuidanceEvalResult(BaseModel):
    reasoning: str = Field(
        description="""Detailed natural-language justification describing how the assistant handled the out-of-scope request, \
referencing the capability manifest and assistant messages."""
    )
    score: int = Field(
        description=f"""Integer between {SCORE_MIN} and {SCORE_MAX}, where {SCORE_MAX} indicates the assistant promptly \
and clearly guided the user or let them know of its inability to fulfill their task, and lower values denote increasingly ungraceful failures."""
    )


class OutputGuidanceEval(BaseModel):
    is_in_scope_reasoning: IsInScopeReasoning
    applicable: bool
    reasoning: str
    score: float


class GuidanceEvaluator:
    def __init__(self, config: GuidanceEvaluatorConfig | None = None) -> None:
        """
        Measures the proportion of out-of-scope scenarios in which the assistant,
        given a capability manifest, promptly and clearly guided the user or let them know of
        its inability to fulfill their task, rather than attempting unsupported actions
        that lead to hallucinations or confusing a user.

        Args:
            config: Optional GuidanceEvaluatorConfig. If not provided, defaults will be used.
        """
        self.config = config or GuidanceEvaluatorConfig()

    async def evaluate(self, messages: ResponseInputParam, tools: list[ChatCompletionToolParam]) -> EvaluationOutput:
        input_data = InputGuidanceEval(
            conversation_history_full=format_full_history(messages, remove_system_messages=False),
            conversation_history_beginning_turn=format_full_history(
                messages, remove_system_messages=True, only_upto_last_user=True
            ),
            tools=tools,
        )
        results = await self.run(input=input_data)
        output = EvaluationOutput(
            eval_name="guidance",
            applicable=results.applicable,
            score=results.score,
            feedback=self._feedback(results),
            metadata={
                "in_scope_reasoning": results.is_in_scope_reasoning.model_dump(mode="json"),
                "reasoning": results.reasoning,
            },
        )
        return output

    async def run(self, input: InputGuidanceEval) -> OutputGuidanceEval:
        # If capability manifest is not provided, generate it
        if not self.config.capability_manifest:
            logger.warning(
                "Generating capability manifest. This will take extra time and may perform worse than providing a pre-computed manifest in the config."
            )
            self.config.capability_manifest = await generate_capability_manifest(
                system_prompt=input.conversation_history_beginning_turn,
                tools=input.tools,
                provider=self.config.provider,
                model=self.config.model,
            )

        # First determine if the request is in scope or not
        is_in_scope_reasoning = await self._evaluate_scope(input)
        if is_in_scope_reasoning.is_in_scope_probability >= self.config.in_scope_probability_threshold:
            # If the request is in scope, we can skip the graceful failure evaluation
            return OutputGuidanceEval(
                is_in_scope_reasoning=is_in_scope_reasoning,
                reasoning="Request is in scope - no graceful failure evaluation needed",
                applicable=False,
                score=-1,
            )

        # Request is out of scope, so evaluate how gracefully it was handled
        graceful_fails_result = await self._evaluate_graceful_failure(input)
        validated_result = self._validate_result(graceful_fails_result)
        return OutputGuidanceEval(
            is_in_scope_reasoning=is_in_scope_reasoning,
            reasoning=validated_result.reasoning,
            applicable=True,
            score=validated_result.score,
        )

    def _validate_result(self, result: GuidanceEvalResult) -> GuidanceEvalResult:
        """Validate and fix the evaluation result.

        Applies these validation rules:
        - Clamps score to the valid range [SCORE_MIN, SCORE_MAX]
        - Ensures reasoning is not empty
        """
        clamped_score = max(SCORE_MIN, min(SCORE_MAX, result.score))
        reasoning = result.reasoning if result.reasoning.strip() else "No reasoning provided"
        return GuidanceEvalResult(reasoning=reasoning, score=clamped_score)

    async def _evaluate_scope(self, input: InputGuidanceEval) -> IsInScopeReasoning:
        """Determine if the request is in scope or not given the assistant's capabilities."""
        user_prompt = render(
            SCOPE_USER_PROMPT,
            conversation_history=input.conversation_history_beginning_turn,
            capability_manifest=self.config.capability_manifest,
        )
        messages: list = [
            EasyInputMessageParam(role="system", content=SCOPE_SYSTEM_PROMPT),
            EasyInputMessageParam(role="user", content=user_prompt),
        ]

        async with create_client(provider=self.config.provider) as client:
            response = await client.responses.parse(
                model=self.config.model,
                input=messages,
                text_format=IsInScopeReasoning,
                store=False,
            )

        if response.output_parsed is None:
            return IsInScopeReasoning(
                capabilities_analysis="Failed to parse capabilities analysis",
                sufficient_capabilities="Failed to parse sufficient capabilities",
                in_scope_reasoning="Failed to parse in scope reasoning",
                is_in_scope_probability=0,
            )
        return response.output_parsed

    async def _evaluate_graceful_failure(self, input: InputGuidanceEval) -> GuidanceEvalResult:
        """Determine if the assistant gracefully handled the out-of-scope request."""
        user_prompt = render(
            EVAL_USER_PROMPT,
            conversation_history=input.conversation_history_full,
            capability_manifest=self.config.capability_manifest,
        )

        messages: list = [
            EasyInputMessageParam(role="system", content=EVAL_SYSTEM_PROMPT),
            EasyInputMessageParam(role="user", content=user_prompt),
        ]

        async with create_client(provider=self.config.provider) as client:
            response = await client.responses.parse(
                model=self.config.model,
                input=messages,
                text_format=GuidanceEvalResult,
            )

        if response.output_parsed is None:
            return GuidanceEvalResult(reasoning="Failed to parse evaluation result", score=SCORE_MIN)
        return response.output_parsed

    def _feedback(self, results: OutputGuidanceEval, score_threshold: float = 50) -> str | None:
        """If the request was out of scope and the assistant handled it poorly (below score_threshold),
        then return the reasoning as the feedback.
        """
        if results.applicable:
            if results.score <= score_threshold:
                return results.reasoning
            else:
                return None


# region Capability Manifest Generation


async def generate_capability_manifest(
    system_prompt: str,
    tools: list[ChatCompletionToolParam] | None = None,
    provider: Literal["openai", "azure_openai"] = "openai",
    model: str = "gpt-5",
) -> str:
    """
    Often the raw system prompt and tools contain enough extra noise that it confuses the LLM.
    This is a preprocessing step you can use to generate a clean capability manifest
    that can be reused for each usage of the GuidanceEvaluator.
    """
    tools_description = "None"
    if tools:
        tools_formatted = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                name = func.get("name", "Unknown")
                description = func.get("description", "No description")
                tools_formatted.append(f"- {name}: {description}")

        if tools_formatted:
            tools_description = "\n".join(tools_formatted)

    user_prompt = render(
        CAPABILITY_MANIFEST_USER_PROMPT,
        system_prompt=system_prompt,
        tools_description=tools_description,
    )

    messages: list = [
        EasyInputMessageParam(role="system", content=CAPABILITY_MANIFEST_SYSTEM_PROMPT),
        EasyInputMessageParam(role="user", content=user_prompt),
    ]
    async with create_client(provider=provider) as client:
        response = await client.responses.create(
            model=model,
            input=messages,
            store=False,
        )

    if response.output_text is None:
        return ""
    return response.output_text


# endregion
