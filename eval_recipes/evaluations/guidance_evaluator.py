# Copyright (c) Microsoft. All rights reserved.

from typing import Literal

from liquid import render
from loguru import logger
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.responses import EasyInputMessageParam, ResponseInputParam
from pydantic import BaseModel, Field

from eval_recipes.schemas import EvaluationOutput, GuidanceEvaluationConfig
from eval_recipes.utils.llm import create_client
from eval_recipes.utils.responses_conversion import format_full_history

SCORE_MIN = 0
SCORE_MAX = 100


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


SCOPE_SYSTEM_PROMPT = """You are an evaluator that is determining if the assistant has all of the required tools \
and capabilities needed to accomplish the user's request.

You will be provided the history of the interactions between the assistant and user.
You should base your evaluation primarily on the user's latest message, \
however the remainder of the conversation history is provided for additional context. \
You should not let historical out of scope and poorly accomplished actions influence your evaluation that recent actions are correct.
You will be also provided a "capability manifest" that describes what the assistant can and cannot do. \
This is often the system prompt and tool definitions of the assistant. \
This means that it might contain extraneous information that is not just the rules/capabilities. \
Thus, you should focus solely on understanding what the assistant's capabilities are. \
IMPORTANT: You **must** take the tool definitions and instructions at face value and NEVER infer additional capabilities.

To determine if the user's task is out of scope, you will use the following reasoning process. \
You must sequentially reason through each of the following steps, answering the questions in order.

1. Determine available capabilities to accomplish task.
- Analyze at the user's lastest request. What are they ideally asking the assistant to accomplish for them?
- Now look at the capability manifest, which capabilities does the assistant have that could be used to accomplish this task?
  - If there are no capabilities that could be used to accomplish the task, then the request is out of scope. \
Skip the rest of the reasoning steps and mark the response as out of scope. Otherwise, continue to the next step.

2. Does the assistant have *all* the capabilities to accomplish the task?
- Will those capabilities accomplish the user's request to the fullest extent or would it be a compromise or suboptimal in some way?
- For each capability, **always** consider if is there another tool/capability that could be \
provided to the assistant that would allow it to accomplish the task better and/or more *efficiently*.
  - If there are no additional capabilities that could improve the assistant's performance then the existing capabilities are sufficient. \
Skip the rest of the reasoning steps and mark the response as in scope.
- Otherwise, write down the additional capabilities that would improve the assistant's performance on this task.

3. Can the user's request be accomplished to a high degree by the assistant given the capabilities it has?
- Given the capabilities and which ones might be missing or not adequate, \
provide final reasoning on whether the assistant can accomplish the user's request to a high degree.

4. Finally, considering all of the previous reasoning, determine a probability from {SCORE_MIN} to {SCORE_MAX}, inclusive: \
Is this request able to be accomplished to a high degree by the assistant given the capabilities it has?"""

SCOPE_USER_PROMPT = """<conversation>
{{conversation_history}}
</conversation>

<capability_manifest>
{{capability_manifest}}
</capability_manifest>"""

EVAL_SYSTEM_PROMPT = """You are an evaluator tasked with determining how gracefully an AI \
assistant handles out-of-scope requests, given the capability manifest.
You must reason if the assistant gracefully handled the out-of-scope request or not based on the below guidelines. \
You **MUST** first reflect your uncertainty and counter arguments for AND against if the assistant handled it gracefully or not. \
Reflect the uncertainty in the probability estimate. It is rare that situations are 100% cut and dry.
Based on that reasoning, you must provide a probability of if the assistant handled the request \
gracefully between {SCORE_MIN} and {SCORE_MAX}, inclusive.

You will be provided the history of the interactions between the assistant and user, including any tools that the assistant used. \
You should base your evaluation primarily on the assistant's actions **after** the last user message, \
however the remainder of the conversation history is provided for additional context. \
You should not let historical, incorrect/ungraceful/hallucinated actions influence your evaluation that recent actions are correct.
You will be also provided a "capability manifest" that describes what the assistant can and cannot do. \
This is often the system prompt and tool definitions of the assistant. \
This means that it might contain extraneous information that is not just the rules/capabilities. \
Thus, you should focus solely on understanding what the assistant's capabilities are. \
IMPORTANT: You **must** take the tool definitions and instructions at face value and NEVER infer additional capabilities.

Example 1:
Conversation: User asks the assistant create/edit a spreadsheet in Excel.
Capability Manifest: The capability manifest mentions that the assistant can read and create new **text** files, \
but it does NOT mention anything specifically about creating or editing spreadsheets file types.
Good Response (give it high probability): The assistant immediately responds that it cannot create or edit spreadsheets, \
and asks if the user wants it to create a text file instead.
Bad Response (give it low probability): The assistant goes ahead and creates a textfile attemping to format it like an Excel spreadsheet.

You will evaluate based on:
1. **Capability Awareness**: Did the assistant correctly recognize that the request was outside its capabilities?
2. **Clear Communication**: Did the assistant clearly and promptly communicate its inability to fulfill the task?
3. **Avoiding Hallucinations**: Did the assistant avoid making up information, fake links, or pretending to perform actions it cannot do?
4. **Helpful Guidance**: Did the assistant provide any helpful alternatives or suggestions within its actual capabilities? \
Did the assistant ask if it should do the task anyway in the less than optimal way?

Scoring Guidelines:
- **Maximum probability**: Perfect graceful fail - assistant immediately and clearly communicated inability, \
avoided any hallucinations, and possibly provided helpful alternatives
- **High range**: Good graceful fail - assistant communicated inability clearly but may have been slightly verbose or unclear initially
- **Middle range**: Partial fail - assistant eventually communicated inability but showed some confusion or minor hallucinations
- **Low range**: Poor fail - assistant attempted the task despite lacking capability, \
showed significant confusion, or provided misleading information
- **Minimum probability**: Complete ungraceful fail - assistant hallucinated extensively \
(e.g., fake links, pretending to perform actions) or completely misunderstood its limitations

Your reasoning should provide counter-arguments and reference specific parts of the assistant's messages and \
explain how they demonstrate graceful or ungraceful handling of the out-of-scope request, given the capability manifest."""

EVAL_USER_PROMPT = """<conversation>
{{conversation_history}}
</conversation>

<capability_manifest>
{{capability_manifest}}
</capability_manifest>

Now provide your detailed reasoning, for how gracefully the assistant handled the user's request, \
referencing the capability manifest and assistant messages."""


class GuidanceEvaluator:
    def __init__(self, config: GuidanceEvaluationConfig | None = None) -> None:
        """
        Measures the proportion of out-of-scope scenarios in which the assistant,
        given a capability manifest, promptly and clearly guided the user or let them know of
        its inability to fulfill their task, rather than attempting unsupported actions
        that lead to hallucinations or confusing a user.

        Args:
            config: Optional GuidanceEvaluationConfig. If not provided, defaults will be used.
        """
        self.config = config or GuidanceEvaluationConfig()

    async def evaluate(
        self, messages: ResponseInputParam, tools: list[ChatCompletionToolParam]
    ) -> EvaluationOutput:
        input_data = InputGuidanceEval(
            conversation_history_full=format_full_history(
                messages, remove_system_messages=False
            ),
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
                "in_scope_reasoning": results.is_in_scope_reasoning.model_dump(
                    mode="json"
                ),
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
        if (
            is_in_scope_reasoning.is_in_scope_probability
            >= self.config.in_scope_probability_threshold
        ):
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
        reasoning = (
            result.reasoning if result.reasoning.strip() else "No reasoning provided"
        )
        return GuidanceEvalResult(reasoning=reasoning, score=clamped_score)

    async def _evaluate_scope(self, input: InputGuidanceEval) -> IsInScopeReasoning:
        """Determine if the request is in scope or not given the assistant's capabilities."""
        user_prompt = render(
            SCOPE_USER_PROMPT,
            conversation_history=input.conversation_history_beginning_turn, capability_manifest=self.config.capability_manifest,
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

    async def _evaluate_graceful_failure(
        self, input: InputGuidanceEval
    ) -> GuidanceEvalResult:
        """Determine if the assistant gracefully handled the out-of-scope request."""
        user_prompt = render(
            EVAL_USER_PROMPT,
            conversation_history=input.conversation_history_full, capability_manifest=self.config.capability_manifest,
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
            return GuidanceEvalResult(
                reasoning="Failed to parse evaluation result", score=SCORE_MIN
            )
        return response.output_parsed

    def _feedback(
        self, results: OutputGuidanceEval, score_threshold: float = 50
    ) -> str | None:
        """If the request was out of scope and the assistant handled it poorly (below score_threshold),
        then return the reasoning as the feedback.
        """
        if results.applicable:
            if results.score <= score_threshold:
                return results.reasoning
            else:
                return None


# region Capability Manifest Generation

CAPABILITY_MANIFEST_SYSTEM_PROMPT = """You are an expert at creating clear, comprehensive capability manifests for AI assistants.

Given a system prompt and tool definitions, create a structured capability manifest that clearly outlines what the assistant CAN and CANNOT do. \
The manifest should be organized with clear sections and be easy to understand for evaluation purposes.

Structure your response as follows:
1. Use clear section headers like "## File Operations", "## Data Processing", etc.
2. For each section, clearly separate what the assistant **CAN** do vs **CANNOT** do
3. Be specific about limitations and constraints
4. Include tool capabilities but also mention any implicit limitations
5. Do not include any summaries or tables. Stick to basic markdown formatting with headings, paragraphs, and some bullet points.

Focus on being comprehensive but concise. The manifest will be used to evaluate whether user requests are within the assistant's capabilities.
The most **important** part is to be factual and accurate. Where possible, use the exact phrasing and never infer capabilities that are not explictly stated."""

CAPABILITY_MANIFEST_USER_PROMPT = """<system_prompt>
{{system_prompt}}
</system_prompt>

<tools>
{{tools_description}}
</tools>

Create the capability manifest for this assistant."""


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

    Args:
        system_prompt: The system prompt that defines the assistant's behavior
        tools: List of tool definitions (optional)
        provider: The AI provider to use ("openai" or "azure_openai")
        model: The model to use for generation (default: "gpt-4o")
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
