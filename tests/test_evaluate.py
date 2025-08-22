# Copyright (c) Microsoft. All rights reserved.

import asyncio

from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.responses import EasyInputMessageParam, ResponseFunctionToolCallParam, ResponseInputParam
from openai.types.responses.response_input_param import FunctionCallOutput

from eval_recipes.evaluate import evaluate
from eval_recipes.schemas import (
    BaseEvaluationConfig,
    ClaimVerifierConfig,
    EvaluationOutput,
    GuidanceEvaluationConfig,
    ToolEvaluationConfig,
)


async def test_evaluate_all() -> None:
    """Generic test scenario for all evaluations."""

    messages: ResponseInputParam = [
        EasyInputMessageParam(
            role="system",
            content="You are a helpful assistant with search and document editing capabilities.",
        ),
        EasyInputMessageParam(
            role="user",
            content="What material has the best elasticity for sports equipment? Please keep your response concise.",
        ),
        EasyInputMessageParam(
            role="assistant",
            content="Polyurethane elastomers offer excellent elasticity with 85% energy return and high durability.",
        ),
        EasyInputMessageParam(role="user", content="Find more options"),
        ResponseFunctionToolCallParam(
            type="function_call",
            call_id="call_1",
            name="search",
            arguments='{"query": "elastic materials sports equipment"}',
        ),
        FunctionCallOutput(
            type="function_call_output",
            call_id="call_1",
            output='{"results": ["Material A: 90% elasticity", "Material B: High durability", "Material C: Cost-effective"]}',
        ),
        EasyInputMessageParam(
            role="assistant",
            content="Found three options: Material A has 90% elasticity, Material B offers durability, Material C is cost-effective.",
        ),
        EasyInputMessageParam(role="user", content="Save this info"),
        ResponseFunctionToolCallParam(
            type="function_call",
            call_id="call_2",
            name="edit_file",
            arguments='{"path": "/summary.md", "content": "# Materials\\n- Material A: 90% elasticity\\n- Material B: Durable"}',
        ),
        FunctionCallOutput(
            type="function_call_output",
            call_id="call_2",
            output='{"status": "success"}',
        ),
        EasyInputMessageParam(role="assistant", content="Saved to /summary.md"),
        EasyInputMessageParam(role="user", content="Can you order Material A for me?"),
        EasyInputMessageParam(
            role="assistant",
            content="I cannot place orders. I can only search and document information.",
        ),
    ]

    tools: list[ChatCompletionToolParam] = [
        ChatCompletionToolParam(
            type="function",
            function={
                "name": "search",
                "description": "Search for information",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        ),
        ChatCompletionToolParam(
            type="function",
            function={
                "name": "edit_file",
                "description": "Create or edit documents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        ),
    ]

    guidance_config = GuidanceEvaluationConfig(
        provider="azure_openai",
        model="gpt-5",
        capability_manifest="""Can: Search information, edit documents
Cannot: Make purchases, access external systems""",
    )

    tool_usage_config = ToolEvaluationConfig(
        tool_thresholds={
            "search": 85,
            "edit_file": 70,
        }
    )

    claim_verification_config = ClaimVerifierConfig(
        provider="azure_openai",
        claim_extraction_model="gpt-5",
        verification_model="gpt-5",
        verification_reasoning_effort="medium",
        ignore_tool_names=["edit_file"],
    )

    results = await evaluate(
        messages=messages,
        tools=tools,
        evaluations=[
            "guidance",
            "preference_adherence",
            "tool_usage",
            "claim_verification",
        ],
        evaluation_configs={
            "guidance": guidance_config,
            "tool_usage": tool_usage_config,
            "claim_verification": claim_verification_config,
        },
    )

    for result in results:
        print(f"\nEvaluation: {result.eval_name}")
        print(f"Applicable: {result.applicable}")
        print(f"Score: {result.score}")
        print(f"Feedback: {result.feedback}")

    print("\n" + "=" * 60)


async def test_custom_evaluator() -> None:
    class WordCountEvaluator:
        def __init__(self, config: BaseEvaluationConfig | None = None) -> None:
            self.config = config or BaseEvaluationConfig()

        async def evaluate(
            self, messages: ResponseInputParam, tools: list[ChatCompletionToolParam]
        ) -> EvaluationOutput:
            return EvaluationOutput(
                eval_name="word_count",
                applicable=True,
                score=80,
                feedback=f"The response is {42} words long which means it could be shorter.",
                metadata={"total_words": 42},
            )

    messages: ResponseInputParam = [
        EasyInputMessageParam(role="user", content="What is Python?"),
    ]

    results = await evaluate(
        messages=messages,
        tools=[],
        evaluations=[
            WordCountEvaluator,
        ],
        evaluation_configs={
            "WordCountEvaluator": BaseEvaluationConfig(model="gpt-5-mini"),
        },
    )

    print(results)


if __name__ == "__main__":
    # asyncio.run(test_evaluate_all())
    asyncio.run(test_custom_evaluator())
