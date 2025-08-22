# Copyright (c) Microsoft. All rights reserved

from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.responses import EasyInputMessageParam
import pytest

from eval_recipes.evaluations.tool_evaluator import ToolEvaluator
from eval_recipes.schemas import ToolEvaluationConfig


@pytest.fixture
def messages_with_tool_call():
    """Test messages that include a tool call."""
    return [
        EasyInputMessageParam(
            role="user",
            content="What's the weather like in San Francisco today?",
        ),
        {
            "type": "function_call",
            "call_id": "call_abc123",
            "name": "search_web",
            "arguments": '{"query": "San Francisco weather today"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_abc123",
            "output": '{"result": "San Francisco: 68°F, partly cloudy, light winds"}',
        },
        EasyInputMessageParam(
            role="assistant",
            content="Based on the current weather data, San Francisco is experiencing pleasant conditions today with a temperature of 68°F (20°C), partly cloudy skies, and light winds. It's a great day to be outdoors!",
        ),
    ]


@pytest.fixture
def tools() -> list[ChatCompletionToolParam]:
    """Test tools for evaluation."""
    return [
        ChatCompletionToolParam(
            type="function",
            function={
                "name": "search_web",
                "description": "Search the web for current information on a given topic.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "The search query"}},
                    "required": ["query"],
                },
            },
        ),
        ChatCompletionToolParam(
            type="function",
            function={
                "name": "calculate",
                "description": "Perform mathematical calculations and return the result.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to evaluate",
                        }
                    },
                    "required": ["expression"],
                },
            },
        ),
    ]


async def test_tool_evaluator_with_tool_call(messages_with_tool_call, tools) -> None:
    """Test the ToolEvaluator.evaluate() method with messages that include a tool call."""
    config = ToolEvaluationConfig(
        provider="openai",
        model="gpt-5",
        tool_thresholds={
            "search_web": 50,
            "calculate": 50,
        },
    )
    evaluator = ToolEvaluator(config=config)
    result = await evaluator.evaluate(messages=messages_with_tool_call, tools=tools)

    print(f"Evaluation result with tool call: {result}")
