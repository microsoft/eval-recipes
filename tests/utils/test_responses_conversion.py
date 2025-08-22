# Copyright (c) Microsoft. All rights reserved.

from openai.types.responses import (
    EasyInputMessageParam,
    ResponseFunctionToolCallParam,
    ResponseInputParam,
    ResponseInputTextParam,
    ResponseOutputMessageParam,
    ResponseOutputTextParam,
)
from openai.types.responses.response_input_param import FunctionCallOutput, Message
import pytest

from eval_recipes.utils.responses_conversion import format_full_history


@pytest.fixture
def sample_messages() -> ResponseInputParam:
    """Create a sample conversation with various message types."""
    return [
        Message(
            role="system",
            type="message",
            content=[
                ResponseInputTextParam(
                    type="input_text", text="You are a helpful AI assistant."
                )
            ],
        ),
        EasyInputMessageParam(role="user", content="What's the weather like?"),
        ResponseFunctionToolCallParam(
            type="function_call",
            call_id="call_123",
            name="get_weather",
            arguments='{"location": "Seattle"}',
        ),
        FunctionCallOutput(
            type="function_call_output",
            call_id="call_123",
            output='{"temp": 65, "conditions": "sunny"}',
        ),
        ResponseOutputMessageParam(
            type="message",
            id="msg_456",
            role="assistant",
            status="completed",
            content=[
                ResponseOutputTextParam(
                    type="output_text",
                    text="The weather in Seattle is sunny with 65°F.",
                    annotations=[],
                )
            ],
        ),
        EasyInputMessageParam(role="user", content="Thanks! What about tomorrow?"),
        EasyInputMessageParam(
            role="assistant",
            content="I would need to make another API call for tomorrow's forecast.",
        ),
    ]


@pytest.mark.parametrize(
    "remove_system_messages,only_upto_last_user,test_name",
    [
        (False, False, "Default - all messages"),
        (True, False, "Remove system messages"),
        (False, True, "Only up to last user"),
        (True, True, "Both parameters enabled"),
    ],
)
def test_format_full_history_parameters(
    sample_messages: ResponseInputParam,
    remove_system_messages: bool,
    only_upto_last_user: bool,
    test_name: str,
) -> None:
    """Test format_full_history with different parameter combinations using parametrize."""
    print(f"\n{'=' * 60}")
    print(f"Test: {test_name}")
    print(
        f"Parameters: remove_system_messages={remove_system_messages}, only_upto_last_user={only_upto_last_user}"
    )
    print(f"{'=' * 60}")

    result = format_full_history(
        sample_messages,
        remove_system_messages=remove_system_messages,
        only_upto_last_user=only_upto_last_user,
    )

    print(result)
    print(f"{'=' * 60}\n")
