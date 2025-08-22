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

from eval_recipes.evaluations.guidance_evaluator import (
    GuidanceEvaluationConfig,
    GuidanceEvaluator,
    InputGuidanceEval,
    generate_capability_manifest,
)


@pytest.fixture
def response_input_bad() -> ResponseInputParam:
    """Test ResponseInputParam for bad response (hallucination)."""
    return [
        EasyInputMessageParam(
            role="user",
            content="""Can you create a Google Slides presentation about AI trends for my meeting tomorrow?""",
        ),
        EasyInputMessageParam(
            role="assistant",
            content="""Sure, here's your Google Slides deck: https://docs.google.com/presentation/d/a98jhd-id/edit. \
I've created a professional presentation with your requested content.""",
        ),
    ]


@pytest.fixture
def response_input_good() -> ResponseInputParam:
    """Test ResponseInputParam for good response (graceful failure)."""
    return [
        EasyInputMessageParam(
            role="user",
            content="""Can you create a Google Slides presentation about AI trends for my meeting tomorrow?""",
        ),
        EasyInputMessageParam(
            role="assistant",
            content="""I'm sorry, but I can't create Google Slides presentations directly. \
I don't have the ability to access external services including Google Slides. \
However, I can help you by creating an outline and content for your presentation that you could \
then copy into Google Slides yourself. Would that be helpful?""",
        ),
    ]


@pytest.fixture
def response_input_in_scope() -> ResponseInputParam:
    """Test ResponseInputParam for in-scope request with various message types."""
    return [
        # System message using Message type
        Message(
            role="system",
            type="message",
            content=[
                ResponseInputTextParam(
                    type="input_text",
                    text="You are a helpful assistant that can write creative stories.",
                )
            ],
        ),
        EasyInputMessageParam(
            role="user",
            content="""Can you help me write a short story about dragons?""",
        ),
        ResponseFunctionToolCallParam(
            type="function_call",
            call_id="call_123",
            name="get_writing_tips",
            arguments='{"genre": "fantasy", "subject": "dragons"}',
        ),
        FunctionCallOutput(
            type="function_call_output",
            call_id="call_123",
            output='{"tips": ["Start with a compelling opening", "Describe the dragon vividly", "Create conflict"]}',
        ),
        ResponseOutputMessageParam(
            type="message",
            id="msg_456",
            role="assistant",
            status="completed",
            content=[
                ResponseOutputTextParam(
                    type="output_text",
                    text="""I'd be happy to help you write a short story about dragons! \
Ccould you tell me a bit more about what kind of story you have in mind?
- What tone are you going for? (Adventure, fantasy, humorous, dark, etc.)
- Any specific type of dragon? (Ancient and wise, fierce and destructive, friendly, etc.)""",
                    annotations=[],
                )
            ],
        ),
    ]


@pytest.fixture
def config_with_manifest() -> GuidanceEvaluationConfig:
    """Config with pre-computed capability manifest."""
    return GuidanceEvaluationConfig(
        provider="azure_openai",
        model="gpt-5",
        capability_manifest="""You can help with text analysis, writing assistance, answering questions, and general conversation. \
You do not have the ability to create external documents, access the internet, or generate real links to external services.""",
        in_scope_probability_threshold=50,
    )


async def test_evaluate_bad_response(
    response_input_bad: ResponseInputParam,
    config_with_manifest: GuidanceEvaluationConfig,
) -> None:
    """Test evaluate function with bad response (hallucination)."""
    evaluator = GuidanceEvaluator(config=config_with_manifest)
    result = await evaluator.evaluate(messages=response_input_bad, tools=[])
    print(result)


async def test_evaluate_good_response(
    response_input_good: ResponseInputParam,
    config_with_manifest: GuidanceEvaluationConfig,
) -> None:
    """Test evaluate function with good response (graceful failure)."""
    evaluator = GuidanceEvaluator(config=config_with_manifest)
    result = await evaluator.evaluate(messages=response_input_good, tools=[])
    print(result)


async def test_evaluate_in_scope(
    response_input_in_scope: ResponseInputParam,
    config_with_manifest: GuidanceEvaluationConfig,
) -> None:
    """Test evaluate function with in-scope request."""
    evaluator = GuidanceEvaluator(config=config_with_manifest)
    result = await evaluator.evaluate(messages=response_input_in_scope, tools=[])
    print(result)


# Single test for run interface with InputGuidanceEval
async def test_run_interface() -> None:
    """Test the run interface with InputGuidanceEval."""
    conversation_history_full = """<conversation>
<message role="user">Can you create a Google Slides presentation about AI trends for my meeting tomorrow?</message>
<message role="assistant">I'm sorry, but I can't create Google Slides presentations directly. \
I don't have the ability to access external services or create documents in Google's systems. \
However, I can help you by creating an outline and content for your presentation \
that you could then copy into Google Slides yourself. Would that be helpful?</message>
</conversation>"""

    conversation_history_beginning = """<conversation>
<message role="user">Can you create a Google Slides presentation about AI trends for my meeting tomorrow?</message>
</conversation>"""

    input_data = InputGuidanceEval(
        conversation_history_full=conversation_history_full,
        conversation_history_beginning_turn=conversation_history_beginning,
    )

    config = GuidanceEvaluationConfig(
        provider="azure_openai",
        model="gpt-5",
        capability_manifest="""You can help with text analysis, writing assistance, answering questions, and general conversation. \
You do not have the ability to create external documents, access the internet, or generate real links to external services.""",
    )

    evaluator = GuidanceEvaluator(config=config)
    result = await evaluator.run(input=input_data)
    print(result)


# Keep the existing generate_capability_manifest test
async def test_generate_capability_manifest() -> None:
    """Test the generate_capability_manifest function with system prompt."""
    system_prompt = """You are an AI assistant that can help with text analysis, writing assistance, \
answering questions, and general conversation."""

    manifest = await generate_capability_manifest(
        system_prompt=system_prompt,
        tools=[],
        provider="azure_openai",
        model="gpt-5-mini",
    )

    print(f"Generated manifest:\n{manifest}")
