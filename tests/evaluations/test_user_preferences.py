# Copyright (c) Microsoft. All rights reserved.

from openai.types.responses import EasyInputMessageParam
import pytest

from eval_recipes.evaluations.user_preferences import UserPreferencesEvaluator
from eval_recipes.schemas import BaseEvaluationConfig


@pytest.fixture
def messages() -> list:
    return [
        EasyInputMessageParam(
            role="system",
            content="""You are a helpful assistant.
User memories and preferences:
- The user prefers responses in paragraph form rather than bullet points
- The user likes when technical concepts are explained with real-world analogies
- The user's favorite video game genre is retro 2D platformers
- The user prefers concise responses without excessive detail""",
        ),
        EasyInputMessageParam(
            role="user",
            content="Can you help me understand Python decorators?",
        ),
        EasyInputMessageParam(
            role="assistant",
            content="""Python decorators are like gift wrapping for functions. \
Just as you can wrap a gift to make it look different or add a bow without changing what's inside, \
decorators wrap functions to add extra functionality without modifying the original function's code. \
At their core, decorators are functions that take another function as input and return a modified version. \
You use them by placing @decorator_name above your function definition. \
This pattern is commonly used for adding logging, checking permissions, or measuring how long a function takes to run.""",
        ),
    ]


@pytest.mark.skip(reason="Time")
async def test_user_preferences_evaluator(messages) -> None:
    """Test the UserPreferencesEvaluator."""
    config = BaseEvaluationConfig(
        provider="openai",
        model="gpt-5",
    )
    evaluator = UserPreferencesEvaluator(config=config)
    result = await evaluator.evaluate(messages=messages, tools=[])
    print(result)
