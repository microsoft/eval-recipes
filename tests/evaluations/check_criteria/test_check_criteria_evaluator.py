# Copyright (c) Microsoft. All rights reserved.

from openai.types.responses import EasyInputMessageParam
import pytest

from eval_recipes.evaluations.check_criteria.check_criteria_evaluator import (
    CheckCriteriaEvaluator,
    CheckCriteriaEvaluatorConfig,
)


@pytest.fixture
def messages() -> list:
    return [
        EasyInputMessageParam(
            role="system",
            content="You are a helpful assistant.",
        ),
        EasyInputMessageParam(
            role="user",
            content="write me a short story about a bear and a magic lamp",
        ),
        EasyInputMessageParam(
            role="assistant",
            content="""Deep in the forest, a curious bear stumbled upon a magic lamp half-buried in \
moss—its golden surface gleaming as if it had been waiting just for him. \
With a hesitant paw, the bear rubbed it, and in a burst of light a genie appeared, stretching high above the treetops. \
“Three wishes are yours,” the genie declared, his voice rolling like thunder. \
The bear thought hard—honey, fish, and sleep were nice, but he longed for something more. “I wish,” he rumbled, “for the forest to always be safe.” \
With a nod, the genie granted it, and from that day forward, storms bent around the woods, \
hunters lost their way, and the bear—content and proud—returned to his den, the lamp tucked safely under his paw.""",
        ),
    ]


async def test_check_criteria_evaluator(messages) -> None:
    config = CheckCriteriaEvaluatorConfig(
        provider="openai",
        model="gpt-5-mini",
        criteria=[
            "The response contains no em dashes",
            "The response is at most one paragraph long.",
        ],
        max_concurrency=2,
    )
    evaluator = CheckCriteriaEvaluator(config=config)
    result = await evaluator.evaluate(messages=messages, tools=[])
    print(result)
