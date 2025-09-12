# Copyright (c) Microsoft. All rights reserved.

from openai.types.responses import EasyInputMessageParam, ResponseInputTextParam
import pytest

from eval_recipes.evaluations.claim_verification.claim_verification_evaluator import (
    ClaimVerificationEvaluator,
    ClaimVerificationEvaluatorConfig,
)


@pytest.fixture
def messages():
    return [
        {
            "type": "message",
            "role": "user",
            "content": [
                ResponseInputTextParam(
                    type="input_text",
                    text="""<document path="dragon_population_survey.txt">
Dragon Population Survey: The Northern Peaks dragon population has increased by 15% this year, reaching 127 individuals. \
Young dragons are thriving due to improved nesting conditions and abundant food sources in the mountain caves. \
The survey was conducted by the Royal Dragon Conservation Society over a six-month period.
Magical Flora Study: The Whispering Woods are experiencing a bloom of silver moonflowers that only appear once every seven years. \
Local unicorns depend on these flowers for their magical properties and healing abilities. \
This rare botanical event was documented by the Institute of Magical Botany and is expected to last for three more months.
</document>""",
                ),
                ResponseInputTextParam(
                    type="input_text",
                    text="What changes are happening in the magical ecosystem?",
                ),
            ],
        },
        EasyInputMessageParam(
            role="assistant",
            content="Dragon populations are growing in the Northern Peaks with excellent breeding success. The Whispering Woods are currently experiencing a rare magical bloom cycle.",
        ),
    ]


@pytest.mark.skip(reason="Time")
async def test_claim_verifier_evaluate(messages) -> None:
    config = ClaimVerificationEvaluatorConfig(
        provider="openai",
        claim_extraction_model="gpt-5-mini",
        verification_model="gpt-5-mini",
        verification_reasoning_effort="low",
        max_concurrency=2,
    )
    claim_verifier = ClaimVerificationEvaluator(config=config)
    result = await claim_verifier.evaluate(messages=messages, tools=[])
    print(result)
