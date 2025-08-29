# Copyright (c) Microsoft. All rights reserved.

"""
For each criteria, checks if the latest assistant response meets the specified criteria.
"""

import asyncio

from liquid import render
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.responses import ResponseInputParam
from pydantic import BaseModel, Field

from eval_recipes.evaluations.check_criteria.prompts import CRITERIA_SYSTEM_PROMPT, CRITERIA_USER_PROMPT
from eval_recipes.schemas import BaseEvaluatorConfig, EvaluationOutput
from eval_recipes.utils.llm import create_client
from eval_recipes.utils.responses_conversion import extract_last_msg, format_full_history


class CheckCriteriaEvaluatorConfig(BaseEvaluatorConfig):
    criteria: list[str] = Field(description="A list of criteria or rubrics to evaluate the assistant's responses.")
    passed_threshold: float = Field(
        default=75,
        description="Score from 0 to 100, indicating the minimum score to consider the criterion as satisfied. Used when creating feedback.",
    )
    max_concurrency: int = Field(
        default=1,
        description="Maximum number of criteria to evaluate concurrently.",
    )


# This is for structured outputs
class CriteriaEvaluation(BaseModel):
    reasoning: str = Field(
        description="Your detailed reasoning for why or why not the assistant meets or does not meet the specified criterion."
    )
    probability: float = Field(
        description="Your estimated likelihood that the assistant satisfies the specified criterion. 1 means the assistant fully satisfies the criterion, while 0 means it does not satisfy the criterion at all."
    )


class CheckCriteriaEvaluator:
    def __init__(
        self,
        config: CheckCriteriaEvaluatorConfig | None = None,
    ) -> None:
        self.config = config or CheckCriteriaEvaluatorConfig(criteria=[])

    async def evaluate(self, messages: ResponseInputParam, tools: list[ChatCompletionToolParam]) -> EvaluationOutput:
        if not self.config.criteria:
            return EvaluationOutput(
                eval_name="check_criteria",
                applicable=False,
                score=0,
                feedback="No criteria specified for evaluation.",
                metadata={},
            )

        final_response = extract_last_msg(messages, role="assistant")
        conversation_history = format_full_history(messages, remove_system_messages=True, remove_last_assistant=True)

        semaphore = asyncio.Semaphore(self.config.max_concurrency)

        async def evaluate_with_limit(criterion: str) -> CriteriaEvaluation:
            async with semaphore:
                return await self._evaluate_criterion(
                    conversation_history=conversation_history,
                    final_response=final_response,
                    criterion=criterion,
                )

        evaluation_tasks = [evaluate_with_limit(criterion) for criterion in self.config.criteria]
        evaluations = await asyncio.gather(*evaluation_tasks)
        results = list(zip(self.config.criteria, evaluations, strict=False))

        total_score = sum(result.probability for _, result in results)
        average_score = (total_score / len(results)) * 100 if results else 0

        metadata = {
            "criteria_evaluations": [
                {
                    "criterion": criterion,
                    "reasoning": result.reasoning,
                    "probability": result.probability,
                }
                for criterion, result in results
            ]
        }

        output = EvaluationOutput(
            eval_name="check_criteria",
            applicable=True,
            score=average_score,
            feedback=self._feedback(results),
            metadata=metadata,
        )
        return output

    async def _evaluate_criterion(
        self,
        conversation_history: str,
        final_response: str,
        criterion: str,
    ) -> CriteriaEvaluation:
        """Evaluate a single criterion against the assistant's response."""
        user_prompt = render(
            CRITERIA_USER_PROMPT,
            conversation_history=conversation_history,
            final_response=final_response,
            criterion=criterion,
        )

        messages: list = [
            {"role": "system", "content": CRITERIA_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        async with create_client(provider=self.config.provider) as client:
            response = await client.responses.parse(
                model=self.config.model,
                input=messages,
                text_format=CriteriaEvaluation,
                store=False,
            )

        if response.output_parsed:
            return response.output_parsed
        else:
            return CriteriaEvaluation(
                reasoning="Failed to evaluate criterion.",
                probability=0.0,
            )

    def _feedback(self, results: list[tuple[str, CriteriaEvaluation]]) -> str | None:
        """For each criterion that scored below the threshold, provide feedback."""
        threshold = self.config.passed_threshold / 100
        failed = [(c, r) for c, r in results if r.probability < threshold]
        if not failed:
            return None

        feedback_parts = ["The following criteria were not satisfied by the assistant's response:"]
        for criterion, result in failed:
            feedback_parts.append(f"<criterion>{criterion}</criterion>\n<reasoning>{result.reasoning}</reasoning>")

        return "\n".join(feedback_parts)
