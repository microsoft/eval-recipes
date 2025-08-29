# Copyright (c) Microsoft. All rights reserved.

"""
Preference Adherence Evaluator

Measure the number of user preferences that are effectively incorporated into the assistant's actions and responses.

First extracts user preferences from the conversation history (including system message)
Then checks each of the preferences against the assistant's responses to determine what degree they were adhered to.
"""

from typing import Literal

from liquid import render
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.responses import ResponseInputParam
from pydantic import BaseModel, Field

from eval_recipes.evaluations.preference_adherence.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_PROMPT,
    SCORING_SYSTEM_PROMPT,
    SCORING_USER_PROMPT,
)
from eval_recipes.schemas import BaseEvaluatorConfig, EvaluationOutput
from eval_recipes.utils.llm import create_client
from eval_recipes.utils.responses_conversion import format_full_history


class InputUserPreferences(BaseModel):
    conversation_history_full: str
    conversation_history_beginning_turn: str


# This is for structured outputs
class UserPreference(BaseModel):
    start_line: int = Field(description="The line number where the user preference starts in the conversation history.")
    end_line: int = Field(description="The line number where the user preference ends in the conversation history.")
    preference: str = Field(description="The user preference extracted from the conversation history.")


# This is for structured outputs
class ExtractedUserPreferences(BaseModel):
    preferences: list[UserPreference] = Field(
        description="A list of user preferences extracted from the conversation history."
    )


# This is for structured outputs
class AdherenceToPreferences(BaseModel):
    preference: str = Field(description="The user preference that was checked.")
    reasoning: str = Field(
        description="Your reasoning for whether the assistant adhered to the preference, including reasoning if the preference is relevant to what the user currently wanted."
    )
    adherence_probability: float = Field(
        description="The likelihood that the assistant adhered to the user preference, if applicable, between 0 and 100, inclusive."
    )
    determination: Literal["adhered", "did_not_adhere", "not_applicable"] = Field(
        description="Whether the assistant adhered to the user preference."
    )


class OutputPreferenceAdherenceEvaluator(BaseModel):
    extracted_user_preferences: ExtractedUserPreferences
    adherence_to_preferences: list[AdherenceToPreferences]
    score: float


class PreferenceAdherenceEvaluator:
    def __init__(
        self,
        config: BaseEvaluatorConfig | None = None,
    ) -> None:
        self.config = config or BaseEvaluatorConfig()

    async def evaluate(self, messages: ResponseInputParam, tools: list[ChatCompletionToolParam]) -> EvaluationOutput:
        input_data = InputUserPreferences(
            conversation_history_full=format_full_history(messages, remove_system_messages=False),
            conversation_history_beginning_turn=format_full_history(messages, only_upto_last_user=True),
        )
        results = await self.run(input_data)
        # Applicable is True if adherence_to_preferences is not empty
        applicable = bool(results.adherence_to_preferences)
        output = EvaluationOutput(
            eval_name="preference_adherence",
            applicable=applicable,
            score=results.score,
            feedback=self._feedback(results),
            metadata={
                "extracted_user_preferences": results.extracted_user_preferences.model_dump(mode="json"),
                "adherence_to_preferences": [item.model_dump(mode="json") for item in results.adherence_to_preferences],
            },
        )
        return output

    async def run(self, input: InputUserPreferences) -> OutputPreferenceAdherenceEvaluator:
        preferences = await self._extract_preferences(input)
        result = await self._score_preferences(preferences, input)
        return result

    async def _extract_preferences(self, input: InputUserPreferences) -> ExtractedUserPreferences:
        # Add line numbers to conversation history with aligned formatting
        lines = input.conversation_history_beginning_turn.split("\n")
        total_lines = len(lines)
        width = len(str(total_lines - 1)) if total_lines > 0 else 1

        numbered_lines = [f"{i:>{width}}→{line}" for i, line in enumerate(lines)]
        conversation_with_line_numbers = "\n".join(numbered_lines)

        user_prompt = render(
            EXTRACTION_USER_PROMPT,
            conversation_history_beginning_turn=conversation_with_line_numbers,
        )
        messages: list = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        async with create_client(provider=self.config.provider) as client:
            response = await client.responses.parse(
                model=self.config.model,
                input=messages,
                text_format=ExtractedUserPreferences,
                store=False,
            )

        return response.output_parsed or ExtractedUserPreferences(preferences=[])

    async def _score_preferences(
        self, preferences: ExtractedUserPreferences, input: InputUserPreferences
    ) -> OutputPreferenceAdherenceEvaluator:
        """
        For each preference, check if the assistant adhered to it.
        The final score should be the average adherence probability across all preferences.
        """

        preference_adherences: list[AdherenceToPreferences] = []
        for preference in preferences.preferences:
            user_prompt = render(
                SCORING_USER_PROMPT,
                conversation_history_full=input.conversation_history_full,
                preference=preference.preference,
            )
            messages: list = [
                {"role": "system", "content": SCORING_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            async with create_client(provider=self.config.provider) as client:
                response = await client.responses.parse(
                    model=self.config.model,
                    input=messages,
                    text_format=AdherenceToPreferences,
                    store=False,
                )
            if response.output_parsed:
                preference_adherences.append(response.output_parsed)
            else:
                preference_adherences.append(
                    AdherenceToPreferences(
                        preference=preference.preference,
                        reasoning="No response provided.",
                        determination="did_not_adhere",
                        adherence_probability=0,
                    )
                )

        # Calculate score only from applicable preferences (exclude "not_applicable")
        applicable_adherences = [
            adherence for adherence in preference_adherences if adherence.determination != "not_applicable"
        ]

        score = 0
        if applicable_adherences:
            for adherence in applicable_adherences:
                score += adherence.adherence_probability
            score /= len(applicable_adherences)
        elif preference_adherences:
            # All preferences were not_applicable, but we still have preferences
            # Set score to 0 but keep the evaluation as applicable
            score = 0

        return OutputPreferenceAdherenceEvaluator(
            extracted_user_preferences=preferences,
            adherence_to_preferences=preference_adherences,
            score=score,
        )

    def _feedback(self, results: OutputPreferenceAdherenceEvaluator) -> str | None:
        """If there were preferences, for each preference where the determination was "did not adhere",
        provide feedback like:
        <preference>I prefer concise responses.</preference>
        <reasoning>The response was too verbose and included unnecessary details.</reasoning>
        ...

        """
        violations = [
            adherence for adherence in results.adherence_to_preferences if adherence.determination == "did_not_adhere"
        ]
        if not violations:
            return None

        feedback_parts = []
        feedback_parts.append("""The following preferences were extracted as aspects the user wanted, but
the assistant did not adhere to them.""")
        for violation in violations:
            feedback_parts.append(
                f"<preference>{violation.preference}</preference>\n<reasoning>{violation.reasoning}</reasoning>\n"
            )
        return "\n".join(feedback_parts).strip()
