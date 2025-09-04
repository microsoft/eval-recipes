# Copyright (c) Microsoft. All rights reserved.

"""
This module implements the claim extraction methodology from the Claimify paper (https://arxiv.org/abs/2502.10855).
This is not the official implementation of the methodology. Please cite the original paper if you use this in your work.
"""

import asyncio
from typing import Literal

from liquid import render
from loguru import logger
from openai.types.responses import EasyInputMessageParam
from pydantic import BaseModel
from rich import print as rich_print

from eval_recipes.evaluations.claim_verification.prompts import (
    DECOMPOSITION_SYSTEM_PROMPT,
    DECOMPOSITION_USER_PROMPT,
    DISAMBIGUATION_SYSTEM_PROMPT,
    DISAMBIGUATION_USER_PROMPT,
    SELECTION_SYSTEM_PROMPT,
    SELECTION_USER_PROMPT,
)
from eval_recipes.utils.llm import create_client


class InputClaimExtraction(BaseModel):
    sentence: str  # The sentence to extract claims from
    excerpt: str  # The surrounding context for the sentence, used to provide additional information
    user_question: str  # The user question to provide context for the extraction


class OutputSelectionStep(BaseModel):
    has_verifiable_claims: bool  # Does the sentence have any verifiable claims?
    sentence_for_next_step: str | None = None  # The sentence to be used in the next step, if applicable


class OutputDisambiguationStep(BaseModel):
    disambiguated_sentences: list[str] = []


class OutputDecompositionStep(BaseModel):
    claims: list[str] = []


# The following classes are used for structured outputs for the LLM and
# should be kept separate from the actual return types in order to reserve the ability to change them for prompt engineering purposes.
# This class for Structured Outputs only.
class SelectionResult(BaseModel):
    sentence: str
    thought_process: str
    final_submission: str  # Either "Contains a specific and verifiable proposition" or "Does NOT contain a specific and verifiable proposition"
    sentence_with_verifiable_info: str  # The sentence with only verifiable information, 'remains unchanged' , or 'None'


# This class for Structured Outputs only.
class DisambiguationResult(BaseModel):
    incomplete_names_acronyms_abbreviations_reasoning: str
    linguistic_ambiguity_reasoning: str
    changes_needed_reasoning: str
    changes_needed: bool
    decontextualized_sentences: list[str]  # The final decontextualized sentence or collection of sentences


# This class for Structured Outputs only.
class DecompositionResult(BaseModel):
    sentence: str
    referential_terms_to_clarify: str
    max_clarified_sentence: str
    range_of_possible_propositions: str
    specific_verifiable_decontextualized_propositions: list[str]
    specific_verifiable_decontextualized_propositions_with_clarifications: list[str]


class ClaimExtraction:
    def __init__(
        self,
        input_data: InputClaimExtraction,
        provider: Literal["openai", "azure_openai"] = "openai",
        claim_extraction_model: Literal["gpt-5", "gpt-5-mini", "gpt-5-nano", "o3", "o4-mini"] = "gpt-5-mini",
    ) -> None:
        self.input_data = input_data
        self.provider: Literal["openai", "azure_openai"] = provider
        self.claim_extraction_model: Literal["gpt-5", "gpt-5-mini", "gpt-5-nano", "o3", "o4-mini"] = (
            claim_extraction_model
        )

    async def run(self) -> list[str]:
        # Process the sentence (assuming it's already been split by caller)
        sentence_text = self.input_data.sentence
        if not sentence_text:
            return []

        results = await self._process_sentence(self.input_data.sentence, sentence_text)

        logger.info(f"Claim extraction complete. Total claims extracted: {len(results) if results else 0}")
        return results if results else []

    async def _process_sentence(self, sentence: str, context: str) -> list[str] | None:
        """Process a single sentence."""
        selection_result = await self._selection()
        logger.info(f"Selection result: has_verifiable_claims={selection_result.has_verifiable_claims}")

        if not selection_result.has_verifiable_claims:
            return None

        disambiguation = await self._disambiguation(selection_result, context)
        logger.info(f"Disambiguation result: {len(disambiguation.disambiguated_sentences)} disambiguated sentences")

        if not disambiguation.disambiguated_sentences:
            return None

        decomposition = await self._decomposition(disambiguation, context)
        logger.info(f"Decomposition result: {len(decomposition.claims)} claims extracted")

        sentence_results: list[str] = []
        for claim in decomposition.claims:
            sentence_results.append(claim)
        return sentence_results

    async def _selection(self) -> OutputSelectionStep:
        user_prompt = render(
            SELECTION_USER_PROMPT,
            question=self.input_data.user_question,
            excerpt=self.input_data.excerpt,
            sentence=self.input_data.sentence,
        )
        messages: list = [
            EasyInputMessageParam(role="system", content=SELECTION_SYSTEM_PROMPT),
            EasyInputMessageParam(role="user", content=user_prompt),
        ]
        async with create_client(provider=self.provider) as client:
            response = await client.responses.parse(
                model=self.claim_extraction_model,
                input=messages,
                text_format=SelectionResult,
                store=False,
            )
        # has_verifiable_claims is true if the final submission is "Contains a specific and verifiable proposition"
        # and False otherwise.
        # sentence_for_next_step should be set if sentence_with_verifiable_info is not "None" or "remains unchanged"
        if response.output_parsed is None:
            has_verifiable_claims = False
            sentence_for_next_step = None
        else:
            has_verifiable_claims = (
                response.output_parsed.final_submission == "Contains a specific and verifiable proposition"
            )
            sentence_for_next_step = None
            if response.output_parsed.sentence_with_verifiable_info not in (
                "None",
                "remains unchanged",
            ):
                sentence_for_next_step = response.output_parsed.sentence_with_verifiable_info
            else:
                sentence_for_next_step = self.input_data.sentence

        result = OutputSelectionStep(
            has_verifiable_claims=has_verifiable_claims,
            sentence_for_next_step=sentence_for_next_step,
        )
        return result

    async def _disambiguation(self, selection_result: OutputSelectionStep, context: str) -> OutputDisambiguationStep:
        user_prompt = render(
            DISAMBIGUATION_USER_PROMPT,
            question=self.input_data.user_question,
            excerpt=context,
            sentence=selection_result.sentence_for_next_step,
        )
        messages: list = [
            EasyInputMessageParam(role="system", content=DISAMBIGUATION_SYSTEM_PROMPT),
            EasyInputMessageParam(role="user", content=user_prompt),
        ]
        async with create_client(provider=self.provider) as client:
            response = await client.responses.parse(
                model=self.claim_extraction_model,
                input=messages,
                text_format=DisambiguationResult,
                store=False,
            )
            if response.output_parsed is None:
                return OutputDisambiguationStep(disambiguated_sentences=[])
            return OutputDisambiguationStep(disambiguated_sentences=response.output_parsed.decontextualized_sentences)

    async def _decomposition(
        self, disambiguation_result: OutputDisambiguationStep, context: str
    ) -> OutputDecompositionStep:
        claims = []
        for disambiguated_sentence in disambiguation_result.disambiguated_sentences:
            user_prompt = render(
                DECOMPOSITION_USER_PROMPT,
                question=self.input_data.user_question,
                excerpt=context,
                sentence=disambiguated_sentence,
            )
            messages: list = [
                EasyInputMessageParam(role="system", content=DECOMPOSITION_SYSTEM_PROMPT),
                EasyInputMessageParam(role="user", content=user_prompt),
            ]
            async with create_client(provider=self.provider) as client:
                response = await client.responses.parse(
                    model=self.claim_extraction_model,
                    input=messages,
                    text_format=DecompositionResult,
                    store=False,
                )

                if response.output_parsed:
                    claims.extend(
                        response.output_parsed.specific_verifiable_decontextualized_propositions_with_clarifications
                    )

        return OutputDecompositionStep(claims=claims)


if __name__ == "__main__":

    async def main() -> None:
        user_question = "provide an overview of challenges in emerging markets?"
        sentence = """Argentina's rampant inflation, with monthly rates reaching as high as 25.5%, has made many goods unobtainable and plunged the value of the currency, causing severe economic hardship."""

        input_claim_extraction = InputClaimExtraction(
            sentence=sentence,
            excerpt=sentence,
            user_question=user_question,
        )

        claim_extraction = ClaimExtraction(input_claim_extraction)
        result = await claim_extraction.run()
        rich_print(result)

    asyncio.run(main())
