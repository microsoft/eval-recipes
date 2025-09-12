# Copyright (c) Microsoft. All rights reserved.

import asyncio
from collections.abc import AsyncGenerator
from typing import Literal

from liquid import render
from loguru import logger
import nltk
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.responses import EasyInputMessageParam, ResponseInputParam
from openai.types.shared_params.reasoning import Reasoning
from pydantic import BaseModel, Field

from eval_recipes.evaluations.claim_verification.claim_extraction import ClaimExtraction, InputClaimExtraction
from eval_recipes.evaluations.claim_verification.prompts import (
    VERDICT_GENERATION_SYSTEM_PROMPT,
    VERDICT_GENERATION_USER_PROMPT,
)
from eval_recipes.evaluations.claim_verification.schemas import (
    InputClaimVerificationEvaluator,
    InputContext,
    OutputCitation,
    OutputClaimVerificationEvaluator,
    OutputClaimVerificationEvaluatorMetrics,
    OutputClaimVerificationStep,
    OutputSentenceSplittingStep,
)
from eval_recipes.evaluations.claim_verification.utils import FormattedContext
from eval_recipes.schemas import BaseEvaluatorConfig, EvaluationOutput
from eval_recipes.utils.llm import create_client
from eval_recipes.utils.responses_conversion import extract_last_msg, format_messages_as_context


class ClaimVerificationEvaluatorConfig(BaseEvaluatorConfig):
    claim_extraction_model: Literal["gpt-5", "gpt-5-mini", "gpt-5-nano", "o3", "o4-mini"] = Field(default="gpt-5-mini")
    verification_model: Literal["gpt-5", "gpt-5-mini", "gpt-5-nano", "o3", "o4-mini"] = Field(default="gpt-5")
    verification_reasoning_effort: Literal["minimal", "low", "medium", "high"] = Field(default="high")
    verification_threshold: float = Field(default=70)  # Score threshold above which a claim is considered verified
    is_open_domain_threshold: float = Field(default=50)  # Score threshold above which a claim is considered open-domain
    max_line_length: int = Field(default=200)
    max_concurrency: int = Field(default=1)
    ignore_tool_names: list[str] = Field(
        default=[],
        description="A list of tool names to ignore as source context.",
    )


# This class is for Structured Outputs only.
class Citations(BaseModel):
    source_id: str = Field(description="The ID of the source context that supports the claim.")
    start_range: int = Field(description="The starting line number of the content that supports the claim (inclusive)")
    end_range: int = Field(
        description="The ending line number of the content that supports the claim (exclusive). If through your reasoning you determine that the claim is not supported, set this equivalent to start_range."
    )


# This class is for Structured Outputs only.
class ClaimVerificationResult(BaseModel):
    proof: str = Field(
        description="Your justification for why the claim is grounded in the source, with the content from start_range to end_range as proof."
    )
    citations: list[Citations] = Field(description="Each of the citations supporting the claim.")
    verified_probability: float = Field(
        description="Now that you have a justification for if the claim is verified if so generated citations, determine the probability (on a scale of 0 to 100) that the claim is truly verified."
    )
    open_domain_justification: str = Field(
        description="If the claim is not supported with any citations, your justification on if it should be considered as an open-domain claim or not. Be sure to reason about both the case where the claim is and is not open-domain."
    )
    open_domain_probability: float = Field(
        description="Now that you have a justification for if the claim is open-domain, determine the probability (on a scale of 0 to 100) that the claim is truly open-domain."
    )
    is_open_domain: bool = Field(description="true if the claim is open-domain, false otherwise")


class ClaimVerificationEvaluator:
    def __init__(self, config: ClaimVerificationEvaluatorConfig | None = None) -> None:
        """
        Verifies claims by extracting them from text and checking them against source context.
        """
        self.config = config or ClaimVerificationEvaluatorConfig()

        # Initialize NLTK for sentence splitting
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

    async def evaluate(self, messages: ResponseInputParam, tools: list[ChatCompletionToolParam]) -> EvaluationOutput:
        last_assistant_msg = extract_last_msg(messages, "assistant")
        last_user_msg = extract_last_msg(messages, "user")
        context_dicts = format_messages_as_context(
            messages,
            ignore_roles=["assistant", "function_call"],
            ignore_tool_names=self.config.ignore_tool_names,
        )
        # Convert dict format to InputContext objects
        source_context = [
            InputContext(source_id=ctx["source_id"], title=ctx["title"], content=ctx["content"])
            for ctx in context_dicts
        ]
        input_data = InputClaimVerificationEvaluator(
            text=last_assistant_msg,
            user_question=last_user_msg,
            source_context=source_context,
        )

        results = []
        async for result in self.run(input_data):
            results.append(result)

        # The last result should be the final summary metrics
        if results and isinstance(results[-1], OutputClaimVerificationEvaluatorMetrics):
            metrics = results[-1]
            claims = results[:-1]  # All results except the last one are claims
            return EvaluationOutput(
                eval_name="claim_verification",
                applicable=not metrics.ignore_metric_recommended,
                score=metrics.closed_domain_supported,
                feedback=self._feedback([x for x in results if isinstance(x, OutputClaimVerificationEvaluator)]),
                metadata={
                    "total_claims": metrics.total_claims,
                    "num_closed_domain_supported": metrics.num_closed_domain_supported,
                    "num_open_domain_claims": metrics.num_open_domain_claims,
                    "total_claims_closed_domain": metrics.total_claims_closed_domain,
                    "claims": [claim.model_dump(mode="json") for claim in claims],
                },
            )

        return EvaluationOutput(
            eval_name="claim_verification",
            applicable=False,
            score=0.0,
            metadata={"error": "No claims were extracted"},
        )

    async def run(
        self, input: InputClaimVerificationEvaluator
    ) -> AsyncGenerator[OutputClaimVerificationEvaluator | OutputClaimVerificationEvaluatorMetrics, None]:
        """
        Runs claim verification on the input text.

        Yields:
        OutputClaimVerificationEvaluator: Individual claim verification results containing:
            - sentence: Original sentence containing the claim
            - start_index: Start position of the sentence in the original text
            - end_index: End position of the sentence in the original text
            - claim: The claim being verified
            - proof: Justification for verification result
            - citations: List of supporting citations (empty if unsupported). Each citation contains:
                - source_id: ID of the source context matching the InputContext
                - cited_text: Text from the source that supports the claim
            - is_open_domain: Whether claim was determined to be open-domain

        OutputClaimVerificationEvaluatorMetrics: Final metrics object (yielded last) containing:
            - total_claims: Total number of claims processed
            - closed_domain_supported: Percentage of closed-domain claims supported
            - ignore_metric_recommended: Whether closed_domain_supported is recommended to be ignored due to high likelihood of an answer that does not need verification. Heuristically determined.
            - number_supported_claims: Count of claims with citations
            - num_open_domain_claims: Count of claims considered open-domain
            - number_not_supported_claims: Count of claims not supported by citations or open-domain
        """
        logger.info("Starting claim verification")

        self.input = input
        self.formatted_context = FormattedContext(input, self.config.max_line_length)

        sentences = self._sentence_splitting(p=2, f=2)
        logger.info(f"Split text into {len(sentences)} sentences")

        semaphore = asyncio.Semaphore(self.config.max_concurrency)
        tasks = [self._process_sentence_with_semaphore(semaphore, i, sentence) for i, sentence in enumerate(sentences)]

        # Process results as they complete and yield individual claims
        accumulated_results: list[OutputClaimVerificationEvaluator] = []
        for completed_task in asyncio.as_completed(tasks):
            sentence_results = await completed_task
            for result in sentence_results:
                accumulated_results.append(result)
                yield result

        logger.info(f"Claim verification complete. Total claims verified: {len(accumulated_results)}")

        metrics = self._compute_metrics(accumulated_results)
        yield metrics

    async def _process_sentence_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        sentence_index: int,
        sentence: OutputSentenceSplittingStep,
    ) -> list[OutputClaimVerificationEvaluator]:
        """Process a single sentence with semaphore control for concurrency limiting."""
        async with semaphore:
            return await self._process_sentence(sentence_index, sentence)

    async def _process_sentence(
        self, sentence_index: int, sentence: OutputSentenceSplittingStep
    ) -> list[OutputClaimVerificationEvaluator]:
        """
        Extracts all claims from the sentence.
        Then verifies the claim using the claim_verifier model with parallel processing.
        """
        logger.info(f'Processing sentence {sentence_index + 1}: "{sentence.sentence[:100]!r}..."')

        # Extract all claims from the sentence
        sentence_input = InputClaimExtraction(
            sentence=sentence.sentence,
            excerpt=sentence.sentence_with_surrounding_context,
            user_question=self.input.user_question,
        )
        claim_extractor = ClaimExtraction(
            input_data=sentence_input,
            provider=self.config.provider,
            claim_extraction_model=self.config.claim_extraction_model,
        )
        claims: list[str] = await claim_extractor.run()

        # Verify each claim in parallel
        claim_semaphore = asyncio.Semaphore(self.config.max_concurrency)
        verification_tasks = []
        for claim_output in claims:
            logger.info(f"Verifying claim: {claim_output[:100]}...")
            task = self._verify_single_claim_with_semaphore(claim_semaphore, sentence, claim_output)
            verification_tasks.append(task)

        results = await asyncio.gather(*verification_tasks)
        return results

    async def _verify_single_claim_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        sentence: OutputSentenceSplittingStep,
        claim_output: str,
    ) -> OutputClaimVerificationEvaluator:
        async with semaphore:
            claim_verification_result = await self._claim_verifier(claim=claim_output)
            combined_result = OutputClaimVerificationEvaluator(
                sentence=sentence.sentence,
                start_index=sentence.start_index,
                end_index=sentence.end_index,
                claim=claim_output,
                proof=claim_verification_result.proof,
                citations=claim_verification_result.citations,
                verified_probability=claim_verification_result.verified_probability,
                open_domain_justification=claim_verification_result.open_domain_justification,
                open_domain_probability=claim_verification_result.open_domain_probability,
                is_open_domain=claim_verification_result.is_open_domain,
            )
            return combined_result

    def _sentence_splitting(self, p: int = 2, f: int = 2) -> list[OutputSentenceSplittingStep]:
        """Split the text into sentences with configurable context.

        Args:
            p: Number of preceding sentences to include in context
            f: Number of following sentences to include in context

        Returns:
            List of SentenceContext objects containing sentence and context information
        """
        text = self.input.text
        if not text:
            return []

        sentences = nltk.sent_tokenize(text)

        result = []
        current_position = 0
        for i, sentence in enumerate(sentences):
            # Find the actual character position of this sentence in the original text
            start_index = text.find(sentence, current_position)
            end_index = start_index + len(sentence)
            current_position = end_index

            # Calculate context window boundaries
            context_start_idx = max(0, i - p)
            context_end_idx = min(len(sentences), i + f + 1)

            context_sentences = sentences[context_start_idx:context_end_idx]
            sentence_with_surrounding_context = " ".join(context_sentences)

            sentence_context = OutputSentenceSplittingStep(
                sentence=sentence,
                start_index=start_index,
                end_index=end_index,
                preceding_sentences=sentences[context_start_idx:i],
                following_sentences=sentences[i + 1 : context_end_idx],
                sentence_with_surrounding_context=sentence_with_surrounding_context,
            )
            result.append(sentence_context)
        return result

    async def _claim_verifier(self, claim: str) -> OutputClaimVerificationStep:
        user_prompt = render(
            VERDICT_GENERATION_USER_PROMPT,
            context=self.formatted_context.format_as_xml(),
            user_ask=self.input.user_question,
            claim=claim,
        )
        messages: list = [
            EasyInputMessageParam(role="system", content=VERDICT_GENERATION_SYSTEM_PROMPT),
            EasyInputMessageParam(role="user", content=user_prompt),
        ]

        async with create_client(provider=self.config.provider) as client:
            response = await client.responses.parse(
                model=self.config.model,
                input=messages,
                text_format=ClaimVerificationResult,
                reasoning=Reasoning(
                    effort=self.config.verification_reasoning_effort,
                ),
                store=False,
            )

        if response.output_parsed is None:
            structured_result = ClaimVerificationResult(
                proof="",
                citations=[],
                verified_probability=0.0,
                open_domain_justification="",
                open_domain_probability=0.0,
                is_open_domain=False,
            )
        else:
            structured_result = response.output_parsed

        return self._convert_to_output(structured_result)

    def _convert_to_output(self, result: ClaimVerificationResult) -> OutputClaimVerificationStep:
        """Convert ClaimVerificationResult to OutputClaimVerificationStep."""
        output_citations = []
        for citation in result.citations:
            # Skip citations where start_range == end_range (indicates unsupported/invalid citation)
            if citation.start_range == citation.end_range:
                continue

            cited_text = self.formatted_context.get_cited_text(
                citation.source_id, citation.start_range, citation.end_range
            )
            output_citations.append(OutputCitation(source_id=citation.source_id, cited_text=cited_text))

        return OutputClaimVerificationStep(
            proof=result.proof,
            citations=output_citations,
            verified_probability=result.verified_probability,
            open_domain_justification=result.open_domain_justification,
            open_domain_probability=result.open_domain_probability,
            is_open_domain=result.is_open_domain,
        )

    def _compute_metrics(
        self, results: list[OutputClaimVerificationEvaluator]
    ) -> OutputClaimVerificationEvaluatorMetrics:
        num_closed_domain_supported = 0
        num_open_domain_claims = 0
        total_claims_closed_domain = 0
        for r in results:
            is_supported = r.verified_probability >= self.config.verification_threshold
            is_open_domain = r.open_domain_probability >= self.config.is_open_domain_threshold
            if not is_open_domain:
                total_claims_closed_domain += 1
                if is_supported:
                    num_closed_domain_supported += 1
            else:
                num_open_domain_claims += 1

        # Calculate closed domain supported percentage
        # Closed domain claims are those that are not open domain
        closed_domain_supported = (
            (num_closed_domain_supported / total_claims_closed_domain) * 100 if total_claims_closed_domain > 0 else 0
        )

        # Compute ignore metric recommendation
        # We should set ignore_metric_recommended to True when:
        # - There are less than or equal to 2 claims
        # - The number of open domain claims is greater than the number of closed domain claims AND the closed_domain_supported metric is less than 15%
        total_claims = len(results)
        ignore_metric_recommended = total_claims <= 2 or (
            num_open_domain_claims > total_claims_closed_domain and closed_domain_supported < 15.0
        )

        return OutputClaimVerificationEvaluatorMetrics(
            total_claims=total_claims,
            closed_domain_supported=closed_domain_supported,
            ignore_metric_recommended=ignore_metric_recommended,
            num_closed_domain_supported=num_closed_domain_supported,
            num_open_domain_claims=num_open_domain_claims,
            total_claims_closed_domain=total_claims_closed_domain,
        )

    def _feedback(self, results: list[OutputClaimVerificationEvaluator]) -> str | None:
        """For each claim that was not verified (no citations) and not open domain,
        Create a string:

        <sentence>Looking forward, Microsoft's dominant position...</sentence>
        <unverified_reasoning>Not fully supported. The provided earnings releases ...</unverified_reasoning>

        <sentence>Comparing the two quarters...</sentence>
        <unverified_reasoning>The provided context (earnings releases and user requests) contains...</unverified_reasoning>
        ...

        If all claims were verified or all were open domain, return None
        """
        unverified_claims = [r for r in results if not r.citations and not r.is_open_domain]
        if not unverified_claims:
            return None

        feedback_parts = []
        feedback_parts.append(
            """The following sentences were determined to possibly have portions of them that have unverified content \
in them and reasoning provides the justification for that."""
        )
        for claim in unverified_claims:
            feedback_parts.append(
                f"<sentence>{claim.sentence}</sentence>\n<unverified_reasoning>{claim.proof}</unverified_reasoning>\n"
            )
        return "\n".join(feedback_parts).strip()
