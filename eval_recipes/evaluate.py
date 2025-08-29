# Copyright (c) Microsoft. All rights reserved.

import asyncio

from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.responses import ResponseInputParam

from eval_recipes.evaluations.check_criteria.check_criteria_evaluator import (
    CheckCriteriaEvaluator,
    CheckCriteriaEvaluatorConfig,
)
from eval_recipes.evaluations.claim_verification.claim_verification_evaluator import (
    ClaimVerificationEvaluator,
    ClaimVerificationEvaluatorConfig,
)
from eval_recipes.evaluations.guidance.guidance_evaluator import GuidanceEvaluator, GuidanceEvaluatorConfig
from eval_recipes.evaluations.preference_adherence.preference_adherence_evaluator import PreferenceAdherenceEvaluator
from eval_recipes.evaluations.tool_usage.tool_usage_evaluator import ToolUsageEvaluator, ToolUsageEvaluatorConfig
from eval_recipes.schemas import BaseEvaluatorConfig, EvaluationOutput, EvaluatorProtocol


async def evaluate(
    messages: ResponseInputParam,
    tools: list[ChatCompletionToolParam],
    evaluations: list[str | type[EvaluatorProtocol]],
    evaluation_configs: dict[str, BaseEvaluatorConfig] | None = None,
    max_concurrency: int = 1,
) -> list[EvaluationOutput]:
    """
    Evaluates the model's performance based on the provided messages and tools over the specified evaluations.

    Args:
        messages: OpenAI responses API input messages
        tools: OpenAI tool definitions
        evaluations: The list of evaluation names or custom evaluator classes to perform.
          Built-in options are: "claim_verification", "tool_usage", "guidance", and "preference_adherence".
          You can also pass custom evaluator classes that implement the EvaluatorProtocol.
        evaluation_configs: Optional configs for each evaluation.
          Keys should be the built-in evaluation names or custom evaluator class names.
          If not provided, defaults will be used.
        max_concurrency: Maximum number of evaluations to run concurrently. Default is 1 (sequential).

    Returns:
        A list of EvaluationOutput objects containing the evaluation results.
        Each object includes the evaluation name, score, and optional metadata specific to that evaluation.
    """
    if evaluation_configs is None:
        evaluation_configs = {}

    evaluator_map: dict[str, tuple[type, type[BaseEvaluatorConfig]]] = {
        "guidance": (GuidanceEvaluator, GuidanceEvaluatorConfig),
        "preference_adherence": (PreferenceAdherenceEvaluator, BaseEvaluatorConfig),
        "tool_usage": (ToolUsageEvaluator, ToolUsageEvaluatorConfig),
        "claim_verification": (ClaimVerificationEvaluator, ClaimVerificationEvaluatorConfig),
        "check_criteria": (CheckCriteriaEvaluator, CheckCriteriaEvaluatorConfig),
    }

    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_evaluation(
        eval_item: str | type[EvaluatorProtocol],
    ) -> EvaluationOutput | None:
        async with semaphore:
            # Handle string evaluation names (built-in evaluators)
            if isinstance(eval_item, str):
                if eval_item not in evaluator_map:
                    return None
                evaluator_class, config_type = evaluator_map[eval_item]
                config = evaluation_configs.get(eval_item)
                validated_config = config if isinstance(config, config_type) else None
                evaluator = evaluator_class(config=validated_config)
            # Handle custom evaluator classes
            else:
                # Check if the class implements the required protocol
                if not isinstance(eval_item, type) or not issubclass(eval_item, EvaluatorProtocol):
                    raise ValueError(f"Custom evaluator {eval_item} must be a class that implements EvaluatorProtocol")
                # Get config using the class name
                class_name = eval_item.__name__
                config = evaluation_configs.get(class_name)
                evaluator = eval_item(config=config)

            return await evaluator.evaluate(messages, tools)

    tasks = [run_evaluation(eval_item) for eval_item in evaluations]
    results = await asyncio.gather(*tasks)
    return [result for result in results if result is not None]
