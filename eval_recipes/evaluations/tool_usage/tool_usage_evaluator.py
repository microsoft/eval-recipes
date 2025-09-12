# Copyright (c) Microsoft. All rights reserved.

"""
An online evaluation to catch if the assistant failed to call a tool when it should have.

Output for each tool the probability/score that it should be called
"""

from liquid import render
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.responses import EasyInputMessageParam, ResponseInputParam
from pydantic import BaseModel, Field

from eval_recipes.evaluations.tool_usage.prompts import MISSED_TOOL_SYSTEM_PROMPT, MISSED_TOOL_USER_PROMPT
from eval_recipes.schemas import BaseEvaluatorConfig, EvaluationOutput
from eval_recipes.utils.llm import create_client
from eval_recipes.utils.responses_conversion import extract_tool_calls, extract_tool_info, format_full_history


class ToolUsageEvaluatorConfig(BaseEvaluatorConfig):
    tool_thresholds: dict[str, float] = Field(
        default={},
        description="A dictionary mapping tool names to the threshold probabilities indicated that the tool should be called.",
    )


# This class for Structured Outputs only.
class ToolProbability(BaseModel):
    tool_name: str = Field(description="The name of the tool being evaluated.")
    reasoning: str = Field(description="Your reasoning for why or why not the tool should be called.")
    probability: float = Field(description="The probability from 0 to 100 that the tool should be called.")


# This class for Structured Outputs only.
class ToolEvaluation(BaseModel):
    tool_evaluations: list[ToolProbability] = Field(description="A list of evaluations for each tool.")


class InputTool(BaseModel):
    tool_name: str
    tool_text: str
    was_called: bool
    # TODO: Move thresholds for each tool to the config
    threshold: float = Field(default=50)


class InputToolUsageEvaluator(BaseModel):
    tools: list[InputTool]
    conversation_history_full: str


class OutputToolUsageEvaluator(BaseModel):
    tool_evaluations: ToolEvaluation
    score: float


class ToolUsageEvaluator:
    def __init__(self, config: ToolUsageEvaluatorConfig | None = None) -> None:
        """
        Initialize the ToolUsageEvaluator.

        Args:
            config: Optional ToolUsageEvaluatorConfig. If not provided, defaults will be used.
        """
        self.config = config or ToolUsageEvaluatorConfig()

    async def evaluate(self, messages: ResponseInputParam, tools: list[ChatCompletionToolParam]) -> EvaluationOutput:
        # If no tools are provided, return not applicable
        if not tools:
            return EvaluationOutput(
                eval_name="tool_usage",
                applicable=False,
                score=0.0,
                metadata={"tool_evaluations": "No tools provided"},
            )

        tools_called = extract_tool_calls(messages)
        tool_infos = extract_tool_info(tools)
        input_tools: list[InputTool] = [
            InputTool(
                tool_name=tool_name,
                tool_text=tool_info,
                was_called=tools_called.get(tool_name, False),
                threshold=self.config.tool_thresholds.get(tool_name, 50),  # Default to 50 if not specified
            )
            for tool_name, tool_info in tool_infos.items()
        ]
        input_data = InputToolUsageEvaluator(
            tools=input_tools,
            # TODO: Consider making this also include any subsequent assistant messages
            conversation_history_full=format_full_history(messages, only_upto_last_user=True),
        )
        results = await self.run(input_data)
        output = EvaluationOutput(
            eval_name="tool_usage",
            applicable=True,
            score=results.score,
            feedback=self._feedback(results, input_data),
            metadata={"tool_evaluations": results.tool_evaluations.model_dump(mode="json")},
        )
        return output

    async def run(self, input: InputToolUsageEvaluator) -> OutputToolUsageEvaluator:
        user_prompt = render(
            MISSED_TOOL_USER_PROMPT,
            conversation=input.conversation_history_full,
            tools=self._format_tools(input),
        )
        messages: list = [
            EasyInputMessageParam(role="system", content=MISSED_TOOL_SYSTEM_PROMPT),
            EasyInputMessageParam(role="user", content=user_prompt),
        ]

        async with create_client(provider=self.config.provider) as client:
            response = await client.responses.parse(
                model=self.config.model,
                input=messages,
                text_format=ToolEvaluation,
                store=False,
            )

        if response.output_parsed is None:
            structured_result = ToolEvaluation(tool_evaluations=[])
        else:
            structured_result = response.output_parsed

        validated_result = self._validate_tool_evaluation(structured_result, input)
        score = self._compute_metric(validated_result, input)
        return OutputToolUsageEvaluator(tool_evaluations=validated_result, score=score)

    def _format_tools(self, input: InputToolUsageEvaluator) -> str:
        """Format tools as XML for the prompt."""
        tools_xml = ""
        for tool in input.tools:
            tools_xml += f"<tool name='{tool.tool_name}'>\n"
            tools_xml += f"<description>{tool.tool_text}</description>\n"
            tools_xml += "</tool>\n\n"
        return tools_xml.strip()

    def _validate_tool_evaluation(self, result: ToolEvaluation, input: InputToolUsageEvaluator) -> ToolEvaluation:
        """Validate and fix tool evaluation results.

        Applies these validation rules:
        - Clamps each probability to the valid range [0, 100]
        - Ensures each input tool has an evaluation; creates missing ones with empty reasoning and 0
        """

        validated_evaluations = []
        evaluated_tool_names = set()
        for tool_eval in result.tool_evaluations:
            clamped_probability = max(0, min(100, tool_eval.probability))
            validated_evaluations.append(
                ToolProbability(
                    tool_name=tool_eval.tool_name,
                    reasoning=tool_eval.reasoning,
                    probability=clamped_probability,
                )
            )
            evaluated_tool_names.add(tool_eval.tool_name)

        # Add missing tool evaluations
        for tool in input.tools:
            if tool.tool_name not in evaluated_tool_names:
                validated_evaluations.append(
                    ToolProbability(
                        tool_name=tool.tool_name,
                        reasoning="",
                        probability=0,
                    )
                )
        return ToolEvaluation(tool_evaluations=validated_evaluations)

    def _compute_metric(self, tool_evaluations: ToolEvaluation, input: InputToolUsageEvaluator) -> float:
        """
        - If no tools should be called (was_called=False for all)
          - Each probability should be below the threshold. If so, return 100, 0 otherwise.
        - If was_called=True for any tool:
          - Check if any of those tool's probabilities are above its threshold, if so return 100.
            Otherwise if none of the "was_called=True" have a probability over the threshold, return 0.

        Args:
            tool_evaluations: The tool evaluation results

        Returns:
            Score as 100.0 (success) or 0.0 (failure)
        """
        # Find all tools that should be called
        called_tools = [tool for tool in input.tools if tool.was_called]

        # Create a mapping of tool evaluations by name for easier lookup
        eval_by_name = {eval.tool_name: eval for eval in tool_evaluations.tool_evaluations}

        # Case 1: No tools should be called (was_called=False for all)
        if len(called_tools) == 0:
            # Check that ALL tool probabilities are below their thresholds
            for tool in input.tools:
                tool_eval = eval_by_name.get(tool.tool_name)
                if tool_eval is None:
                    continue

                if tool_eval.probability >= tool.threshold:
                    return 0.0
            return 100.0

        # Case 2: One or more tools should be called (was_called=True for any tools)
        else:
            for called_tool in called_tools:
                tool_eval = eval_by_name.get(called_tool.tool_name)

                probability = 0.0 if tool_eval is None else tool_eval.probability

                if probability >= called_tool.threshold:
                    return 100.0

            return 0.0

    def _feedback(self, results: OutputToolUsageEvaluator, input: InputToolUsageEvaluator) -> str | None:
        """Writes a string that states what tool should have been or not been called
        based on the output of the evaluator.
        Case 1: If no tools should be called, but tools were called, state that no tools should have been called.
        Case 2: If a tool should have been called, but none or a different one was called, state which tool(s) could be called.
        """
        called_tools = [tool for tool in input.tools if tool.was_called]
        eval_by_name = {eval.tool_name: eval for eval in results.tool_evaluations.tool_evaluations}

        # Case 1
        if len(called_tools) == 0:
            # Check if any tool probability is above its threshold
            tools_above_threshold = []
            for tool in input.tools:
                tool_eval = eval_by_name.get(tool.tool_name)
                if tool_eval and tool_eval.probability >= tool.threshold:
                    tools_above_threshold.append((tool.tool_name, tool_eval.reasoning))

            # If there are tools above threshold, we should have called them (score would be 0)
            if tools_above_threshold:
                feedback_parts = ["Based on the conversation context, the following tool(s) could have been called:"]
                for tool_name, reasoning in tools_above_threshold:
                    feedback_parts.append(f"<tool>{tool_name}</tool>\n<reasoning>{reasoning}</reasoning>\n")
                return "\n".join(feedback_parts).strip()

        # Case 2: One or more tools were called (was_called=True for any tools)
        else:
            # Check if any called tool has probability >= threshold
            for called_tool in called_tools:
                tool_eval = eval_by_name.get(called_tool.tool_name)
                if tool_eval and tool_eval.probability >= called_tool.threshold:
                    return None

            # None of the called tools have probability >= threshold (score would be 0)
            # Find tools that should have been called instead
            tools_that_should_be_called = []
            for tool in input.tools:
                tool_eval = eval_by_name.get(tool.tool_name)
                if tool_eval and tool_eval.probability >= tool.threshold:
                    tools_that_should_be_called.append((tool.tool_name, tool_eval.reasoning))

            if tools_that_should_be_called:
                feedback_parts = [
                    "Based on the conversation context, the following tool(s) should have been called instead:"
                ]
                for tool_name, reasoning in tools_that_should_be_called:
                    feedback_parts.append(f"<tool>{tool_name}</tool>\n<reasoning>{reasoning}</reasoning>\n")
                return "\n".join(feedback_parts).strip()
            else:
                # No tools should have been called
                return "Based on the conversation context, no tools should have been called."

        return None
