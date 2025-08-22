# Copyright (c) Microsoft. All rights reserved.

"""
An online evaluation to catch if the assistant failed to call a tool when it should have.

Output for each tool the probability/score that it should be called
"""

from liquid import render
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.responses import EasyInputMessageParam, ResponseInputParam
from pydantic import BaseModel, Field

from eval_recipes.schemas import EvaluationOutput, ToolEvaluationConfig
from eval_recipes.utils.llm import create_client
from eval_recipes.utils.responses_conversion import extract_tool_calls, extract_tool_info, format_full_history

PROBABILITY_SCALE = "0 to 100"
PROBABILITY_MIN = 0
PROBABILITY_MAX = 100


# This class for Structured Outputs only.
class ToolProbability(BaseModel):
    tool_name: str = Field(description="The name of the tool being evaluated.")
    reasoning: str = Field(description="Your reasoning for why or why not the tool should be called.")
    probability: float = Field(description=f"The probability from {PROBABILITY_SCALE} that the tool should be called.")


# This class for Structured Outputs only.
class ToolEvaluation(BaseModel):
    tool_evaluations: list[ToolProbability] = Field(description="A list of evaluations for each tool.")


class InputTool(BaseModel):
    tool_name: str
    tool_text: str
    was_called: bool
    # TODO: Move thresholds for each tool to the config
    threshold: float = Field(default=50)


class InputToolEvaluator(BaseModel):
    tools: list[InputTool]
    conversation_history_full: str


class OutputToolEvaluator(BaseModel):
    tool_evaluations: ToolEvaluation
    # Score: 100 if the called tool (if any) has probability >= its threshold, 0 otherwise
    # - true positives: cases where the probability >= threshold and the true label is 1.
    # - false negatives: cases where the probability < threshold and the true label is 1.
    # - false positives: cases where the probability >= threshold and the true label is 0.
    # - beta = 2, so recall is twice as important as precision.
    # This is making the assumption that calling too many tools is fine - Consider F1 or Fβ
    score: float


MISSED_TOOL_SYSTEM_PROMPT = f"""You are an evaluator tasked with determining the likelihood that tool(s) should be called in a particular scenario.
It is your job to write down what you think the probability of **each** tool being called is, along with your reasoning for the probability you will provide.
The probability of any particular tool needing to be called, is INDEPENDENT of the other tools, so you should provide a probability for each tool separately. \
Even though each tool should be considered independently, it might be the case that NO tools should be called \
and in that case the probability for each tool will be low. \
It could also be the case that **multiple tools should be called**, and in that case the probabilities for those tools will be high. \
IMPORTANT: It is very important to consider that multiple tools can be called at once.
You must first reflect your uncertainty and counter arguments for AND against calling the tool in your reasoning. \
Then reflect uncertainty in the probability estimate. It is rare that situations are 100% cut and dry.

You will be provided the conversation up until the point where the assistant might need to call a tool. \
Please note that this conversation will contain a system prompt which includes the instructions for THAT assistant. \
Absolutely do not let that confuse you with your own instructions.
You should not use the previous tool calls in the conversation to skew your probability estimates; focus on the likelihood based the current state. \
The rest of the context is there to help you make a better informed decision. \
You should especially pay attention to the user's intent for what they want to achieve and how they want the assistant to behave.

The probability should be a number between {PROBABILITY_SCALE} and can include decimals. \
A score of {PROBABILITY_MIN} means there is no chance that calling the tool makes sense, \
while a score of {PROBABILITY_MAX} means that it is certain that the tool should be called."""

MISSED_TOOL_USER_PROMPT = """Conversation:
{{conversation}}

Tools:
{{tools}}"""


class ToolEvaluator:
    def __init__(self, config: ToolEvaluationConfig | None = None) -> None:
        """
        Initialize the ToolEvaluator.

        Args:
            input: InputToolEvaluator containing tools and conversation data to evaluate
            provider: The AI provider to use ("openai" or "openai")
            model: The model to use for evaluation (default: "o3")
        """
        self.config = config or ToolEvaluationConfig()

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
        input_data = InputToolEvaluator(
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

    async def run(self, input: InputToolEvaluator) -> OutputToolEvaluator:
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
        return OutputToolEvaluator(tool_evaluations=validated_result, score=score)

    def _format_tools(self, input: InputToolEvaluator) -> str:
        """Format tools as XML for the prompt."""
        tools_xml = ""
        for tool in input.tools:
            tools_xml += f"<tool name='{tool.tool_name}'>\n"
            tools_xml += f"<description>{tool.tool_text}</description>\n"
            tools_xml += "</tool>\n\n"
        return tools_xml.strip()

    def _validate_tool_evaluation(self, result: ToolEvaluation, input: InputToolEvaluator) -> ToolEvaluation:
        """Validate and fix tool evaluation results.

        Applies these validation rules:
        - Clamps each probability to the valid range [PROBABILITY_MIN, PROBABILITY_MAX]
        - Ensures each input tool has an evaluation; creates missing ones with empty reasoning and PROBABILITY_MIN
        """

        validated_evaluations = []
        evaluated_tool_names = set()
        for tool_eval in result.tool_evaluations:
            # Clamp probability to valid range
            clamped_probability = max(PROBABILITY_MIN, min(PROBABILITY_MAX, tool_eval.probability))

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
                        probability=PROBABILITY_MIN,
                    )
                )

        return ToolEvaluation(tool_evaluations=validated_evaluations)

    def _compute_metric(self, tool_evaluations: ToolEvaluation, input: InputToolEvaluator) -> float:
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

    def _feedback(self, results: OutputToolEvaluator, input: InputToolEvaluator) -> str | None:
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
