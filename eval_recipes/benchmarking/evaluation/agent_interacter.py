# Copyright (c) Microsoft. All rights reserved.

from typing import Literal

from liquid import render
from openai.types.shared_params.reasoning import Reasoning
from pydantic import BaseModel, Field
import tiktoken

from eval_recipes.utils.llm import create_client

INTERACTION_SYSTEM_PROMPT = """You are observing an agent executing a task and "roleplaying" as a user who just wants the task to be done.
Your job is to decide whether to send a message to the agent to get it to finish the task, or decide that it is already done to decide and no message is needed. \
You are NOT assessing if the task was done correctly, you are only deciding if you should reply or not.

When a reply to the agent is needed, it will often be to answer questions that the agent has asked. \
You will be provided with the original task instructions and you must use that as the sole context for how to reply to the agent. \
If there is ambiguity, you must tell the agent to choose what it thinks is best and encourage it to just go and complete the task to the best of its ability.
DO NOT add any new instructions or requirements that were not in the original task instructions.

You will be provided the raw log of the agent's execution which includes all messages and tool calls. \
The agent's last message will be at the very end of the log, the rest is provided as context.
You will also be provided with the original task instructions. You should use this to reply, if needed."""

INTERACTION_USER_PROMPT = """The user originally asked the agent to the following:
{{task_instructions}}

This is the log of what the agent has done:
{{agent_log}}

Now reason about if you need to send a message to get the agent to go ahead and complete the task.
If you decide a message is needed, set should_reply to true and write the message to send to the agent.
Otherwise, set should_reply to false and leave the reply_to_agent field empty."""


class ResponseToAgent(BaseModel):
    """Structured output for the continuation decision."""

    was_task_completed_reasoning: str = Field(
        description="Your reasoning about if the agent stated that it completed the task or not."
    )
    is_agent_asking_for_answers_or_clarifications_reasoning: str = Field(
        description="Your reasoning about if the agent is asking for clarifications or answers to questions."
    )
    was_task_completed: bool = Field(description="Decision on whether the agent has completed the task or not.")
    is_agent_asking_for_answers_or_clarifications: bool = Field(
        description="Decision on whether the agent is asking for clarifications or answers to questions."
    )
    reply_to_agent: str = Field(
        description="The message to send to the agent to get it to complete original task if task was not completed and agent is asking for answers or clarifications. Leave empty if no reply is needed."
    )


def _truncate_to_token_limit(text: str, max_tokens: int = 80000) -> str:
    """Truncate text to max_tokens, removing from the beginning to keep recent content."""
    encoding = tiktoken.get_encoding("o200k_base")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    # Keep the last max_tokens tokens (truncate from beginning)
    truncated_tokens = tokens[-max_tokens:]
    return encoding.decode(truncated_tokens)


async def interact_with_agent(
    agent_log: str,
    task_instructions: str,
    provider: Literal["openai", "azure_openai"] = "openai",
    model: str = "gpt-5.1",
) -> str:
    """
    Sends a prompt to an LLM to decide if a continuation message should be sent to the agent.

    Uses structured outputs to determine:
    1. Whether the task was completed
    2. Whether the agent is asking for clarifications

    Only returns a reply if the task is NOT completed AND the agent IS asking for clarifications.

    Args:
        agent_log: The log of the agent's execution
        task_instructions: The original task instructions
        provider: The LLM provider to use (openai or azure_openai)
        model: The model to use for continuation decision

    Returns:
        The message to send to the agent, or empty string if no reply is needed.
    """
    # Truncate agent_log to prevent exceeding context limits
    truncated_log = _truncate_to_token_limit(agent_log)
    user_prompt = render(INTERACTION_USER_PROMPT, task_instructions=task_instructions, agent_log=truncated_log)
    messages: list = [
        {"role": "system", "content": INTERACTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    async with create_client(provider=provider) as client:
        response = await client.responses.parse(
            model=model,
            input=messages,
            text_format=ResponseToAgent,
            reasoning=Reasoning(
                effort="low",
            ),
            store=False,
        )

    if response.output_parsed:
        parsed_response: ResponseToAgent = response.output_parsed
        # Only reply if task is NOT completed AND agent IS asking for clarifications
        if not parsed_response.was_task_completed and parsed_response.is_agent_asking_for_answers_or_clarifications:
            return parsed_response.reply_to_agent

    return ""
