# Copyright (c) Microsoft. All rights reserved.

"""
Conversion utilities for OpenAI Responses Inputs.
"""

import json
from typing import Literal

from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.responses import ResponseInputParam, ResponseInputTextParam


def format_full_history(
    messages: ResponseInputParam,
    remove_system_messages: bool = False,
    only_upto_last_user: bool = False,
    remove_last_assistant: bool = False,
) -> str:
    """
    Convert conversation history into XML-like syntax.

    Args:
        messages: List of message items from ResponseInputParam
        remove_system_messages: If True, skip system messages in the output
        only_upto_last_user: If True, only include messages up to, and including, the last user message
        remove_last_assistant: If True, removes the final assistant message, iff it is the last message

    Returns:
        Formatted XML-like string representation of the conversation
    """

    # If only_upto_last_user is True, find the index of the last user message
    if only_upto_last_user:
        last_user_index = -1
        for i, item in enumerate(messages):
            if isinstance(item, dict):
                role = item.get("role", "")
                if role == "user":
                    last_user_index = i
        # If we found a user message, truncate the messages list
        if last_user_index >= 0:
            messages = messages[: last_user_index + 1]

    # If remove_last_assistant is True, check if the last message is from assistant and remove it
    if remove_last_assistant and messages:
        last_item = messages[-1]
        if isinstance(last_item, dict) and last_item.get("role") == "assistant":
            messages = messages[:-1]

    formatted_parts = []
    for item in messages:
        if isinstance(item, dict):
            # Handle system, user, and assistant messages
            if (
                "role" in item
                and "content" in item
                and item.get("type") != "function_call"
            ):
                role = item["role"]

                if remove_system_messages and role == "system":
                    continue

                content = item["content"]
                # Handle content that might be a string or list
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    # Extract text from content list items
                    text_parts = []
                    for content_item in content:
                        if isinstance(content_item, dict):
                            if "text" in content_item:
                                text_parts.append(content_item["text"])
                            elif (
                                "type" in content_item
                                and content_item["type"] == "input_text"
                            ):
                                text_parts.append(content_item.get("text", ""))
                    text = "\n".join(text_parts)
                else:
                    text = str(content)

                formatted_parts.append(f'<message role="{role}">\n{text}\n</message>')

            item_type = item.get("type", "")
            if item_type == "function_call":
                call_id = item.get("call_id", "")
                name = item.get("name", "")
                arguments = item.get("arguments", "")
                formatted_parts.append(
                    f'<function_call call_id="{call_id}" tool_name="{name}">\n{arguments}\n</function_call>'
                )

            elif item_type == "function_call_output":
                call_id = item.get("call_id", "")
                output = item.get("output", "")
                formatted_parts.append(
                    f'<function_call_output call_id="{call_id}">\n{output}\n</function_call_output>'
                )

    return "\n".join(formatted_parts)


def extract_tool_calls(messages: ResponseInputParam) -> dict[str, bool]:
    """
    Extract the tools that were called since the last user message.

    Args:
        messages: List of message items from ResponseInputParam

    Returns:
        Mapping of the tool name to a boolean indicating whether the tool was called.
    """
    # Find the index of the last user message
    last_user_index = -1
    for i, item in enumerate(messages):
        if isinstance(item, dict):
            role = item.get("role", "")
            if role == "user":
                last_user_index = i

    # If no user message found, return empty dict
    if last_user_index < 0:
        return {}

    # Track tool calls after the last user message
    tool_calls: dict[str, str] = {}  # call_id -> tool_name mapping
    tools_called: dict[str, bool] = {}  # tool_name -> True
    for item in messages[last_user_index + 1 :]:
        if isinstance(item, dict):
            item_type = item.get("type", "")

            # Check for ResponseFunctionToolCallParam
            if item_type == "function_call":
                call_id = item.get("call_id", "")
                name = item.get("name", "")
                if call_id and name:
                    tool_calls[call_id] = name

            # Check for FunctionCallOutput
            elif item_type == "function_call_output":
                call_id = item.get("call_id", "")
                if call_id in tool_calls:
                    # Mark this tool as having been called
                    tool_name = tool_calls[call_id]
                    tools_called[tool_name] = True

    return tools_called


def extract_tool_info(tools: list[ChatCompletionToolParam]) -> dict[str, str]:
    """
    Extract the name and description + parameters from the tool definitions.
    The description and parameters are concatenated into a single string in XML format.

    Args:
        tools: List of ChatCompletionToolParam

    Returns:
        Mapping of tool names to their descriptions + parameters
    """

    tool_info: dict[str, str] = {}
    for tool in tools:
        if tool.get("type") == "function":
            function = tool.get("function", {})
            name = function.get("name", "")
            description = function.get("description", "")
            parameters = function.get("parameters", {})

            if name:
                # Format as XML-like structure
                xml_parts = [f'<tool name="{name}">']

                if description:
                    xml_parts.append(f"  <description>{description}</description>")

                if parameters:
                    # Convert parameters dict to formatted JSON string
                    params_json = json.dumps(parameters, indent=2)
                    # Indent each line for better XML formatting
                    params_lines = params_json.split("\n")
                    params_indented = "\n".join("    " + line for line in params_lines)
                    xml_parts.append(
                        f"  <parameters>\n{params_indented}\n  </parameters>"
                    )

                xml_parts.append("</tool>")
                tool_info[name] = "\n".join(xml_parts)

    return tool_info


def extract_last_msg(
    messages: ResponseInputParam, role: Literal["assistant", "user"]
) -> str:
    """
    Extract the content of the last message of the given role.

    Args:
        messages: List of message items from ResponseInputParam
        role: The role to filter for ("assistant" or "user")

    Returns:
        Content of the last message with the specified role, or an empty string if none found.
        If content is a list, returns only the first text item.
    """
    # Iterate through messages in reverse to find the last message with the specified role
    for item in reversed(messages):
        if isinstance(item, dict):
            # Check if this is a message with the desired role
            if item.get("role") == role:
                content = item.get("content", "")

                # Handle content that might be a string or list
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    # Extract only the first text item from the content list
                    for content_item in content:
                        if isinstance(content_item, dict):
                            # Check for different text field names
                            if "text" in content_item:
                                return content_item["text"]
                            elif content_item.get("type") == "input_text" or content_item.get("type") == "output_text":
                                return content_item.get("text", "")
                    return ""  # No text found in any content item
                else:
                    return str(content)

    return ""


def format_messages_as_context(
    messages: ResponseInputParam,
    ignore_roles: list[
        Literal["assistant", "system", "user", "function_call", "function_call_output"]
    ],
    ignore_tool_names: list[str],
) -> list[dict[str, str]]:
    """
    Formats messages as a list of individual context items, where each
    message is represented as a dictionary with source_id (index of the message in the list),
    title (role of the message + index), and content (the message content).
    This is primary used by claim verification to format input context.

    Note: Automatically ignores the last content part of the last user message
    (which is typically the actual question, not context).
    """

    # First pass: build mapping of function call IDs to names
    tool_call_names = {}
    for item in messages:
        if isinstance(item, dict):
            item_type = item.get("type", "")
            if item_type == "function_call":
                call_id = item.get("call_id", "")
                name = item.get("name", "")
                if call_id and name:
                    tool_call_names[call_id] = name

    # Second pass: process messages
    context_items = []
    for index, item in enumerate(messages):
        if isinstance(item, dict):
            role = item.get("role", "")
            item_type = item.get("type", "")

            # Handle regular messages (user, assistant, system)
            if role and role not in ignore_roles:
                content = item.get("content", "")

                # Handle content that might be a string or list
                if isinstance(content, str) and content:
                    context_items.append(
                        {
                            "source_id": str(index),
                            "title": f"{role}_{index}",
                            "content": content,
                        }
                    )
                elif isinstance(content, list):
                    # Create separate context items for each text part
                    # If this is the last user message, skip the last content part
                    content_to_process = content
                    # if index == last_user_index and len(content) > 1:
                    #    content_to_process = content[:-1]  # Exclude the last part

                    part_index = 0
                    for content_item in content_to_process:
                        if isinstance(content_item, dict):
                            text = ""
                            # Check for different text field names
                            if "text" in content_item:
                                text = content_item["text"]
                            elif content_item.get("type") == "input_text" or content_item.get("type") == "output_text":
                                text = content_item.get("text", "")

                            if text:
                                context_items.append(
                                    {
                                        "source_id": f"{index}_{part_index}",
                                        "title": f"{role}_{index}_part_{part_index}",
                                        "content": text,
                                    }
                                )
                                part_index += 1

            # Handle function calls
            elif item_type == "function_call" and "function_call" not in ignore_roles:
                call_id = item.get("call_id", "")
                name = item.get("name", "")
                arguments = item.get("arguments", "")
                if name not in ignore_tool_names and arguments:
                    context_items.append(
                        {
                            "source_id": str(index),
                            "title": f"function_call_{index}_{name}",
                            "content": f"Tool: {name}\nArguments: {arguments}",
                        }
                    )

            # Handle function call outputs
            elif (
                item_type == "function_call_output"
                and "function_call_output" not in ignore_roles
            ):
                call_id = item.get("call_id", "")
                output = item.get("output", "")
                # Check if this output's tool should be ignored
                tool_name = tool_call_names.get(call_id, "")
                if tool_name not in ignore_tool_names and output:
                    context_items.append(
                        {
                            "source_id": str(index),
                            "title": f"function_call_output_{index}_{tool_name}"
                            if tool_name
                            else f"function_call_output_{index}",
                            "content": output,
                        }
                    )

    return context_items


def convert_chat_completion_to_responses(
    chat_completion: list[ChatCompletionMessageParam],
) -> ResponseInputParam:
    """
    Converts a list of ChatCompletionMessageParam to the ResponseInputParam format focusing on the most commonly used message types.
    This is NOT a comprehensive conversion and may not handle all edge cases or message types.

    Supports:
    - System/Developer/User messages -> Message type
    - Assistant messages with content -> Message type
    - Assistant messages with tool_calls -> Message + function_call types
    - Tool messages -> function_call_output type

    Does not support:
    - Audio, images, refusals, function_call (deprecated), and other advanced features
    """

    response_items = []
    for msg in chat_completion:
        role = msg.get("role", "")

        # Handle System, Developer, and User messages
        if role in ["system", "developer", "user"]:
            content = msg.get("content", "")

            # Convert content to ResponseInputTextParam list
            content_list = []
            if isinstance(content, str):
                if content:  # Only add if not empty
                    content_list.append(
                        ResponseInputTextParam(type="input_text", text=content)
                    )
            elif isinstance(content, list):
                # Handle list of content parts
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text", "")
                        if text:
                            content_list.append(
                                ResponseInputTextParam(type="input_text", text=text)
                            )

            if content_list:  # Only add message if it has content
                response_items.append(
                    {"type": "message", "role": role, "content": content_list}
                )

        # Handle Assistant messages
        elif role == "assistant":
            # First, add the assistant message content if present
            content = msg.get("content")
            if content:
                content_list = []
                if isinstance(content, str):
                    content_list.append(
                        ResponseInputTextParam(type="input_text", text=content)
                    )
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text = part.get("text", "")
                            if text:
                                content_list.append(
                                    ResponseInputTextParam(type="input_text", text=text)
                                )

                if content_list:
                    response_items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": content_list,
                        }
                    )

            # Then, handle tool calls if present
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict):
                        call_id = tool_call.get("id", "")
                        function = tool_call.get("function", {})
                        if call_id and function:
                            response_items.append(
                                {
                                    "type": "function_call",
                                    "call_id": call_id,
                                    "name": function.get("name", ""),
                                    "arguments": function.get("arguments", ""),
                                }
                            )

        # Handle Tool messages (responses to tool calls)
        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            content = msg.get("content", "")

            # Convert content to string
            output = ""
            if isinstance(content, str):
                output = content
            elif isinstance(content, list):
                # Extract text from content parts
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text", "")
                        if text:
                            text_parts.append(text)
                output = "\n".join(text_parts)

            if tool_call_id and output:
                response_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": output,
                    }
                )

    return response_items
