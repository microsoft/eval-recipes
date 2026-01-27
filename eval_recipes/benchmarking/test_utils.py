# Copyright (c) Microsoft. All rights reserved.

"""
Shared utilities for benchmark test scripts.

This module provides common functionality used across all test.py files to ensure
consistency and reduce code duplication.
"""

import json
import os
from pathlib import Path


def write_test_result(output_dir: Path, test_id: str, score: float, metadata: dict) -> None:
    """
    Write test results to the standard result file format.

    Args:
        output_dir: Directory where the result file should be written
        test_id: Unique test ID (typically from EVAL_RECIPES_TEST_ID env var)
        score: Test score (will be clamped to 0-100 range)
        metadata: Additional metadata about the test results

    The result file will be written to: {output_dir}/.eval_recipes_test_results_{test_id}.json
    """
    # Clamp score to valid range
    score = max(0.0, min(100.0, float(score)))

    result_file = output_dir / f".eval_recipes_test_results_{test_id}.json"
    result = {"score": score, "metadata": metadata}
    result_file.write_text(json.dumps(result, indent=2), encoding="utf-8")


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein distance (edit distance) between two strings.

    The Levenshtein distance is the minimum number of single-character edits
    (insertions, deletions, or substitutions) required to change one string into another.

    Args:
        s1: First string
        s2: Second string

    Returns:
        The edit distance as an integer

    Examples:
        >>> levenshtein_distance("kitten", "sitting")
        3
        >>> levenshtein_distance("hello", "hello")
        0
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def load_text_from_path_or_content(text: str) -> str:
    """
    Load text from either a file path or treat as direct content.

    This utility supports the pattern where CLI arguments can be either:
    1. A path to a file (if the path exists)
    2. Direct content string (if the path doesn't exist or is invalid)

    This allows tests to be run both in containers (with default file paths)
    and locally/debugger with direct content strings for quick testing.

    Args:
        text: Either a file path or direct text content

    Returns:
        The text content as a string

    Examples:
        >>> # If file exists, reads from file
        >>> content = load_text_from_path_or_content("/path/to/file.txt")
        >>> # If file doesn't exist, treats as direct content
        >>> content = load_text_from_path_or_content("This is my content")
    """
    try:
        text_path = Path(text)
        if text_path.exists():
            return text_path.read_text()
        else:
            # Path doesn't exist, treat as direct content
            return text
    except (OSError, ValueError):
        # Invalid path (e.g., too long or invalid characters), treat as direct content
        return text


def get_test_id_from_env_or_default(default: str = "dev") -> str:
    """
    Get test ID from environment variable or use default.

    Args:
        default: Default test ID if environment variable is not set

    Returns:
        Test ID string
    """
    return os.environ.get("EVAL_RECIPES_TEST_ID", default)


def get_agent_log_hint(metadata_path: Path | None = None) -> str | None:
    """
    Read agent_log_hint from the agent metadata file.

    The harness writes agent metadata to /project/.agent_metadata.json in the container.
    This function reads the agent_log_hint field, which can be passed to semantic_test()
    to provide the auditing agent with information about where to find the agent's logs.

    Args:
        metadata_path: Path to metadata file. Defaults to /project/.agent_metadata.json

    Returns:
        The agent_log_hint string if present, None otherwise.

    Examples:
        >>> # In container (default behavior)
        >>> hint = get_agent_log_hint()  # Reads /project/.agent_metadata.json
        >>> # Custom path
        >>> hint = get_agent_log_hint(Path("/custom/path/.agent_metadata.json"))
    """
    if metadata_path is None:
        metadata_path = Path("/project/.agent_metadata.json")

    if not metadata_path.exists():
        return None

    try:
        with metadata_path.open(encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata.get("agent_log_hint")
    except (json.JSONDecodeError, OSError):
        return None


def get_instructions_from_file_or_default(default: str = "", instructions_file: Path | None = None) -> str:
    """
    Get task instructions from file or use default.

    The harness writes instructions to /project/instructions.txt in the container.
    For local testing, you can pass a custom path or provide instructions via CLI.

    Args:
        default: Default instructions if file doesn't exist
        instructions_file: Optional custom path to instructions file.
                          If None, defaults to ./instructions.txt (current directory)

    Returns:
        Instructions string

    Examples:
        >>> # In container (default behavior)
        >>> instructions = get_instructions_from_file_or_default()  # Reads ./instructions.txt
        >>> # Local testing with custom file
        >>> instructions = get_instructions_from_file_or_default(instructions_file=Path("my_instructions.txt"))
        >>> # Local testing with default fallback
        >>> instructions = get_instructions_from_file_or_default(default="Fallback instructions")
    """
    if instructions_file is None:
        # Default to instructions.txt in current directory (where test.py runs)
        instructions_file = Path("instructions.txt")

    if instructions_file.exists():
        return instructions_file.read_text()

    return default
