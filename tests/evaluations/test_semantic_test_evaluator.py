# Copyright (c) Microsoft. All rights reserved.

from pathlib import Path

import pytest

from eval_recipes.evaluations.semantic_test.semantic_test_evaluator import (
    SemanticTestEvaluator,
    SemanticTestEvaluatorConfig,
)


@pytest.mark.skip(reason="Time")
async def test_semantic_test_evaluator_fibonacci(tmp_path: Path) -> None:
    """Test that SemanticTestEvaluator can correctly audit a simple code implementation task."""
    # Setup: Create fibonacci.py with recursive implementation
    fibonacci_py = tmp_path / "fibonacci.py"
    fibonacci_py.write_text(
        """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Generate first 10 fibonacci numbers
with open("fibonacci.txt", "w") as f:
    for i in range(10):
        f.write(f"{fibonacci(i)}\\n")
"""
    )

    # Create fibonacci.txt with first 10 fibonacci numbers
    fibonacci_txt = tmp_path / "fibonacci.txt"
    fibonacci_txt.write_text("0\n1\n1\n2\n3\n5\n8\n13\n21\n34\n")

    context = "Create a fibonacci.py script that implements the fibonacci sequence using recursion and writes the first 10 numbers to fibonacci.txt"

    steps = """1. Check if fibonacci.py exists
2. Read fibonacci.py and verify it implements fibonacci using recursion (look for a recursive function call)
3. Check if fibonacci.txt exists
4. Read fibonacci.txt and verify it contains the first 10 fibonacci numbers (0, 1, 1, 2, 3, 5, 8, 13, 21, 34)"""

    rubric = {
        "fibonacci_file_exists": "10 points boolean - does fibonacci.py exist?",
        "uses_recursion": "30 points boolean - does the implementation use recursion?",
        "output_file_exists": "10 points boolean - does fibonacci.txt exist?",
        "output_correct": "50 points boolean - does fibonacci.txt contain the first 10 fibonacci numbers?",
        "score": "number (0-100) - 100 if all checks pass, 0 otherwise",
    }

    config = SemanticTestEvaluatorConfig(
        working_dir=tmp_path,
        steps=steps,
        rubric=rubric,
        context=context,
    )

    evaluator = SemanticTestEvaluator(config=config)
    result = await evaluator.evaluate(messages=[], tools=[])
    assert result.eval_name == "semantic_test"
    assert result.applicable is True
    assert isinstance(result.score, float)
    assert result.metadata is not None
