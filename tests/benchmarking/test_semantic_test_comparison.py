# Copyright (c) Microsoft. All rights reserved.

from pathlib import Path

import pytest

from eval_recipes.benchmarking.semantic_test_comparison import semantic_test_comparison


def setup_comparison_projects(base_dir: Path) -> list[Path]:
    """Create multiple agent project directories for comparison testing."""
    # Agent 1: Good implementation with docs
    agent1_dir = base_dir / "agent1_project"
    agent1_dir.mkdir(parents=True)
    (agent1_dir / "solution.py").write_text(
        '''"""Solution module with proper documentation."""


def solve(x: float) -> float:
    """Double the input value.

    Args:
        x: The input number to double

    Returns:
        The doubled value
    """
    return x * 2
'''
    )
    (agent1_dir / "README.md").write_text(
        """# Solution

A complete solution that doubles the input value.

## Usage

```python
from solution import solve
result = solve(5)  # Returns 10
```
"""
    )

    # Agent 2: Incomplete/incorrect implementation
    agent2_dir = base_dir / "agent2_project"
    agent2_dir.mkdir(parents=True)
    (agent2_dir / "solution.py").write_text("def solve(x): return x")
    # No README, no docstrings, wrong implementation

    # Agent 3: Medium quality - correct but minimal docs
    agent3_dir = base_dir / "agent3_project"
    agent3_dir.mkdir(parents=True)
    (agent3_dir / "solution.py").write_text(
        """def solve(x):
    return x * 2
"""
    )
    (agent3_dir / "README.md").write_text("# Solution\n\nDoubles the input.")

    return [agent1_dir, agent2_dir, agent3_dir]


@pytest.mark.skip(reason="requires claude code installation")
async def test_semantic_test_comparison(tmp_path: Path) -> None:
    """Integration test for comparing multiple agent outputs."""
    directories = setup_comparison_projects(tmp_path)

    original_task = """Create a Python function called 'solve' that doubles its input.
Include proper documentation and a README.md file."""

    guidelines = """When evaluating these solutions, consider:
- Code correctness (does it actually double the input?)
- Documentation quality
- Code style and readability"""

    result = await semantic_test_comparison(
        original_task=original_task,
        directories=directories,
        guidelines=guidelines,
    )

    # Verify result structure
    assert isinstance(result.reasoning, str)
    assert len(result.reasoning) > 0
    assert isinstance(result.rankings, list)
    assert len(result.rankings) == 3
    assert set(result.rankings) == {0, 1, 2}  # All indices present

    print(f"Reasoning: {result.reasoning}")
    print(f"Rankings: {result.rankings}")
