# Copyright (c) Microsoft. All rights reserved.

import json
from pathlib import Path

from eval_recipes.benchmarking.semantic_test import semantic_test


def setup_sample_project(base_dir: Path) -> Path:
    """
    Create a sample project environment for testing semantic_test.

    Simulates an agent that was asked to create a simple Python calculator project.
    """
    project_dir = base_dir / "sample_project"
    project_dir.mkdir(parents=True, exist_ok=True)

    # Create a README.md
    readme = project_dir / "README.md"
    readme.write_text(
        """# Calculator Project

A simple calculator implementation in Python.

## Features
- Addition
- Subtraction
- Multiplication
- Division"""
    )

    # Create a main calculator file
    calculator = project_dir / "calculator.py"
    calculator.write_text(
        '''"""Simple calculator module."""


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
'''
    )

    # Create a test file
    test_file = project_dir / "test_calculator.py"
    test_file.write_text(
        '''"""Tests for calculator module."""

from calculator import add, divide, multiply, subtract


def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0


def test_subtract():
    assert subtract(5, 3) == 2
    assert subtract(0, 5) == -5


def test_multiply():
    assert multiply(2, 3) == 6
    assert multiply(-2, 3) == -6


def test_divide():
    assert divide(6, 2) == 3
    assert divide(5, 2) == 2.5
'''
    )

    # Create a requirements.txt
    requirements = project_dir / "requirements.txt"
    requirements.write_text("pytest>=7.0.0\n")

    return project_dir


async def test_semantic_test_calculator_project(tmp_path: Path) -> None:
    """
    Integration test: Evaluate a sample calculator project.

    This test creates a realistic scenario where an agent was asked to create
    a Python calculator project, and we use semantic_test to audit the result.
    """
    # Setup: Create sample project in tmp directory
    project_dir = setup_sample_project(tmp_path)

    # Define audit parameters
    context = """The agent was asked to create a Python calculator project with the following requirements:
- Implement basic arithmetic operations (add, subtract, multiply, divide)
- Include proper documentation and docstrings
- Write unit tests for all functions
- Create a README.md file
- Include a requirements.txt file"""

    steps = """1. List all files in the project directory
2. Read the README.md and verify it describes the project
3. Read calculator.py and check if all required operations are implemented
4. Read test_calculator.py and verify tests exist for all functions
5. Check if requirements.txt exists and contains necessary dependencies
6. Verify that functions have proper docstrings"""

    rubric = {
        "completeness": "string - are all required features present?",
        "code_quality": "string - is the code well-written with docstrings?",
        "testing": "string - are there adequate tests?",
        "documentation": "string - is the project well-documented?",
        "missing_items": "array of strings - list any missing requirements",
        "score": "number (0-100)",
    }

    await semantic_test(steps=steps, rubric=rubric, context=context, working_dir=project_dir)

    # Verify audit_output was created
    audit_output_dir = project_dir / "audit_output"
    assert audit_output_dir.exists()
    rubric_file = audit_output_dir / "rubric.json"
    assert rubric_file.exists()

    # Verify rubric file contents
    rubric_data = json.loads(rubric_file.read_text())
    print("Rubric Data:\n", rubric_data)
