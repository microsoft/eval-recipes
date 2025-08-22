# Copyright (c) Microsoft. All rights reserved.

"""
Validation script for evaluating the evaluations.
Data files are located in `data/goldset` with groundtruth labels in `data/goldset/labels.yaml`
This script runs through each file and for each evaluation and in the end outputs how the evaluation score compares to its expected score.
"""

import argparse
import asyncio
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Literal

from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.responses import ResponseInputParam
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table
import yaml

from eval_recipes.evaluate import evaluate
from eval_recipes.schemas import (
    BaseEvaluationConfig,
    ClaimVerifierConfig,
    GuidanceEvaluationConfig,
    ToolEvaluationConfig,
)

console = Console()


class ExpectedEvaluation(BaseModel):
    """Expected evaluation results from labels.yaml."""

    applicable: bool
    expected_score: float | None = None
    notes: str | None = None


class FileEvaluations(BaseModel):
    evaluations: dict[str, ExpectedEvaluation] = {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileEvaluations":
        """Create from raw dictionary data."""
        evaluations = {}
        for eval_name, eval_data in data.items():
            evaluations[eval_name] = ExpectedEvaluation(**eval_data)
        return cls(evaluations=evaluations)

    def get_evaluation_names(self) -> list[str]:
        return list(self.evaluations.keys())

    def get_evaluation(self, name: str) -> ExpectedEvaluation | None:
        return self.evaluations.get(name)


class LabelsSchema(BaseModel):
    """Complete schema for labels.yaml file."""

    files: dict[str, FileEvaluations] = {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LabelsSchema":
        files = {}
        for file_name, evaluations in data.items():
            files[file_name] = FileEvaluations.from_dict(evaluations)
        return cls(files=files)

    def get_file_evaluations(self, file_name: str) -> FileEvaluations | None:
        return self.files.get(file_name)

    def get_files(self) -> list[str]:
        return list(self.files.keys())


class ValidationResult(BaseModel):
    """Result of validating a single evaluation."""

    file: str
    evaluation: str
    status: Literal["VALID", "SKIP", "ERROR"]
    message: str | None = None

    # Applicability fields
    expected_applicable: bool | None = None
    actual_applicable: bool | None = None

    # Score fields
    expected_score: float | None = None
    actual_score: float | None = None
    score_difference: float | None = None

    # Additional context
    notes: str | None = None
    metadata: dict[str, Any] | None = None

    @property
    def applicability_match(self) -> bool | None:
        """Check if applicability matches expected."""
        if self.expected_applicable is not None and self.actual_applicable is not None:
            return self.expected_applicable == self.actual_applicable
        return None

    @property
    def score_difference_value(self) -> float | None:
        """Calculate the absolute difference between expected and actual score."""
        if self.expected_score is not None and self.actual_score is not None:
            return abs(self.expected_score - self.actual_score)
        return None

    @property
    def expected_score_str(self) -> str:
        """Get formatted string representation of expected score."""
        if self.expected_score is not None:
            return str(self.expected_score)
        elif not self.expected_applicable:
            return "N/A"
        else:
            return "not specified"


def load_test_data(
    file_path: Path,
) -> tuple[ResponseInputParam, list[ChatCompletionToolParam]]:
    """Load test data from JSON file expecting {messages: [...], tools: [...]} structure."""
    with Path.open(file_path) as f:
        data = json.load(f)

    messages: ResponseInputParam = data["messages"]
    tools: list[ChatCompletionToolParam] = data.get("tools", [])
    return messages, tools


def load_labels(labels_path: Path) -> LabelsSchema:
    """Load expected evaluation results from labels.yaml."""
    with Path.open(labels_path) as f:
        raw_data = yaml.safe_load(f)
    return LabelsSchema.from_dict(raw_data)


async def validate_single_file(
    file_path: Path, labels: LabelsSchema, evaluation_filter: list[str] | None = None
) -> list[ValidationResult]:
    """Validate a single test file against expected results for multiple evaluations.

    Args:
        file_path: Path to the test file
        labels: Expected evaluation results
        evaluation_filter: Optional list of evaluation names to run (if None, runs all)
    """

    file_name = file_path.name
    file_evaluations = labels.get_file_evaluations(file_name)

    if not file_evaluations:
        return [
            ValidationResult(
                file=file_name,
                evaluation="all",
                status="SKIP",
                message="No labels defined for this file",
            )
        ]

    evaluation_names = file_evaluations.get_evaluation_names()

    # Apply evaluation filter if specified
    if evaluation_filter:
        evaluation_names = [name for name in evaluation_names if name in evaluation_filter]
        if not evaluation_names:
            return [
                ValidationResult(
                    file=file_name,
                    evaluation="all",
                    status="SKIP",
                    message=f"No matching evaluations found for filter: {evaluation_filter}",
                )
            ]

    console.print(f"[cyan]Testing {file_name} for {len(evaluation_names)} evaluation(s)...[/cyan]")

    evaluation_configs = {}
    for eval_name in evaluation_names:
        if eval_name == "claim_verification":
            evaluation_configs[eval_name] = ClaimVerifierConfig(provider="openai", max_concurrency=10)
        elif eval_name == "tool_usage":
            evaluation_configs[eval_name] = ToolEvaluationConfig(provider="azure_openai")
        elif eval_name == "guidance":
            evaluation_configs[eval_name] = GuidanceEvaluationConfig(provider="azure_openai")
        else:  # preference_adherence and any others
            evaluation_configs[eval_name] = BaseEvaluationConfig(provider="openai")

    messages, tools = load_test_data(file_path)
    results = await evaluate(
        messages=messages,
        tools=tools,
        evaluations=evaluation_names,
        evaluation_configs=evaluation_configs,
        max_concurrency=4,
    )

    if not results:
        return [
            ValidationResult(
                file=file_name,
                evaluation="all",
                status="ERROR",
                message="No evaluation results returned",
            )
        ]

    validation_results = []
    for result in results:
        eval_name = result.eval_name
        expected = file_evaluations.get_evaluation(eval_name)

        if not expected:
            validation_results.append(
                ValidationResult(
                    file=file_name,
                    evaluation=eval_name,
                    status="SKIP",
                    message=f"No {eval_name} labels defined",
                )
            )
            continue

        validation_result = ValidationResult(
            file=file_name,
            evaluation=eval_name,
            status="SKIP",
            expected_applicable=expected.applicable,
            actual_applicable=result.applicable,
            expected_score=expected.expected_score,
            actual_score=result.score,
            notes=expected.notes,
            metadata=result.metadata,
        )

        validation_result.score_difference = validation_result.score_difference_value

        # Check applicability match
        applicability_match = validation_result.applicability_match
        if applicability_match:
            validation_result.status = "VALID"
        else:
            validation_result.status = "ERROR"
            validation_result.message = (
                f"Applicability mismatch: expected {expected.applicable}, got {result.applicable}"
            )

        validation_results.append(validation_result)
    return validation_results


def calculate_f1_score(expected_applicable: list[bool], actual_applicable: list[bool]) -> float:
    """Calculate F1 score for applicability as binary classification."""
    if not expected_applicable or not actual_applicable:
        return 0.0

    # Convert to binary: True=1, False=0
    true_positive = sum(1 for exp, act in zip(expected_applicable, actual_applicable, strict=False) if exp and act)
    false_positive = sum(1 for exp, act in zip(expected_applicable, actual_applicable, strict=False) if not exp and act)
    false_negative = sum(1 for exp, act in zip(expected_applicable, actual_applicable, strict=False) if exp and not act)

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


def display_results_summary(results: list[ValidationResult]) -> None:
    """Display a simple summary table with Evaluation | Avg Score | N/A F1."""

    # Group results by evaluation type
    eval_data: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "score_differences": [],
            "expected_applicable": [],
            "actual_applicable": [],
        }
    )

    for result in results:
        if result.status in ["VALID", "ERROR"]:
            eval_data[result.evaluation]["expected_applicable"].append(result.expected_applicable)
            eval_data[result.evaluation]["actual_applicable"].append(result.actual_applicable)

            # Only collect score differences for applicable evaluations with scores
            if result.expected_applicable and result.actual_applicable and result.score_difference is not None:
                eval_data[result.evaluation]["score_differences"].append(result.score_difference)

    table = Table(
        title="Evaluation Validation Summary",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Evaluation", style="cyan", width=25)
    table.add_column("Avg |Delta|", justify="right", style="yellow")
    table.add_column("N/A F1", justify="right", style="green")

    all_score_differences: list[float] = []
    all_expected_applicable: list[bool] = []
    all_actual_applicable: list[bool] = []
    for eval_name in sorted(eval_data.keys()):
        data = eval_data[eval_name]
        score_diffs = data["score_differences"]
        expected_app = data["expected_applicable"]
        actual_app = data["actual_applicable"]

        # Calculate average score difference
        avg_score = sum(score_diffs) / len(score_diffs) if score_diffs else None
        avg_score_str = f"{avg_score:.1f}" if avg_score is not None else "N/A"

        # Calculate F1 score for applicability
        f1_score = calculate_f1_score(expected_app, actual_app) * 100
        f1_str = f"{f1_score:.2f}"

        table.add_row(eval_name, avg_score_str, f1_str)

        if score_diffs:
            all_score_differences.extend(score_diffs)
        all_expected_applicable.extend(expected_app)
        all_actual_applicable.extend(actual_app)

    overall_avg_score = sum(all_score_differences) / len(all_score_differences) if all_score_differences else None
    overall_avg_str = f"{overall_avg_score:.1f}" if overall_avg_score is not None else "N/A"
    overall_f1 = calculate_f1_score(all_expected_applicable, all_actual_applicable)
    overall_f1_str = f"{overall_f1:.3f}"

    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{overall_avg_str}[/bold]",
        f"[bold]{overall_f1_str}[/bold]",
        style="bright_white on grey23",
    )

    console.print("\n")
    console.print(table)

    if overall_avg_score is not None:
        console.print(
            f"\n[bold green]Final Score (Average |Expected - Actual|): {overall_avg_score:.2f}[/bold green] [dim](Lower scores are better, 0 = perfect match)[/dim]"
        )
        console.print("")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation",
        "-e",
        action="append",
        choices=[
            "claim_verification",
            "tool_usage",
            "guidance",
            "preference_adherence",
        ],
        default=None,
        help="Specific evaluation(s) to run. Can be specified multiple times. If not specified, runs all evaluations.",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default=None,
        help="Specific test file name to run (e.g., guidance_test_1.json). If not specified, runs all test files.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    goldset_dir = Path(__file__).parents[1] / "data" / "goldset"
    labels_path = goldset_dir / "labels.yaml"

    labels = load_labels(labels_path)

    # Determine which files to test
    if args.file:
        test_file_path = goldset_dir / args.file
        if not test_file_path.exists():
            console.print(f"[red]Error: File {args.file} not found in {goldset_dir}[/red]")
            return
        test_files = [test_file_path]
    else:
        test_files = list(goldset_dir.glob("*.json"))

    console.print(f"[green]Found {len(test_files)} test file(s)[/green]")

    all_results: list[ValidationResult] = []
    for test_file in test_files:
        file_results = await validate_single_file(test_file, labels, evaluation_filter=args.evaluation)
        all_results.extend(file_results)

    display_results_summary(all_results)


if __name__ == "__main__":
    asyncio.run(main())
