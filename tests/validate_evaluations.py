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

from rich.console import Console
from rich.table import Table
import yaml

from eval_recipes.evaluate import evaluate
from eval_recipes.evaluations.claim_verification.claim_verification_evaluator import ClaimVerificationEvaluatorConfig
from eval_recipes.evaluations.guidance.guidance_evaluator import GuidanceEvaluatorConfig
from eval_recipes.evaluations.tool_usage.tool_usage_evaluator import ToolUsageEvaluatorConfig
from eval_recipes.schemas import BaseEvaluatorConfig

console = Console()


async def validate_single_file(file_path: Path, labels: dict, evaluation_filter: list[str] | None = None) -> list:
    """Validate a single test file against expected results."""
    file_name = file_path.name
    file_evaluations = labels.get(file_name, {})
    if not file_evaluations:
        return []

    evaluation_names = list(file_evaluations.keys())
    if evaluation_filter:
        evaluation_names = [name for name in evaluation_names if name in evaluation_filter]
    console.print(f"[cyan]Testing {file_name} for {len(evaluation_names)} evaluation(s)...[/cyan]")

    configs = {
        "claim_verification": ClaimVerificationEvaluatorConfig(provider="openai", max_concurrency=10),
        "tool_usage": ToolUsageEvaluatorConfig(provider="openai"),
        "guidance": GuidanceEvaluatorConfig(provider="openai"),
    }
    evaluation_configs = {name: configs.get(name, BaseEvaluatorConfig(provider="openai")) for name in evaluation_names}

    with Path.open(file_path) as f:
        data = json.load(f)

    results = await evaluate(
        messages=data["messages"],
        tools=data.get("tools", []),
        evaluations=evaluation_names,
        evaluation_configs=evaluation_configs,
        max_concurrency=4,
    )

    validation_results = []
    for result in results:
        expected = file_evaluations.get(result.eval_name, {})
        expected_score = expected.get("expected_score")
        actual_score = result.score
        score_diff = (
            abs(expected_score - actual_score)
            if expected_score and actual_score and result.applicable and expected.get("applicable")
            else None
        )
        validation_results.append(
            {
                "file": file_name,
                "evaluation": result.eval_name,
                "expected_applicable": expected.get("applicable"),
                "actual_applicable": result.applicable,
                "score_difference": score_diff,
                "match": expected.get("applicable") == result.applicable,
            }
        )
    return validation_results


def display_results_summary(results: list) -> None:
    """Display summary table with Evaluation | Avg Score | N/A F1."""
    if not results:
        return

    # Group by evaluation
    eval_data = defaultdict(lambda: {"scores": [], "expected": [], "actual": []})
    for r in results:
        eval = r["evaluation"]
        if r["score_difference"] is not None:
            eval_data[eval]["scores"].append(r["score_difference"])
        eval_data[eval]["expected"].append(r["expected_applicable"])
        eval_data[eval]["actual"].append(r["actual_applicable"])

    # Create table
    table = Table(title="Evaluation Validation Summary", show_header=True, header_style="bold magenta")
    table.add_column("Evaluation", style="cyan", width=25)
    table.add_column("Avg |Delta|", justify="right", style="yellow")
    table.add_column("N/A F1", justify="right", style="green")

    all_scores, all_exp, all_act = [], [], []
    for eval_name in sorted(eval_data.keys()):
        data = eval_data[eval_name]
        avg_score = sum(data["scores"]) / len(data["scores"]) if data["scores"] else None

        # Calculate F1
        tp = sum(1 for e, a in zip(data["expected"], data["actual"], strict=False) if e and a)
        fp = sum(1 for e, a in zip(data["expected"], data["actual"], strict=False) if not e and a)
        fn = sum(1 for e, a in zip(data["expected"], data["actual"], strict=False) if e and not a)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        table.add_row(eval_name, f"{avg_score:.1f}" if avg_score else "N/A", f"{f1 * 100:.2f}")

        all_scores.extend(data["scores"])
        all_exp.extend(data["expected"])
        all_act.extend(data["actual"])

    # Overall stats
    overall_avg = sum(all_scores) / len(all_scores) if all_scores else None
    tp = sum(1 for e, a in zip(all_exp, all_act, strict=False) if e and a)
    fp = sum(1 for e, a in zip(all_exp, all_act, strict=False) if not e and a)
    fn = sum(1 for e, a in zip(all_exp, all_act, strict=False) if e and not a)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    overall_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    overall_avg_str = f"{overall_avg:.1f}" if overall_avg else "N/A"
    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{overall_avg_str}[/bold]",
        f"[bold]{overall_f1:.3f}[/bold]",
        style="bright_white on grey23",
    )

    console.print("\n", table)
    if overall_avg:
        console.print(f"\n[bold green]Final Score: {overall_avg:.2f}[/bold green] [dim](Lower = better)[/dim]\n")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--evaluation",
        action="append",
        choices=["claim_verification", "tool_usage", "guidance", "preference_adherence"],
        default=[],
    )
    parser.add_argument("-f", "--file", type=str, action="append", default=[])
    args = parser.parse_args()

    goldset_dir = Path(__file__).parents[1] / "data" / "goldset"
    with Path.open(goldset_dir / "labels.yaml") as f:
        labels = yaml.safe_load(f)

    test_files = [goldset_dir / f for f in args.file] if args.file else list(goldset_dir.glob("*.json"))
    console.print(f"[green]Found {len(test_files)} test file(s)[/green]")

    all_results = []
    for test_file in test_files:
        if test_file.exists():
            results = await validate_single_file(test_file, labels, args.evaluation)
            all_results.extend(results)

    display_results_summary(all_results)


if __name__ == "__main__":
    asyncio.run(main())
