# Copyright (c) Microsoft. All rights reserved.

from typing import Literal

from liquid import render

from eval_recipes.benchmarking.schemas import AggregateReport, ComparisonRunResult, FinalAggregateReport
from eval_recipes.utils.llm import create_client

SYSTEM_PROMPT = """\
You are an expert evaluator analyzing AI agent comparison results.
Your task is to synthesize multiple comparison judgments into a clear summary that
explains WHY agents were ranked the way they were.

Be factual and accurate. Only report patterns clearly supported by the comparison
reasoning provided. Do not speculate or add information not present in the comparisons."""

USER_PROMPT_TEMPLATE = """\
## Original Task
{{ task_instructions }}

## Comparison Results ({{ comparison_count }} runs)
{% for result in results %}
### Comparison Run {{ result.run_num }}
**Rankings (best to worst):**
{% for agent in result.ranked_agents %}
{{ forloop.index }}. {{ agent }}
{% endfor %}

**Reasoning:**
{{ result.reasoning }}

---
{% endfor %}

## Your Analysis Task
Based on the {{ comparison_count }} comparison runs above, provide a concise summary \
explaining WHY the rankings were what they were.

Output a bulleted list of 5-10 items, each about one sentence. Focus on:
- Why one agent consistently won (if applicable)
- Key factors that separated the agents
- Areas of agreement or disagreement between comparison runs

Be specific and cite evidence from the reasoning provided. Do not fabricate details."""


async def generate_aggregate_report(
    task_name: str,
    task_instructions: str,
    comparison_results: list[ComparisonRunResult],
    provider: Literal["openai", "azure_openai"] = "openai",
    model: str = "gpt-5.2",
) -> AggregateReport:
    """Generate aggregate report from comparison results using LLM.

    Args:
        task_name: Name of the task being analyzed.
        task_instructions: Original task instructions.
        comparison_results: List of comparison run results to analyze.
        provider: LLM provider to use.
        model: Model to use for analysis.

    Returns:
        AggregateReport with synthesized analysis of comparison results.
    """
    if not comparison_results:
        return AggregateReport(
            task_name=task_name,
            agent_names=[],
            analysis="No comparison results provided.",
        )

    # Build context for each comparison run (de-anonymized)
    results_context = []
    for result in comparison_results:
        # Map rankings to agent names (rankings are indices into agent_names)
        ranked_agents = [result.agent_names[i] for i in result.result.rankings]
        results_context.append(
            {
                "run_num": result.comparison_run_num,
                "ranked_agents": ranked_agents,
                "reasoning": result.result.reasoning,
            }
        )

    user_prompt = render(
        USER_PROMPT_TEMPLATE,
        task_instructions=task_instructions,
        comparison_count=len(comparison_results),
        results=results_context,
    )

    messages: list = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    agent_names = comparison_results[0].agent_names

    async with create_client(provider=provider) as client:
        response = await client.responses.create(
            model=model,
            input=messages,
            store=False,
        )

    analysis = response.output_text or "Failed to generate analysis."

    return AggregateReport(
        task_name=task_name,
        agent_names=agent_names,
        analysis=analysis,
    )


FINAL_SYSTEM_PROMPT = """\
You are an expert evaluator synthesizing AI agent comparison results across multiple tasks.
Your task is to explain WHY the overall scores and rankings were what they were, based on
the individual task analyses provided.

Be factual and accurate. Only report patterns clearly supported by the task analyses.
Do not speculate or add information not present in the context."""

FINAL_USER_PROMPT_TEMPLATE = """\
## Overall Results
{% for agent in agents %}
**{{ agent.name }}**: Average Rank {{ agent.avg_rank | round: 2 }}, Win Rate {{ agent.win_rate | round: 0 }}%
{% endfor %}

## Per-Task Summaries ({{ task_count }} tasks)
{% for task in tasks %}
### {{ task.name }}
**Instructions:** {{ task.instructions }}

**Rankings:** {{ task.rankings }}

**Analysis:**
{{ task.analysis }}

---
{% endfor %}

## Your Task
Based on the {{ task_count }} task analyses above, provide a final summary explaining \
WHY the overall scores were what they were.

Write at most one paragraph followed by a bulleted list of key insights. Focus on:
- The primary factors that drove the overall rankings
- Patterns that emerged across multiple tasks
- Key strengths or weaknesses that consistently appeared

Be specific and cite evidence from the task analyses. Do not fabricate details."""


async def generate_final_aggregate_report(
    agent_names: list[str],
    overall_avg_rank: dict[str, float],
    overall_win_rate: dict[str, float],
    task_summaries: list[dict],
    provider: Literal["openai", "azure_openai"] = "openai",
    model: str = "gpt-5.2",
) -> FinalAggregateReport:
    """Generate final aggregate report synthesizing all task analyses.

    Args:
        agent_names: List of agent names being compared.
        overall_avg_rank: Average rank per agent across all tasks.
        overall_win_rate: Win rate per agent across all tasks.
        task_summaries: List of dicts with keys: name, instructions, rankings, analysis.
        provider: LLM provider to use.
        model: Model to use for analysis.

    Returns:
        FinalAggregateReport with synthesized analysis across all tasks.
    """
    if not task_summaries:
        return FinalAggregateReport(
            agent_names=agent_names,
            analysis="No task summaries provided.",
        )

    # Build agent context sorted by avg rank (best first)
    agents_context = []
    for agent in sorted(agent_names, key=lambda a: overall_avg_rank.get(a, float("inf"))):
        agents_context.append(
            {
                "name": agent,
                "avg_rank": overall_avg_rank.get(agent, 0),
                "win_rate": overall_win_rate.get(agent, 0),
            }
        )

    user_prompt = render(
        FINAL_USER_PROMPT_TEMPLATE,
        agents=agents_context,
        tasks=task_summaries,
        task_count=len(task_summaries),
    )

    messages: list = [
        {"role": "system", "content": FINAL_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    async with create_client(provider=provider) as client:
        response = await client.responses.create(
            model=model,
            input=messages,
            store=False,
        )

    analysis = response.output_text or "Failed to generate analysis."

    return FinalAggregateReport(
        agent_names=agent_names,
        analysis=analysis,
    )
