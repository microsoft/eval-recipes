# Copyright (c) Microsoft. All rights reserved.

import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import re

    import marimo as mo
    from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
    from openai.types.responses import ResponseInputParam

    from eval_recipes.evaluate import evaluate
    from eval_recipes.evaluations.check_criteria.check_criteria_evaluator import CheckCriteriaEvaluatorConfig
    from eval_recipes.schemas import BaseEvaluatorConfig, EvaluationOutput
    return (
        BaseEvaluatorConfig,
        ChatCompletionToolParam,
        CheckCriteriaEvaluatorConfig,
        EvaluationOutput,
        ResponseInputParam,
        evaluate,
        mo,
        re,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Extensibility

    This notebook provides an overview of how you can customize the package to your needs:

    ## Custom Evaluations

    Define your own evaluations using the `EvaluatorProtocol`.

    You evaluation must implement this protocol:
    ```python
    class EvaluatorProtocol(Protocol):
        \"""Protocol for custom evaluator classes.\"""

        def __init__(self, config: BaseEvaluationConfig | None = None) -> None:
            \"""Initialize the evaluator with an optional configuration.
            If config is not provided, it should be instantiated with defaults.
            \"""
            ...

        async def evaluate(self, messages: ResponseInputParam, tools: list[ChatCompletionToolParam]) -> EvaluationOutput:
            \"""Evaluate messages and tools, returning an EvaluationOutput.\"""
            ...
    ```

    Then you can pass it to `evaluate` API like any built-in evaluation.

    ```python
    result = await evaluate(
        messages=messages,
        tools=[],
        evaluations=[MyCustomEvaluation],
        evaluation_configs={"my_custom_evaluation": my_custom_config},
    )
    ```

    ## Custom Criteria Evaluation

    Use the `CheckCriteriaEvaluator` which let's you provide your own criteria and rubrics. Think of this as a catch all for any straighforward criterion or rubrics you want to evaluate against.

    ```python
    criteria_config = CheckCriteriaEvaluationConfig(
        criteria=[
            "The response contains no emojis.",
            "The response is largely in paragraph form, rather than using excessive headings and bulleted lists.",
        ],
    )

    result = await evaluate(
        messages=messages,
        tools=[],
        evaluations=["check_criteria"],
        evaluation_configs={"check_criteria": criteria_config},
        max_concurrency=1,
    )
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Creating a Custom Evaluator

    Let's look at creating a deterministic evaluation for AI-generated responses. One common issue is that LLMs will generate syntax that is not renderable or undesirable. This evaluator will check for the following:

    * em-dashes
    * headings greater than level 3 (####, #####, and so on)
    """
    )
    return


@app.cell
def _(
    BaseEvaluatorConfig,
    ChatCompletionToolParam,
    EvaluationOutput,
    ResponseInputParam,
    re,
):
    class SyntaxEvaluator:
        """Custom evaluator that checks for undesirable syntax patterns."""

        def __init__(self, config: BaseEvaluatorConfig | None = None) -> None:
            self.config = config or BaseEvaluatorConfig()

        async def evaluate(self, messages: ResponseInputParam, tools: list[ChatCompletionToolParam]) -> EvaluationOutput:
            # Extract the last assistant message
            assistant_message = ""
            for message in reversed(messages):
                if "role" in message and message["role"] == "assistant":
                    if "content" in message and message["content"]:
                        assistant_message = str(message["content"])
                        break

            issues = []
            # Check for em-dashes (—)
            em_dash_count = assistant_message.count("—")
            if em_dash_count > 0:
                issues.append(f"Found {em_dash_count} em-dash(es)")

            # Check for headings greater than level 3 (####, #####, etc)
            excessive_heading_pattern = r"^#{4,}\s"
            excessive_headings = re.findall(excessive_heading_pattern, assistant_message, re.MULTILINE)
            if excessive_headings:
                issues.append(f"Found {len(excessive_headings)} heading(s) with level > 4")

            # If any issues are found, score is 0
            feedback = None
            score = 100
            if issues:
                feedback = f"Syntax issues detected: {'; '.join(issues)}"
                score = 0

            return EvaluationOutput(
                eval_name="markdown_syntax",
                applicable=True,
                score=score,
                feedback=feedback,
                metadata={
                    "em_dash_count": em_dash_count,
                    "excessive_heading_count": len(excessive_headings),
                    "issues": issues,
                },
            )
    return (SyntaxEvaluator,)


@app.cell
def _(mo):
    mo.md(r"""# Creating a Custom Criteria Evaluation""")
    return


@app.cell
async def _(
    CheckCriteriaEvaluatorConfig,
    SyntaxEvaluator,
    evaluate,
    messages: "ResponseInputParam",
):
    criteria_config = CheckCriteriaEvaluatorConfig(
        provider="openai",
        model="gpt-5-mini",
        criteria=[
            "The response contains no emojis.",
            "The response is largely in paragraph form, rather than using excessive headings and bulleted lists.",
            "The user's request was fully addressed in terms of technical detail.",
        ],
        passed_threshold=65,
        max_concurrency=3,
    )

    result = await evaluate(
        messages=messages,
        tools=[],
        evaluations=[SyntaxEvaluator, "check_criteria"],
        evaluation_configs={"check_criteria": criteria_config},
        max_concurrency=1,
    )
    return (result,)


@app.cell(hide_code=True)
def _(ResponseInputParam, mo):
    messages: ResponseInputParam = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "How do RSS feeds work? How do you typically discover them?",
        },
        {
            "role": "assistant",
            "content": """RSS feeds are a way to keep up with updates from websites without having to visit them manually. Here's a breakdown:

    ## How RSS Feeds Work

    **Format:** RSS (Really Simple Syndication) is an XML-based format. It contains structured data about new content—like headlines, article summaries, publication dates, and links back to the full content on the site.

    **Delivery:** Websites publish an RSS feed that updates whenever new content is added (e.g., a blog post, podcast episode, or news article).

    **Consumption:** You use an RSS reader (also called a feed aggregator) to subscribe to feeds. The reader regularly checks the feed's XML file and displays updates in one place, so you don't need to check multiple sites.

    **Automation:** Most readers let you mark items as read, filter content by keywords, and sometimes even push notifications when new entries appear.

    ## How to Discover RSS Feeds

    ### Look for the RSS Icon:
    Many sites still display the familiar orange RSS symbol (📡). Clicking it usually takes you to the feed URL.

    ### Check the Page Source:
    Even if not visibly advertised, sites often include `<link rel="alternate" type="application/rss+xml">` in their HTML header. You can find it by viewing the source code.

    ### RSS Reader Search:
    Some feed readers (like Feedly, Inoreader, or NewsBlur) let you search by website name or keyword and will find the feed automatically if it exists.

    ### Guess the URL:
    Common patterns include:
    - `https://example.com/feed/`
    - `https://example.com/rss/`
    - `https://example.com/category/feed/`

    ### Browser Extensions & Tools:
    Tools like RSS Finder or Get RSS Feed URL can automatically detect feeds on any page.

    ### Third-party Aggregators:
    For sites that don't have RSS feeds, services like RSS.app or Feedity can generate one by scraping updates.

    👉 Do you want me to also recommend some modern RSS readers (apps/sites) that make discovery and subscribing much easier?""",
        },
        {
            "role": "user",
            "content": "How do RSS apps handle updates to the feed technically? Do they poll? Do they compare the diff?",
        },
        {
            "role": "assistant",
            "content": """Great question — let's go a layer deeper into how RSS readers actually handle updates.

    # How RSS Apps Handle Updates

    ## 1. Polling Feeds

    **Traditional method:** Most RSS apps periodically poll each subscribed feed's URL (the XML file) on a schedule, e.g. every 15 minutes, hourly, or daily.

    They request the feed with an HTTP GET and parse the XML response for new `<item>` (RSS) or `<entry>` (Atom) elements.

    **Scaling:** For apps with thousands of users, this means a lot of network requests. To reduce server load and bandwidth, many apps use conditional HTTP headers:
    - **If-Modified-Since:** The server only responds with new content if the feed has changed since the given date.
    - **ETag:** A unique identifier for the current version of the feed; if unchanged, the server can respond with `304 Not Modified` (saving bandwidth).

    ## 2. Change Detection

    Once a new feed file is retrieved, the app compares entries with what it already has stored:

    #### GUID / ID comparison:
    Each item usually has a unique ID (`<guid>` in RSS, `<id>` in Atom). Readers store these IDs and skip duplicates.

    #### Fallback diffing:
    If the feed doesn't use GUIDs consistently, readers may compare URLs, titles, or publication dates to decide if an entry is new.

    ##### Content changes:
    Some feeds silently update an existing item (e.g., typo fix). Most readers don't diff content deeply—they just replace the old entry with the new one if the ID matches.

    ## 3. Push-based Alternatives

    Polling can be inefficient, so newer protocols help:

    **PubSubHubbub (now called WebSub):** A site can declare a "hub" in its feed. Instead of clients polling, the hub pushes updates to subscribers in near real time.

    **Webhooks / Notifications:** Some modern services (like podcast hosts) offer custom push endpoints for faster updates.

    ## 4. Caching & Backoff Strategies

    Readers avoid hammering servers by:
    - Respecting the feed's TTL (time-to-live) hints (`<ttl>` tag).
    - Increasing polling intervals if a feed rarely updates (a technique called backoff).
    - Using shared fetching infrastructure for multiple users subscribed to the same feed.

    ✅ **So in short:**
    Yes, most RSS apps poll feeds, but they optimize with HTTP headers (ETag, If-Modified-Since) and compare IDs/URLs/dates to detect new vs. updated items. More advanced setups use WebSub push hubs for real-time updates without polling.

    Would you like me to sketch out a timeline of what happens step by step when an RSS reader checks for updates? That might make the flow clearer.""",
        },
    ]

    max_part_length = 60
    conversation_summary = "### Conversation Summary\n\n"
    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]

        # Truncate content to first 100 and last 100 chars
        if len(content) > (max_part_length * 2):
            truncated = content[:max_part_length] + " ... " + content[-max_part_length:]
        else:
            truncated = content

        conversation_summary += f"**{role}:**\n```plaintext\n{truncated}\n```\n\n"

    mo.md(conversation_summary)
    return (messages,)


@app.cell
def _(mo, result):
    # Format the evaluation results for display
    results_md = "## Evaluation Results\n\n"

    for eval_output in result:
        results_md += f"### {eval_output.eval_name.replace('_', ' ').title()}\n\n"

        if eval_output.applicable:
            results_md += f"**Score:** {eval_output.score:.1f}/100\n\n"
        else:
            results_md += "**Status:** Not Applicable\n\n"

        if eval_output.feedback:
            results_md += f"**Feedback:**\n```plaintext\n{eval_output.feedback}\n```\n\n"

        if eval_output.eval_name == "check_criteria":
            results_md += "**Individual Criteria Results (from metadata):**\n\n"
            results_md += "| Criteria | Score | Pass/Fail |\n"
            results_md += "|----------|-------|----------|\n"
            for criteria_metadata in eval_output.metadata["criteria_evaluations"]:
                score = criteria_metadata["probability"] * 100
                pass_fail = "✅ Pass" if score >= 75 else "❌ Fail"
                results_md += f"| {criteria_metadata['criterion']} | {score:.1f} | {pass_fail} |\n"
            results_md += "\n"

        results_md += "\n---\n\n"
    mo.md(results_md)
    return


if __name__ == "__main__":
    app.run()
