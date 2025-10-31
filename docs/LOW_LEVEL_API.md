## Low Level API

We provide the low level API because the high level API makes assumptions about the structure of
your messages, tools, and how they are used in the evaluation.
For example, claim verification automatically ignores any assistant messages from the data that is verified against.
The low level API for each evaluation typically allows for more granular control of how to handle the data you'd like to be evaluated.


### Check Criteria

Evaluates assistant responses against custom criteria or rubrics you define. This is a useful "catch-all" evaluation for simpler requirements like tone, format, or content guidelines.

**Metric**: Average probability across all criteria (0-100), where each criterion is evaluated independently

```python
import asyncio

from eval_recipes.evaluations.check_criteria.check_criteria_evaluator import (
    CheckCriteriaEvaluator,
    CheckCriteriaEvaluatorConfig,
)

async def main() -> None:
    config = CheckCriteriaEvaluatorConfig(
        criteria=[
            "The response should be exactly one paragraph",
            "The response should end with a question",
        ],
        passed_threshold=75,  # Criteria scoring below 75% will be included in feedback
        model="gpt-5-mini",
    )
    evaluator = CheckCriteriaEvaluator(config=config)
    messages = [
        {"role": "user", "content": "What is a programming language?"},
        {
            "role": "assistant",
            "content": "A programming language is a formal system of instructions that computers can execute. Popular examples include Python, JavaScript, and Java. Each language has its own syntax and use cases. What type of programming are you interested in learning?"
        }
    ]
    
    result = await evaluator.evaluate(messages, tools=[])
    print(f"Score: {result.score:.1f}%")
    print(f"Feedback: {result.feedback}")
    for eval in result.metadata["criteria_evaluations"]:
        print(f"- {eval['criterion']}: {eval['probability']*100:.0f}%")

asyncio.run(main())
```


### Claim Verification

Verifies factual claims in text against source context.
Implemented as AsyncGenerator that yields partial results as claims are verified.

This evaluation is based on the following two papers: [Claimify](https://arxiv.org/abs/2502.10855) and [VeriTrail](https://arxiv.org/abs/2505.21786).
This is not an official implementation of either and please cite the original papers if you use this evaluation in your work.

**Metric**: Number of verified claims / (total number of claims - number of "open-domain" claims)

```python
import asyncio

from eval_recipes.evaluations.claim_verification.claim_verification_evaluator import (
    ClaimVerificationEvaluator,
    ClaimVerificationEvaluatorConfig,
    InputClaimVerificationEvaluator,
    InputContext,
)

async def main() -> None:
    input_data = InputClaimVerificationEvaluator(
        text="Paris is the capital of France. It has 12 million residents.",
        user_question="Tell me about Paris",
        source_context=[
            InputContext(
                source_id="1",
                title="Wikipedia",
                content="Paris is the capital city of France with 2.1 million inhabitants.",
            )
        ],
    )
    config = ClaimVerificationEvaluatorConfig()  # (optionally) configure models and other parameters here
    verifier = ClaimVerificationEvaluator(config=config)
    async for result in verifier.run(input_data):
        print(result)

asyncio.run(main())
```


### Guidance

Evaluates how gracefully an assistant handles out-of-scope requests.
Determines if requests are within capabilities and evaluates response quality for out-of-scope requests.

```python
import asyncio
from eval_recipes.evaluations.guidance.guidance_evaluator import (
    GuidanceEvaluator,
    GuidanceEvaluatorConfig,
    InputGuidanceEval,
)

async def main() -> None:
    input_data = InputGuidanceEval(
        conversation_history_full="""System: You can help with text tasks.
User: Can you create an Excel spreadsheet for me?
Assistant: I cannot create Excel files, but I can help you create a CSV text file that can be opened in Excel.""",
        conversation_history_beginning_turn="""System: You can help with text tasks.
User: Can you create an Excel spreadsheet for me?""",
    )
    config = GuidanceEvaluatorConfig(
        capability_manifest="## Capabilities\n- Create and edit text files\n- Cannot create binary files like Excel spreadsheets"
    )  # (recommended) provide the capability manifest or it will be auto-generated
    evaluator = GuidanceEvaluator(config=config)
    result = await evaluator.run(input_data)
    print(result)

asyncio.run(main())
```

The `generate_capability_manifest` function helps create the capability_manifest from system prompts and tool definitions
This is useful for preprocessing noisy system prompts into clear capability descriptions.

```python
import asyncio
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from eval_recipes.evaluations.guidance.guidance_evaluator import generate_capability_manifest

async def main() -> None:
    system_prompt = "You are a helpful assistant that can search the web"
    tools = [
        ChatCompletionToolParam(
            {"type": "function", "function": {"name": "calculator", "description": "Perform mathematical calculations"}}
        )
    ]
    manifest = await generate_capability_manifest(
        system_prompt=system_prompt, tools=tools, provider="openai", model="gpt-5"
    )
    print(manifest)

asyncio.run(main())
```


### Preference Adherence

Evaluates how well an assistant adheres to user preferences.
It first extracts user preferences from messages and then evaluates adherence to **each** of them.

**Metric**: Number of preferences adhered to / total number of preferences

```python
import asyncio

from eval_recipes.evaluations.preference_adherence.preference_adherence_evaluator import (
    InputUserPreferences,
    PreferenceAdherenceEvaluator,
)
from eval_recipes.schemas import BaseEvaluatorConfig

async def main() -> None:
    input_data = InputUserPreferences(
        conversation_history_beginning_turn="""System: You are a helpful assistant. You are concise and avoid emojis in your response
User: What is Python?""",
        conversation_history_full="""System: You are a helpful assistant. You are concise and avoid emojis in your response
User: What is Python?
Assistant: Python is a high-level, interpreted programming language known for simplicity and readability.""",
    )
    config = BaseEvaluatorConfig(model="gpt-5-mini")
    evaluator = PreferenceAdherenceEvaluator(config=config)
    result = await evaluator.run(input_data)
    print(result)

asyncio.run(main())
```


### Tool Usage

Evaluates whether an assistant correctly uses available tools.
Calculates probability that each tool should be called based on conversation context.

**Metric**
- If no tools were called (was_called=False for all)
  - Each probability should be below its threshold. If so, return 100, 0 otherwise.
- If was_called=True for any tool:
  - Check **any** of those tool's probability is above its threshold, if so return 100.
  - Otherwise if none of the "was_called=True" have a probability over the threshold, return 0. (This implies a tool was called when it should not have been)

```python
import asyncio
from eval_recipes.evaluations.tool_usage.tool_usage_evaluator import (
    InputTool,
    InputToolUsageEvaluator,
    ToolUsageEvaluator,
    ToolUsageEvaluatorConfig,
)

async def main() -> None:
    input_data = InputToolUsageEvaluator(
        tools=[
            InputTool(
                tool_name="search",
                tool_text="Search for information on the web",
                was_called=False,
                threshold=50,
            ),
            InputTool(
                tool_name="calculator",
                tool_text="Perform mathematical calculations",
                was_called=True,
                threshold=50,
            ),
        ],
        conversation_history_full="User: What is 15% of 200?\nAssistant: I'll calculate that for you.",
    )
    config = ToolUsageEvaluatorConfig()  # (optionally) configure models and tool thresholds here
    evaluator = ToolUsageEvaluator(config=config)
    result = await evaluator.run(input_data)
    print(result)

asyncio.run(main())
```


### Semantic Test

Evaluates agent work by using an AI agent to audit deliverables against custom steps and rubrics.
The auditor agent explores a working directory, follows specified steps, and completes a structured rubric.

**Metric**: Score extracted from the completed rubric (0-100).

```python
import asyncio
from pathlib import Path

from eval_recipes.evaluations.semantic_test.semantic_test_evaluator import (
    SemanticTestEvaluator,
    SemanticTestEvaluatorConfig,
)

async def main() -> None:
    # Setup: working_dir should contain the agent's deliverables
    working_dir = Path("/path/to/agent/work")

    context = "Create a Python script that calculates fibonacci numbers using recursion"

    steps = """1. Check if the Python file exists
1. Read the file and verify it implements fibonacci using recursion
2. Test the implementation by running it"""

    rubric = {
        "file_exists": "20 points boolean - does the file exist?",
        "uses_recursion": "40 points boolean - does it use recursion?",
        "works_correctly": "40 points boolean - does it produce correct results?",
        "score": "number (0-100) - total score based on criteria above",
    }

    config = SemanticTestEvaluatorConfig(
        working_dir=working_dir,
        steps=steps,
        rubric=rubric,
        context=context,
    )
    evaluator = SemanticTestEvaluator(config=config)
    result = await evaluator.run(
        working_dir=working_dir,
        steps=steps,
        rubric=rubric,
        context=context,
    )
    print(f"Score: {result.score:.1f}")
    print(f"Feedback: {result.feedback}")
    print(f"Metadata: {result.metadata}")

asyncio.run(main())
```
