## Low Level API

We provide the low level API because the high level API makes assumptions about the structure of
your messages, tools, and how they are used in the evaluation.
For example, claim verification automatically ignores any assistant messages from the data that is verified against.
The low level API for each evaluation typically allows for more granular control of how to handle the data you'd like to be evaluated.


### Claim Verification

Verifies factual claims in text against source context.
Implemented as AsyncGenerator that yields partial results as claims are verified.

This evaluation is based on the following two papers: [Claimify](https://arxiv.org/abs/2502.10855) and [VeriTrail](https://arxiv.org/abs/2505.21786).
This is not an official implementation of either and please cite the original papers if you use this evaluation in your work.


**Metric**: Number of verified claims / (total number of claims - number of "open-domain" claims)

```python
import asyncio
from eval_recipes.evaluations.claim_verification.claim_verifier import ClaimVerifier, InputClaimVerifier, InputContext
from eval_recipes.schemas import ClaimVerifierConfig

async def main() -> None:
    input_data = InputClaimVerifier(
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
    config = ClaimVerifierConfig() # (optionally) configure models and other parameters here
    verifier = ClaimVerifier(config=config)
    async for result in verifier.run(input_data):
        print(result)

asyncio.run(main())
```


### Preference Adherence

Evaluates how well an assistant adheres to user preferences.
It first will extracts user preferences from messages and then evaluates adherence to **each** of them.

**Metric**: Number of preferences adhered to / total number of preferences

```python
import asyncio

from eval_recipes.evaluations.user_preferences import InputUserPreferences, UserPreferencesEvaluator
from eval_recipes.schemas import BaseEvaluationConfig

async def main() -> None:
    input_data = InputUserPreferences(
        conversation_history_beginning_turn="""System: You are a helpful assistant. You are concise and avoid emojis in your response
User: What is Python?""",
        conversation_history_full="""System: You are a helpful assistant. You are concise and avoid emojis in your response
User: What is Python?
Assistant: Python is a high-level, interpreted programming language known for simplicity and readability.""",
    )
    config = BaseEvaluationConfig(model="gpt-5-mini")
    evaluator = UserPreferencesEvaluator(config=config)
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
from eval_recipes.evaluations.tool_evaluator import ToolEvaluator, InputToolEvaluator, InputTool
from eval_recipes.schemas import ToolEvaluationConfig

async def main() -> None:
    input_data = InputToolEvaluator(
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
    config = ToolEvaluationConfig() # (optionally) configure models and tool thresholds here
    evaluator = ToolEvaluator(config=config)
    result = await evaluator.run(input_data)
    print(result)

asyncio.run(main())
```


### Guidance

Evaluates how gracefully an assistant handles out-of-scope requests.
Determines if requests are within capabilities and evaluates response quality for out-of-scope requests.

```python
import asyncio
from eval_recipes.evaluations.guidance_evaluator import GuidanceEvaluator, InputGuidanceEval
from eval_recipes.schemas import GuidanceEvaluationConfig

async def main() -> None:
    input_data = InputGuidanceEval(
        conversation_history_full="""System: You can help with text tasks.
User: Can you create an Excel spreadsheet for me?
Assistant: I cannot create Excel files, but I can help you create a CSV text file that can be opened in Excel.""",
        conversation_history_beginning_turn="""System: You can help with text tasks.
User: Can you create an Excel spreadsheet for me?""",
    )
    config = GuidanceEvaluationConfig(
        capability_manifest="## Capabilities\n- Create and edit text files\n- Cannot create binary files like Excel spreadsheets"
    ) # (recommended) provide the capability manifest or it will be auto-generated
    evaluator = GuidanceEvaluator(config=config)
    result = await evaluator.run(input_data)
    print(result)

asyncio.run(main())
```

The `generate_capability_manifest` function helps create the capability_manifest from system prompts and tool definitions
This is useful for preprocessing noisy system prompts into clear capability descriptions.

```python
import asyncio
from eval_recipes.evaluations.guidance_evaluator import generate_capability_manifest

async def main() -> None:
    system_prompt = "You are a helpful assistant that can search the web"
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Perform mathematical calculations"
            }
        }
    ]

    manifest = await generate_capability_manifest(
        system_prompt=system_prompt,
        tools=tools,
        provider="openai",
        model="gpt-5"
    )
    print(manifest)

asyncio.run(main())
```
