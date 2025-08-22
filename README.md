# Eval Recipes

A library for making it easy to evaluate AI chat assistants in an online, "recipe-like" way.
Each evaluation will analyze an LLMs's responses against the inputs that generated it (messages and tools) over some domain, such as if user preferences were adhered to.


## Get Started Quick!


Run demo notebooks (the `.py` files located at [demos/](./demos)) with [`marimo`](https://docs.marimo.io/getting_started/installation/).
Follow the installation section below if you do not have `uv` installed or environment variables configured.

```bash
uv run marimo edit demos/1_evaluate.py
# Select Y to run in a sandboxed venv
```

Install the library with `uv`:

```bash
uv pip install "git+https://github.com/microsoft/eval-recipes"
```

> [!WARNING]
> This library is very early and everything is subject to change. Consider pinning the dependency.


## Installation
### Prerequisites
- make
  - For Windows, you can download it using [UniGetUI](https://github.com/marticliment/UnigetUI) and use [ezwinports make](https://github.com/microsoft/winget-pkgs/tree/master/manifests/e/ezwinports/make)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Install Dependencies & Configure Environment

```bash
make install
cp .env.sample .env
# Configure API keys in .env
# Make sure the venv gets activated
. .venv/bin/activate # Linux example
```

This library requires either OpenAI or Azure OpenAI to be configured. You must set the correct environment variables in the `.env` file.

Check [utils.py `create_client`](./eval_recipes/utils/llm.py) to troubleshoot any configuration issues.


## High Level API

The primary way of interacting with the package is high-level API which takes in a list of messages
(defined by [OpenAI's responses API](https://platform.openai.com/docs/api-reference/responses/create#responses_create-input))
and a list of [custom tool definitions](https://platform.openai.com/docs/api-reference/responses/create#responses_create-tools) (built-in tools are not supported).

Each evaluation will output if it is deemed applicable to your input, an overall `score` from 0 to 100, and additional metadata specific to that evaluation.

Currently there are several built in evaluations: `claim_verification`, `tool_usage`, `guidance`, and `preference_adherence`.
For more details how these evaluations work, check the Low Level API section below.
Each evaluation can be additionally configured, such as selecting the LLM used. The full configurations are defined in [schemas.py](./eval_recipes/schemas.py).

`evaluate` will return a list of [`EvaluationOutput`](./eval_recipes/schemas.py) instances corresponding to each evaluation.


> [!TIP]
> All of the code examples in this readme can be pasted into a `.py` file and run as is!

```python
import asyncio
from eval_recipes.evaluate import evaluate
from eval_recipes.schemas import BaseEvaluationConfig
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.responses import (
    EasyInputMessageParam,
    ResponseInputParam,
)

async def main() -> None:
    messages: ResponseInputParam = [
        EasyInputMessageParam(
            role="system", content="You are a helpful assistant with search and document editing capabilities."
        ),
        EasyInputMessageParam(
            role="user",
            content="What material has the best elasticity for sports equipment? Please keep your response concise.",
        ),
        EasyInputMessageParam(
            role="assistant",
            content="Polyurethane elastomers offer excellent elasticity with 85% energy return and high durability.",
        ),
    ]

    tools: list[ChatCompletionToolParam] = [
        ChatCompletionToolParam(
            type="function",
            function={
                "name": "search",
                "description": "Search for information",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            },
        ),
    ]
    config_preference_adherence = BaseEvaluationConfig(model="gpt-5-mini")  # Sample config
    result = await evaluate(
        messages=messages,
        tools=tools,
        evaluations=["preference_adherence", "claim_verification", "tool_usage", "guidance"],
        evaluation_configs={"preference_adherence": config_preference_adherence},
        max_concurrency=1,
    )
    print(result)

asyncio.run(main())
```


### Custom Evaluations

You can create custom evaluators by implementing a class that follows the [`EvaluatorProtocol`](./eval_recipes/schemas.py).
This allows you to extend the evaluation framework with domain-specific metrics tailored to your needs.

Custom evaluators must implement:
1. An `__init__` method that accepts an optional `BaseEvaluationConfig` parameter. If a config is not not provided, you must initialize a default.
2. An async `evaluate` method that takes messages and tools as input and returns an `EvaluationOutput`

Here is an example of a custom evaluator that scores based on the length of the assistant's response being used in conjunction with the `preference_adherence` evaluator:

```python
import asyncio
from eval_recipes.evaluate import evaluate
from eval_recipes.schemas import BaseEvaluationConfig, EvaluationOutput
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.responses import (
    EasyInputMessageParam,
    ResponseInputParam,
)

class ResponseLengthEvaluator:
    """Custom evaluator that scores based on response brevity."""
    def __init__(self, config: BaseEvaluationConfig | None = None) -> None:
        self.config = config or BaseEvaluationConfig()

    async def evaluate(self, messages: ResponseInputParam, tools: list[ChatCompletionToolParam]) -> EvaluationOutput:
        total_length = 0
        for message in reversed(messages):  # Only look at the last assistant message
            if "role" in message and message["role"] == "assistant":
                if "content" in message and message["content"]:
                    total_length += len(str(message["content"]))
                    break

        score = max(0, 100 - int(total_length // 25)) # Decrease score as length increases
        return EvaluationOutput(eval_name="response_length", applicable=True, score=score, metadata={})

async def main() -> None:
    messages: ResponseInputParam = [
        EasyInputMessageParam(
            role="user",
            content="What material has the best elasticity for sports equipment? Please keep your response concise.",
        ),
        EasyInputMessageParam(
            role="assistant",
            content="Polyurethane elastomers offer excellent elasticity with 85% energy return and high durability.",
        ),
    ]
    result = await evaluate(
        messages=messages,
        tools=[],
        evaluations=[ResponseLengthEvaluator, "preference_adherence"],
        evaluation_configs={"ResponseLengthEvaluator": BaseEvaluationConfig(model="gpt-5-mini")},
        max_concurrency=1,
    )
    print(result)

asyncio.run(main())
```


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
