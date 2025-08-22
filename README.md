# Eval Recipes

A library for making it easy to evaluate AI chat assistants in an online, "recipe-like" way.
Each evaluation will analyze an LLMs's responses against the inputs that generated it (messages and tools) over some domain, such as if user preferences were adhered to.


## Get Started Quick!

### 1. View notebooks directly on GitHub

Located in [demos/](./demos).

### 2. Run interactive notebooks with marimo

Run demo notebooks (the `.py` files located at [demos/](./demos)) with [`marimo`](https://docs.marimo.io/getting_started/installation/).
Follow the installation section below if you do not have `uv` installed or environment variables configured.

```bash
uv run marimo edit demos/1_evaluate.py
# Select Y to run in a sandboxed venv
```

### 3. Start using the package

```bash
uv pip install "git+https://github.com/microsoft/eval-recipes"
```

> [!WARNING]
> This library is very early and everything is subject to change. Consider pinning the dependency to a commit.


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

[LOW_LEVEL_API.md](./docs/LOW_LEVEL_API.md)


## Roadmap

[ROADMAP.md](./docs/ROADMAP.md)


## Development

[Generating Jupyter Notebooks](./docs/NOTEBOOKS.md)

