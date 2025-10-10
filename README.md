# Eval Recipes

**Eval Recipes** is an evaluation framework that makes it easy to evaluate LLM chat assistants, and does so at a high degree of quality. 
We use [recipes](https://sundaylettersfromsam.substack.com/p/what-is-an-ai-recipe) to develop specialized evaluation workflows that score conversations on 
dimensions such as how well the assistant adhered to user preferences or if they did not generate any hallucinations.

![Eval Recipes Animation](demos/data/EvalRecipesAnimation.gif)


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
> This library is very early and everything is subject to change. Consider pinning the dependency to a commit with the command like: `uv pip install "git+https://github.com/microsoft/eval-recipes@v0.0.7"`


## High Level API

The primary way of interacting with the package is the high-level API which takes in a list of messages
(defined by [OpenAI's responses API](https://platform.openai.com/docs/api-reference/responses/create#responses_create-input))
and a list of [custom tool definitions](https://platform.openai.com/docs/api-reference/responses/create#responses_create-tools) (built-in tools are not supported).

Each evaluation will output if it is deemed applicable to your input, an overall `score` from 0 to 100, and additional metadata specific to that evaluation.

Currently there are several built-in evaluations: `claim_verification`, `tool_usage`, `guidance`, and `preference_adherence`.
For more details on how these evaluations work, check the Low Level API section below.
Each evaluation can be additionally configured, such as selecting the LLM used. The full configurations are defined in [schemas.py](./eval_recipes/schemas.py).

`evaluate` will return a list of [`EvaluationOutput`](./eval_recipes/schemas.py) instances corresponding to each evaluation.


> [!TIP]
> All of the code examples in this readme can be pasted into a `.py` file and run as is!

```python
import asyncio
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.responses import EasyInputMessageParam, ResponseInputParam
from eval_recipes.evaluate import evaluate
from eval_recipes.evaluations.check_criteria.check_criteria_evaluator import CheckCriteriaEvaluatorConfig
from eval_recipes.schemas import BaseEvaluatorConfig

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
    config_preference_adherence = BaseEvaluatorConfig(model="gpt-5-mini")  # Sample config
    check_criteria = CheckCriteriaEvaluatorConfig(criteria=["Your response should be at least one paragraph long."])
    result = await evaluate(
        messages=messages,
        tools=tools,
        evaluations=["check_criteria", "claim_verification", "guidance", "preference_adherence", "tool_usage"],
        evaluation_configs={"preference_adherence": config_preference_adherence, "check_criteria": check_criteria},
        max_concurrency=1,
    )
    print(result)

asyncio.run(main())
```


### Custom Evaluations

You can create custom evaluators by implementing a class that follows the [`EvaluatorProtocol`](./eval_recipes/schemas.py).
This allows you to extend the evaluation framework with domain-specific metrics tailored to your needs.

Custom evaluators must implement:
1. An `__init__` method that accepts an optional `BaseEvaluatorConfig` parameter. If a config is not provided, you must initialize a default.
2. An async `evaluate` method that takes messages and tools as input and returns an `EvaluationOutput`

Here is an example of a custom evaluator that scores based on the length of the assistant's response being used in conjunction with the `preference_adherence` evaluator:

```python
import asyncio
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.responses import EasyInputMessageParam, ResponseInputParam
from eval_recipes.evaluate import evaluate
from eval_recipes.schemas import BaseEvaluatorConfig, EvaluationOutput

class ResponseLengthEvaluator:
    """Custom evaluator that scores based on response brevity."""
    def __init__(self, config: BaseEvaluatorConfig | None = None) -> None:
        self.config = config or BaseEvaluatorConfig()

    async def evaluate(self, messages: ResponseInputParam, tools: list[ChatCompletionToolParam]) -> EvaluationOutput:
        total_length = 0
        for message in reversed(messages):  # Only look at the last assistant message
            if ("role" in message and message["role"] == "assistant") and message.get("content"):
                total_length += len(str(message["content"]))
                break

        score = max(0, 100 - int(total_length // 25))  # Decrease score as length increases
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
        evaluation_configs={"ResponseLengthEvaluator": BaseEvaluatorConfig(model="gpt-5-mini")},
        max_concurrency=1,
    )
    print(result)

asyncio.run(main())
```


## Benchmarking

Eval Recipes provides a benchmarking harness for evaluating AI agents on real-world tasks in isolated Docker containers. 
We have a few sample tasks ranging from creating CLI applications to automations. Agents are automatically scored based on deterministic and semantic tests.

### Running Benchmarks

```bash
# The default agents/tasks require these environment variables
export ANTHROPIC_API_KEY=your_anthropic_key
export OPENAI_API_KEY=your_openai_key

uv run scripts/run_benchmarks.py

# Get more info about available arguments
uv run scripts/run_benchmarks.py --help
```

Results are saved to timestamped directories in `data/benchmarking/runs/` containing agent logs, test outputs, and structured results.
Any of these files may contain secrets that were used during the evaluation run. **NEVER** commit these files to source control without first checking for secrets.
For detailed documentation on creating custom agents and tasks, see [BENCHMARKING.md](./docs/BENCHMARKING.md).


## Development Installation
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

### Other

- [Generating Jupyter Notebooks](./docs/NOTEBOOKS.md)
- To re-create the [Manim](https://www.manim.community/) animation:
  - `make install-all` to install manim. See the docs if you have issues on a Linux-based system. Note this will also require `ffmpeg` to be installed.
  - `uv run manim scripts/create_animation.py EvalRecipesAnimation -qh && ffmpeg -y -i media/videos/create_animation/1080p60/EvalRecipesAnimation.mp4 -vf "fps=30,scale=1920:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 demos/data/EvalRecipesAnimation.gif`
- [Validating Evaluations](./tests/validate_evaluations.py):
  - This script will run evaluations against a small "goldset" (see [data/goldset](data/goldset/)) where we have inputs to evaluate with labels of what the scores should be (defined in [data/goldset/labels.yaml](data/goldset/labels.yaml)).


## Low Level API

[LOW_LEVEL_API.md](./docs/LOW_LEVEL_API.md)


## Changelog

[CHANGELOG.md](./docs/CHANGELOG.md)


## Roadmap

[ROADMAP.md](./docs/ROADMAP.md)


## Attributions

The built-in `claim_verification` evaluation is based on these two papers: [Claimify](https://arxiv.org/abs/2502.10855) and [VeriTrail](https://arxiv.org/abs/2505.21786). This is not an official implementation of either and please cite the original papers if you use this evaluation in your work.
