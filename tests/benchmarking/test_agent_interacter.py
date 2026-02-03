# Copyright (c) Microsoft. All rights reserved.

import pytest

from eval_recipes.benchmarking.evaluation.agent_interacter import interact_with_agent


@pytest.mark.skip(reason="requires API access")
@pytest.mark.parametrize(
    ("task_instructions", "agent_log"),
    [
        (
            "Create a Python script that calculates fibonacci numbers",
            """User: Create a Python script that calculates fibonacci numbers
Assistant: I can help you create a fibonacci calculator. Would you like me to:
1. Use recursion
2. Use iteration
3. Use memoization

Which approach would you prefer?""",
        ),
        (
            "Create a hello world script",
            """User: Create a hello world script
Assistant: I'll create a hello world script for you.

I've created hello.py with the following content:
print("Hello, World!")

The script is complete and ready to run.""",
        ),
    ],
)
async def test_interact_with_agent(task_instructions: str, agent_log: str) -> None:
    """
    Simple integration test for interact_with_agent.

    Tests that the function can process agent logs and task instructions
    and return an appropriate response (either a message or empty string).
    """
    response = await interact_with_agent(agent_log=agent_log, task_instructions=task_instructions)
    assert isinstance(response, str)
    print(f"Response: {response!r}")
