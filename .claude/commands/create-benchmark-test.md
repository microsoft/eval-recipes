$ARGUMENTS

The user has optionally provided a description of a task they want or a bunch of data or something in-between.
It is your job to create a new benchmark task for them based on whatever they have provided. The following outlines the process which you should make a todo list for yourself and then execute it.

READ AND UNDERSTAND:
1. Creating high quality benchmark tests is challenging. You must first work to gain a deep understanding of what makes a good benchmark and think very hard throughout this process.
2. Start by reading the documentation at `docs/BENCHMARKING.md`
3. Then, read in the entire `scripts/run_benchmarks.py` and `eval_recipes/benchmarking/harness.py` to gain an understanding of how the inputs you will be creating will be used.
4. Next, read through existing examples of tasks in `data/tasks/`
  - Read everything under `data/tasks/cross_repo_improvement_tool`
  - Read everything under `data/tasks/email_drafting`


INSTANTIATE FROM TEMPLATE:
1. Create a new directory for the task under `data/tasks/` giving a good name and copy the template files into it.


FILL OUT TEMPLATE:
1. Start by filling out the `instructions.txt`. This will be the "user's" description for the agent describing what they want to have done.
   1. Remember, these instructions are meant to imitate **real** users who might not be engineers or experts in the field or AI. Create a todo-list item to self-reflect on your initial draft of instructions to make it more user-like.
2. Next, update the `setup.dockerfile`. This should **only** be updated if the user has specified a specific dependency that the agent could not determine to install itself. If anything, keep this file as minimal as possible. Don't forget to remove the placeholder comment.
3. Update `task.yaml` with the required environment variables, task info, and, for now, leave the `test_command` as the default.
4. Leave the `test.py` alone as a template for now. At the end, let the user know that they can use the command /create-semantic-tests to create semantic tests for this new task.
