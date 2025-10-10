The user is trying to create a test.py for their new task.

$ARGUMENTS

1. If they did not provide a task folder, ask them to give you the path to the task folder they are working on. It should have an instructions.txt
2. Read in all the files that currently exist in it.
3. If it does not already have a test.py, create one copying data/tasks/test_template.py
4. Look at data/tasks/email_drafting/test.py and data/tasks/email_drafting/instructions.txt for a reference example for a task that leverages semantic tests well. Also look at eval_recipes/benchmarking/semantic_test.py to understand how it works.
5. Now based on the user's instructions.txt, create a test.py that uses semantic tests intelligently in order to validate if the task was completed successfully.
