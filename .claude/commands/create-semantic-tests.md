$ARGUMENTS

The user is trying to create a test.py for their new task. Creating high quality semantic tests is challenging and requires careful thought. You must break the problem down step by step to ensure tests are effective, focused, and actually validate that the agent's solution works.
This is the process for creating excellent semantic tests. Follow these steps carefully.


READ AND UNDERSTAND:
1. If the user did not provide a task folder, ask them for the path to the task folder. It should have an instructions.txt file.
2. Read in all existing files in the task folder, especially instructions.txt to understand what the agent needs to build.
3. Study existing high-quality examples:
   - Read data/tasks/email_drafting/test.py and data/tasks/email_drafting/instructions.txt
   - Read data/tasks/style_blender/test.py and data/tasks/style_blender/instructions.txt
   - Read eval_recipes/benchmarking/semantic_test.py to understand how semantic tests work
4. Note how these examples structure their tests and what makes them effective.


PROVIDE TEST DATA (Highly Recommended):
1. Consider whether the task would benefit from pre-provided test inputs in a data/ directory.
2. Benefits of providing test data:
   - Makes tests more robust (test doesn't need to generate inputs)
   - Enables specific, deterministic validation and makes test behavior more predictable and reproducible
   - Allows tests to focus on output quality rather than input generation
3. Look at data/tasks/email_drafting/data/ and data/tasks/style_blender/data/ as examples.
4. If appropriate, create a data/ directory with sample inputs the tests can use.
5. For tasks requiring distinct inputs (e.g., different writing styles, data formats), make inputs very different to make validation clear.


DESIGN AND PLAN SEMANTIC TESTS:
1. Design 3 focused semantic tests. Guidelines:
   - Fewer, high-quality tests are better than many diluted tests
   - Each test should be distinct and justify its existence
   - Avoid redundant tests that check the same things
   - Consider combining related concerns (e.g., "CLI + output organization" rather than separate tests)
   - IMPORTANT - Test Independence:
     * Each test MUST be completely independent and self-contained
     * Tests should NOT rely on outputs or state from previous tests
     * If a test needs to run the tool, it should find the README and run it itself
     * This allows tests to be run in any order or individually
     * Example: If Test 2 and Test 3 both need outputs, both should include steps to find README, run tool, and examine outputs

2. Test types to consider:
   a) Architecture/Implementation Review (static analysis)
      - Checks if the code follows required architecture (e.g., 3-stage pipeline)
      - Validates proper code organization and structure

   b) Functional Run + Validation (dynamic testing)
      - Actually runs the tool with test inputs
      - Validates CLI interface works
      - Checks output organization and file structure
      - Verifies error handling and validation

   c) Quality Validation with Test Data (comprehensive testing)
      - Runs the tool with provided test data
      - Validates output quality and correctness
      - Checks for expected characteristics in results
      - Tests end-to-end functionality

3. Structure each semantic test with:
   - Clear STEPS that tell the auditor what to do
   - Detailed RUBRIC with point breakdowns (should sum to 100)
   - Explicit scoring guidance in the rubric

4. Make tests action-oriented:
   - "Find the README and run the tool" NOT "Check if README exists"
   - "Run tool with test data and examine outputs" NOT "Look at the code"
   - "Create test inputs and validate results" NOT "Check if validation logic exists"


WRITE THE TEST.PY:
1. If test.py doesn't exist, copy from data/_template_task/test.py
2. Follow the contract specified in the template (required CLI options, result file format, exit codes)
3. Import required utilities from eval_recipes.benchmarking.test_utils
4. Define each semantic test with STEPS and RUBRIC as constants
5. Implement the async run_test function:
   - Call semantic_test() for each test
   - Calculate weighted final score
   - Include clear metadata showing individual test scores
   - Document scoring weights in metadata
6. Ensure proper error handling returns score of 0 with error metadata


PROVIDE RUN COMMAND:
1. Look at scripts/run_benchmarks.py to understand the command structure
2. Give the user a specific command to test their task:
   ```bash
   export ANTHROPIC_API_KEY=your_key
   export OPENAI_API_KEY=your_key
   uv run scripts/run_benchmarks.py --task-filter name=TASK_NAME --agent-filter name=claude_code
   ```
