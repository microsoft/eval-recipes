# Copyright (c) Microsoft. All rights reserved.

import asyncio
import sys
from pathlib import Path

import click
from loguru import logger

from eval_recipes.benchmarking.semantic_test import semantic_test
from eval_recipes.benchmarking.test_utils import (
    get_instructions_from_file_or_default,
    get_test_id_from_env_or_default,
    write_test_result,
)


INJECTED_DISCREPANCIES = """### Discrepancy #1: Config `get()` Fallback Behavior

| Property | Value |
|----------|-------|
| **Location** | `./docs/config.md` (lines 34-39) |
| **Related Code** | `./knack/config.py` (lines 87-100) |
| **Type** | Behavioral claim mismatch |

#### What Was Injected

Added this **incorrect** section to `docs/config.md`:

```markdown
### Default Behavior

When retrieving a configuration value with `config.get(section, option)`, if the option
is not found in any configuration source and no fallback is provided, the method returns
`None` by default rather than raising an exception. This allows for safe retrieval of
optional configuration values.
```

#### Why It's Wrong

The actual implementation in `./knack/config.py:87-100` **raises an exception** when an option is not found:

```python
def get(self, section, option, fallback=_UNSET):
    env = self.env_var_name(section, option)
    if env in os.environ:
        return os.environ[env]
    last_ex = None
    for config in self._config_file_chain if self.use_local_config else self._config_file_chain[-1:]:
        try:
            return config.get(section, option)
        except (configparser.NoSectionError, configparser.NoOptionError) as ex:
            last_ex = ex

    if fallback is _UNSET:
        raise last_ex  # <-- RAISES EXCEPTION, does NOT return None
    return fallback
```

Key implementation details:
- `_UNSET = object()` is a sentinel value (line 11)
- When `fallback` is not provided, it defaults to `_UNSET`
- If option not found AND `fallback is _UNSET`, the code raises the captured exception
- Only if an explicit `fallback` is provided does it return that value

---

### Discrepancy #2: Validator Execution Order

| Property | Value |
|----------|-------|
| **Location** | `./docs/arguments.md` (line 39) |
| **Related Code** | `./knack/parser.py` (lines 151-161), `./knack/invocation.py` (lines 99-101) |
| **Type** | Subtle word change - incorrect ordering claim |

#### What Was Changed

Changed one phrase in the existing documentation:

**Original**: "...the order in which validators are executed is **random**..."

**Modified**: "...the order in which validators are executed is **alphabetical by argument name**..."

#### Why It's Wrong

The actual execution order is **insertion order** (the order arguments were added to the command), not alphabetical.

**Step 1**: Validators are collected in `./knack/parser.py:151-161`:
```python
argument_validators = []
for arg in metadata.arguments.values():  # dict iteration order
    # ...
    if arg.validator:
        argument_validators.append(arg.validator)  # appended in iteration order
```

**Step 2**: Validators are executed in `./knack/invocation.py:99-101`:
```python
def _validate_arg_level(self, ns, **_):
    for validator in getattr(ns, '_argument_validators', []):
        validator(ns)  # simple list iteration - preserves insertion order
```

Key implementation details:
- `metadata.arguments` is a dict, iterated in insertion order (Python 3.7+)
- Validators are appended to a list in that order
- Execution is a simple `for` loop over the list
- No sorting or alphabetical ordering occurs anywhere

---

### Discrepancy #3: table_transformer and --query Interaction

| Property | Value |
|----------|-------|
| **Location** | `./docs/output.md` (line 16) |
| **Related Code** | `./knack/output.py` (line 67), `./knack/query.py` (line 49), `./knack/invocation.py` (line 234) |
| **Type** | Feature interaction claim - implies compatibility when mutually exclusive |

#### What Was Changed

Extended an existing sentence to add a false claim:

**Original**: "The `table_transformer` is available when registering a command to define how it should look in table output."

**Modified**: "The `table_transformer` is available when registering a command to define how it should look in table output, **and is applied after any `--query` filtering**."

#### Why It's Wrong

The `table_transformer` is **completely disabled** when `--query` is used - they are mutually exclusive, not sequential.

**Step 1**: When `--query` is provided, the flag is set in `./knack/query.py:49`:
```python
cli_ctx.invocation.data['query_active'] = True
```

**Step 2**: This is passed to the result in `./knack/invocation.py:234`:
```python
return CommandResultItem(event_data['result'],
                         ...
                         is_query_active=self.data['query_active'],
                         ...)
```

**Step 3**: The table formatter checks this flag in `./knack/output.py:67`:
```python
if obj.table_transformer and not obj.is_query_active:  # <-- GATE
    # transformer only runs if query is NOT active
```

Key implementation details:
- The condition `not obj.is_query_active` means transformer is SKIPPED when query is active
- There is no "after" relationship - they are mutually exclusive
- The transformer silently doesn't run (no error) when --query is used
- Requires tracing through 3+ files to discover this interaction

---

### Discrepancy #4: Logging Flags Mutual Exclusivity

| Property | Value |
|----------|-------|
| **Location** | `./docs/logging.md` (line 15) |
| **Related Code** | `./knack/log.py` (lines 139-141) |
| **Type** | False compatibility claim - flags are mutually exclusive |

#### What Was Changed

Extended an existing sentence to add a false compatibility claim:

**Original**: "`--only-show-errors` - This flag changes the logging level to Error only, suppressing Warning."

**Modified**: "`--only-show-errors` - This flag changes the logging level to Error only, suppressing Warning. **Can be combined with `--verbose` to show errors with additional context.**"

#### Why It's Wrong

The code explicitly raises an error when these flags are combined. In `./knack/log.py:139-141`:

```python
if CLILogging.ONLY_SHOW_ERRORS_FLAG in args:
    if CLILogging.DEBUG_FLAG in args or CLILogging.VERBOSE_FLAG in args:
        raise CLIError("--only-show-errors can't be used together with --debug or --verbose")
```

Key implementation details:
- The check is explicit and intentional - not an edge case
- Raises `CLIError` with a clear error message
- Both `--verbose` and `--debug` are blocked when combined with `--only-show-errors`
- Can be discovered by reading code OR by simply trying the flags together

---

### Discrepancy #5: Argument Choices Display Order

| Property | Value |
|----------|-------|
| **Location** | `./docs/arguments.md` (line 48) |
| **Related Code** | `./knack/help.py` (line 498) |
| **Type** | Display behavior claim - incorrect ordering |

#### What Was Changed

Extended the existing `choices` documentation to add a false claim:

**Original**: "- `choices` - See https://docs.python.org/3/library/argparse.html#choices. If specified this will also serve as a value completer for people using tab completion."

**Modified**: "- `choices` - See https://docs.python.org/3/library/argparse.html#choices. If specified this will also serve as a value completer for people using tab completion. **When displayed in help output, choices appear in the order they were defined, allowing you to suggest preference by listing recommended values first.**"

#### Why It's Wrong

The actual implementation in `./knack/help.py:497-499` **sorts choices alphabetically**:

```python
@staticmethod
def _get_choices_defaults_sources_str(p):
    choice_str = '  Allowed values: {}.'.format(', '.join(sorted([str(x) for x in p.choices]))) \
        if p.choices else ''
```

Key implementation details:
- The `sorted()` function alphabetically sorts the choices before display
- Definition order is completely ignored
- Example: defining `choices=['zebra', 'mango', 'apple']` displays as `Allowed values: apple, mango, zebra.`
- Requires reading the help formatting code to discover this
- The false claim is plausible since many CLIs do preserve definition order"""


STEPS_CHECK_DISCREPANCIES = """The purpose of this task is to verify whether the agent could correctly verify a set of 5 discrepancies in the documentation that were injected versus the actual implementation/behavior of the code.
This the report of discrepancies that were injected:

# Injected Discrepancies
{{INJECTED_DISCREPANCIES}}

# Steps
Please follow these steps to evaluate the agent's findings:

1. Check if DISCREPANCIES.md exists in the /project directory.
2. If it exists, read the contents of DISCREPANCIES.md.
3. Review the file to determine if each of the following 5 discrepancies was correctly identified. \
For each discrepancy, the agent should have identified the core issue - exact wording is not required.
4. For each discrepancy, determine if the agent identified the core issue. \
The agent may have used different wording or found additional context, but should have identified the fundamental mismatch.
5. Do NOT penalize the agent for finding additional discrepancies beyond the ones that were injected. Only evaluate whether these specific 5 were found."""

RUBRIC_CHECK_DISCREPANCIES = {
    "file_created": "str - (5 points) Does DISCREPANCIES.md exist in /project? Award 5 points if yes, 0 if no.",
    "discrepancy_1_config_get": "str - (19 points) Did the agent identify that config.get() raises an exception \
(not returns None) when option not found? Award 19 points if correctly identified, 0 if missed.",
    "discrepancy_2_validator_order": "str - (19 points) Did the agent identify that validators run in insertion order \
(not alphabetical)? Award 19 points if correctly identified, 0 if missed.",
    "discrepancy_3_table_transformer_query": "str - (19 points) Did the agent identify that table_transformer is \
disabled/skipped when --query is used (not applied after)? Award 19 points if correctly identified, 0 if missed.",
    "discrepancy_4_logging_flags": "str - (19 points) Did the agent identify that --only-show-errors and --verbose \
cannot be combined (raises error)? Award 19 points if correctly identified, 0 if missed.",
    "discrepancy_5_choices_order": "str - (19 points) Did the agent identify that choices are sorted alphabetically \
in help (not definition order)? Award 19 points if correctly identified, 0 if missed.",
    "score": "float - Score between 0 and 100 based on the above criteria. Sum the points earned from each criterion.",
}


@click.command()
@click.option(
    "--test-id",
    default=lambda: get_test_id_from_env_or_default("dev"),
    help="Test ID for result file naming (defaults to EVAL_RECIPES_TEST_ID env var)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=lambda: Path(__file__).parents[0],
    help="Directory to write result file",
)
@click.option(
    "--instructions-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to instructions file (defaults to ./instructions.txt in working directory)",
)
def main(test_id: str, output_dir: Path, instructions_file: Path | None) -> int:
    """Test script for docs-discrepancy-1 task."""
    return asyncio.run(run_test(test_id, output_dir, instructions_file))


async def run_test(test_id: str, output_dir: Path, instructions_file: Path | None) -> int:
    instructions = get_instructions_from_file_or_default(instructions_file=instructions_file)

    try:
        logger.info("Running semantic test to check for discovered discrepancies...")
        result = await semantic_test(
            steps=STEPS_CHECK_DISCREPANCIES,
            rubric=RUBRIC_CHECK_DISCREPANCIES,
            context=instructions,
            working_dir=Path("/project"),
        )

        final_score = result.score
        metadata = {
            "instructions": instructions,
            "semantic_test_score": result.score,
            "semantic_test_metadata": result.metadata,
            "final_score": final_score,
        }

        write_test_result(output_dir, test_id, final_score, metadata)
        logger.info(f"Test completed with final score: {final_score:.1f}/100")
        return 0

    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        metadata = {
            "instructions": instructions,
            "error": str(e),
        }
        write_test_result(output_dir, test_id, 0, metadata)
        return 0


if __name__ == "__main__":
    sys.exit(main())
