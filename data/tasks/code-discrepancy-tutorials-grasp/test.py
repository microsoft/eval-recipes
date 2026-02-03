# Copyright (c) Microsoft. All rights reserved.

import asyncio
from pathlib import Path
import sys

import click
from loguru import logger

from eval_recipes.benchmarking.evaluation.semantic_test import semantic_test
from eval_recipes.benchmarking.evaluation.test_utils import (
    get_instructions_from_file_or_default,
    get_test_id_from_env_or_default,
    write_test_result,
)

INJECTED_DISCREPANCIES = """### Discrepancy #1: Outdated Parameter Name in AdjacencySpectralEmbed Tutorial

**File**: `graspologic-repo-1/docs/tutorials/embedding/AdjacencySpectralEmbed.ipynb`

**Location**: Cell 12 (code cell under "Dimension specification" section)

**Error When Running**:
```
TypeError: AdjacencySpectralEmbed.__init__() got an unexpected keyword argument 'n_dims'
```

**What's Wrong**:
The tutorial uses the outdated parameter name `n_dims` instead of the current `n_components`:

```python
# Tutorial code (WRONG - old API)
ase = AdjacencySpectralEmbed(n_dims=2, algorithm='truncated')
```

**The Fix**:
Change `n_dims` to `n_components`:

```python
# Correct code (current API)
ase = AdjacencySpectralEmbed(n_components=2, algorithm='truncated')
```

**Also Update** (optional but recommended):
Cell 11 (markdown) mentions `n_dims` parameter - should be updated to `n_components` for consistency.

**How to Verify the Fix**:
The correct parameter name can be found in:
- `graspologic/embed/ase.py:116` - `n_components: Optional[int] = None`
- The docstring at `graspologic/embed/ase.py:25-29` documents the `n_components` parameter

**Detection Methods**:
1. Run the notebook and observe the TypeError
2. Read the AdjacencySpectralEmbed class definition in `graspologic/embed/ase.py`
3. Check the graspologic documentation or `help(AdjacencySpectralEmbed)`

---

### Discrepancy #2: Removed Function in Omnibus Tutorial

**File**: `graspologic-repo-1/docs/tutorials/embedding/Omnibus.ipynb`

**Location**: Cell 9 (code cell under "Visualize the singular values" section)

**Error When Running**:
```
ImportError: cannot import name 'plot_scree' from 'graspologic.plot'
```

**What's Wrong**:
The tutorial uses a function `plot_scree` that doesn't exist in the current library. The actual function is named `screeplot` (no underscore):

```python
# Tutorial code (WRONG - function doesn't exist)
from graspologic.plot import plot_scree
plot_scree(embedder.singular_values_, title="Omnibus Embedding Scree Plot")
```

**The Fix**:
1. Change `plot_scree` to `screeplot`
2. Note that `screeplot` takes a matrix (not singular values), so the call needs adjustment:

```python
# Correct code (current API)
from graspologic.plot import screeplot
# screeplot computes SVD internally, so pass a graph matrix
screeplot(G1, title="Graph 1 Scree Plot")
```

**Alternative Fix**: Remove the cell entirely if scree plot visualization isn't needed.

**How to Verify the Fix**:
- Check `graspologic/plot/__init__.py` - exports `screeplot`, not `plot_scree`
- Check `graspologic/plot/plot.py:1414` for the `screeplot` function signature

**Detection Methods**:
1. Run the notebook and observe the ImportError
2. List available functions in graspologic.plot: `dir(graspologic.plot)` or check `__init__.py`
3. Search for similar function names in the plot module

---

### Discrepancy #3: Non-existent GraphMatch Class in SGM Tutorial

**File**: `graspologic-repo-1/docs/tutorials/matching/sgm.ipynb`

**Location**: Cell 3 (imports) and Cells 11, 14 (graph matching usage)

**Error When Running**:
```
ImportError: cannot import name 'GraphMatch' from 'graspologic.match'
```

**What's Wrong**:
The tutorial uses a class-based API (`GraphMatch` with a `fit()` method) that doesn't exist. The graspologic matching module only exports a functional interface (`graph_match` function):

```python
# Tutorial code (WRONG - class doesn't exist)
from graspologic.match import GraphMatch
gm = GraphMatch(rng=rng)
_, perm_inds, _, _ = gm.fit(A1, A2_shuffle)
```

**The Fix**:
Use the `graph_match` function instead of a non-existent `GraphMatch` class:

```python
# Correct code (current API)
from graspologic.match import graph_match
_, perm_inds, _, _ = graph_match(A1, A2_shuffle, rng=rng)
```

For the seeded version (Cell 14):
```python
# Correct code with partial_match
_, perm_inds, _, _ = graph_match(A1, A2_shuffle, partial_match=partial_match, rng=rng)
```

**How to Verify the Fix**:
- Check `graspologic/match/__init__.py:4-6` - only exports `graph_match` function
- Check `graspologic/match/wrappers.py:50` - defines `graph_match` as a function, not a class
- Run `from graspologic.match import GraphMatch` in Python to confirm ImportError

**Detection Methods**:
1. Run the notebook and observe the ImportError on the import cell
2. Check `graspologic.match.__all__` or `dir(graspologic.match)` to see available exports
3. Read `graspologic/match/__init__.py` to see that only `graph_match` is exported
4. Inspect the source to understand that the API is functional, not class-based

---

### Discrepancy #4: Fake Tutorial with Multiple API Errors (Community Detection)

**File**: `graspologic-repo-1/docs/tutorials/partition/leiden.ipynb`

This is a **completely new tutorial** that looks plausible but contains multiple API errors. The partition module has no official tutorial, making this fake one appear legitimate.

**Errors When Running**:

1. **Cell 1 (imports)**: `ImportError: cannot import name 'LeidenCluster' from 'graspologic.partition'`
2. **Cell 8**: `TypeError: leiden() got an unexpected keyword argument 'n_clusters'`

---

#### Issue A: Non-existent `LeidenCluster` Class

**Location**: Cell 1 (imports) and Cell 6 (usage)

**What's Wrong**:
The tutorial uses a class-based API `LeidenCluster` that doesn't exist. The actual API is the functional `leiden()` function:

```python
# Tutorial code (WRONG - class doesn't exist)
from graspologic.partition import LeidenCluster

lc = LeidenCluster(resolution=1.0, random_seed=42)
communities = lc.fit_predict(G)
```

**The Fix**:
```python
# Correct code (current API)
from graspologic.partition import leiden

communities = leiden(G, resolution=1.0, random_seed=42)
```

---

#### Issue B: Non-existent `n_clusters` Parameter

**Location**: Cell 8

**What's Wrong**:
The tutorial uses `n_clusters` parameter which doesn't exist. Leiden uses `resolution` to indirectly control the number of communities:

```python
# Tutorial code (WRONG - parameter doesn't exist)
communities = leiden(G, n_clusters=4, random_seed=42)
```

**The Fix**:
```python
# Correct code (current API)
# Use resolution parameter - higher values = more communities
# Note: Cannot specify exact number of communities with Leiden
communities = leiden(G, resolution=1.5, random_seed=42)
```

---

**How to Verify the Fix**:
- Check `graspologic/partition/__init__.py` - exports `leiden` function, not `LeidenCluster` class
- Check `graspologic/partition/leiden.py:178-194` for the `leiden()` function signature
- The parameters are: `graph`, `starting_communities`, `extra_forced_iterations`, `resolution`, `randomness`, `use_modularity`, `random_seed`, `weight_attribute`, `is_weighted`, `weight_default`, `check_directed`, `trials`
- No `n_clusters` parameter exists

**Detection Methods**:
1. Run the notebook and observe the ImportError on the import cell
2. Check `graspologic.partition.__all__` to see available exports: `['HierarchicalCluster', 'HierarchicalClusters', 'hierarchical_leiden', 'leiden', 'modularity', 'modularity_components']`
3. Read the `leiden()` function signature in `graspologic/partition/leiden.py`
4. Note that `LeidenCluster` would follow sklearn patterns but graspologic uses functional APIs for partitioning"""


STEPS_CHECK_DISCREPANCIES = """The purpose of this task is to verify whether the agent could correctly verify a set of 4 discrepancies in the documentation that were injected versus the actual implementation/behavior of the code.
This the report of discrepancies that were injected:

# Injected Discrepancies
{{INJECTED_DISCREPANCIES}}

# Steps
Please follow these steps to evaluate the agent's findings:

1. Check if DISCREPANCIES.md exists in the /project directory.
2. If it exists, read the contents of DISCREPANCIES.md.
3. Review the file to determine if each of the following 4 discrepancies was correctly identified. \
For each discrepancy, the agent should have identified the core issue - exact wording is not required.
4. For each discrepancy, determine if the agent identified the core issue. \
The agent may have used different wording or found additional context, but should have identified the fundamental mismatch.
5. Do NOT penalize the agent for finding additional discrepancies beyond the ones that were injected. Only evaluate whether these specific 4 were found."""

RUBRIC_CHECK_DISCREPANCIES = {
    "file_created": "str - (4 points) Does DISCREPANCIES.md exist in /project? Award 4 points if yes, 0 if no.",
    "discrepancy_1_n_dims_param": "str - (24 points) Did the agent identify that AdjacencySpectralEmbed.ipynb tutorial \
uses outdated parameter name 'n_dims' instead of the correct 'n_components'? \
Award 24 points if correctly identified, 0 if missed.",
    "discrepancy_2_plot_scree": "str - (24 points) Did the agent identify that Omnibus.ipynb tutorial \
uses non-existent function 'plot_scree' instead of the correct 'screeplot'? \
Award 24 points if correctly identified, 0 if missed.",
    "discrepancy_3_graph_match_class": "str - (24 points) Did the agent identify that sgm.ipynb tutorial \
uses non-existent 'GraphMatch' class instead of the correct 'graph_match' function? \
Award 24 points if correctly identified, 0 if missed.",
    "discrepancy_4_leiden_tutorial": "str - (24 points) Did the agent identify issues in the fake leiden.ipynb tutorial: \
non-existent 'LeidenCluster' class (should use 'leiden' function) and/or non-existent 'n_clusters' parameter? \
Award 24 points if at least one of these issues was identified, 0 if missed.",
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
