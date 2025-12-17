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


INJECTED_DISCREPANCIES = """### Discrepancy #1: GaussianCluster `bic_` DataFrame Column Ordering

| Property | Value |
|----------|-------|
| **Location** | `./graspologic/cluster/gclust.py` (lines 98-102) |
| **Related Code** | `./graspologic/cluster/gclust.py` (lines 168-171, 259-263) |
| **Type** | DataFrame column order claim - incorrect ordering |

#### What Was Changed

Extended the `bic_` attribute docstring to add a false claim about column ordering:

**Original** (lines 98-101):
```
bic_ : pandas.DataFrame
    A pandas DataFrame of BIC values computed for all possible number of clusters
    given by ``range(min_components, max_components + 1)`` and all covariance
    structures given by :attr:`covariance_type`.
```

**Modified** (lines 98-102):
```
bic_ : pandas.DataFrame
    A pandas DataFrame of BIC values computed for all possible number of clusters
    given by ``range(min_components, max_components + 1)`` and all covariance
    structures given by :attr:`covariance_type`. Columns are ordered according to
    the order of covariance types specified in the input parameter.
```

#### Why It's Wrong

The implementation **silently reorders** covariance types to a fixed internal order, regardless of user input.

**In `__init__` (lines 168-171):**
```python
new_covariance_type = []
for cov in ["spherical", "diag", "tied", "full"]:
    if cov in covariance_type:
        new_covariance_type.append(cov)
```

This loop iterates through a fixed order `["spherical", "diag", "tied", "full"]` and only includes types the user specified, but **in the fixed order**, not the user's order.

**When creating `bic_` DataFrame (lines 259-263):**
```python
self.bic_ = pd.DataFrame(
    bics,
    index=np.arange(lower_ncomponents, upper_ncomponents + 1),
    columns=self.covariance_type,  # Uses the REORDERED list!
)
```

#### Example

```python
from graspologic.cluster import GaussianCluster
import numpy as np

X = np.random.randn(100, 2)
gc = GaussianCluster(covariance_type=['full', 'diag'])  # User's order
gc.fit(X)

print(gc.bic_.columns.tolist())
# Output: ['diag', 'full']  <- REORDERED, not ['full', 'diag']!
```

#### Key Implementation Details

- The reordering happens silently in `__init__`, not during `fit()`
- Affects both `bic_` and `ari_` DataFrames
- The false claim is plausible since most APIs preserve user-specified order
- Requires reading the `__init__` method AND understanding how `self.covariance_type` is used later
- Could cause subtle bugs if users rely on column position rather than column name

---

### Discrepancy #2: select_svd `svd_seed=None` Behavior

| Property | Value |
|----------|-------|
| **Location** | `./graspologic/embed/svd.py` (lines 238-242) |
| **Related Code** | `./graspologic/embed/svd.py` (lines 317-320) |
| **Type** | Default parameter behavior - false claim about randomness |

#### What Was Changed

Extended the `svd_seed` parameter docstring to add a false claim about None behavior:

**Original** (lines 238-240):
```
svd_seed : int or None (default ``None``)
    Only applicable for ``algorithm="randomized"``; allows you to seed the
    randomized svd solver for deterministic, albeit pseudo-randomized behavior.
```

**Modified** (lines 238-242):
```
svd_seed : int or None (default ``None``)
    Only applicable for ``algorithm="randomized"``; allows you to seed the
    randomized svd solver for deterministic, albeit pseudo-randomized behavior.
    When set to None, the algorithm uses system entropy for non-deterministic
    results, which is useful when reproducibility is not required.
```

#### Why It's Wrong

The implementation **silently converts None to 0**, making the output deterministic even when the user doesn't provide a seed.

**In `select_svd` (lines 317-320):**
```python
elif algorithm == "randomized":
    # for some reason, randomized_svd defaults random_state to 0 if not provided
    # which is weird because None is a valid starting point too
    svd_seed = svd_seed if svd_seed is not None else 0
```

The code even has a comment acknowledging this is "weird" behavior! When `svd_seed=None` is passed, it gets converted to `0` before being passed to sklearn's `randomized_svd`.

#### Example

```python
from graspologic.embed import select_svd
import numpy as np

X = np.random.randn(50, 30)

# Run multiple times with svd_seed=None
results = []
for _ in range(3):
    U, D, V = select_svd(X, n_components=5, algorithm='randomized', svd_seed=None)
    results.append(U.copy())

# All results are IDENTICAL - not random at all!
print(np.allclose(results[0], results[1]))  # True
print(np.allclose(results[0], results[2]))  # True

# svd_seed=None gives same result as svd_seed=0
U_none, _, _ = select_svd(X, n_components=5, algorithm='randomized', svd_seed=None)
U_zero, _, _ = select_svd(X, n_components=5, algorithm='randomized', svd_seed=0)
print(np.allclose(U_none, U_zero))  # True
```

#### Key Implementation Details

- The conversion happens at runtime in `select_svd()`, not in parameter defaults
- The code comment explicitly notes this is "weird" behavior
- Users expecting non-deterministic behavior will get reproducible results
- This affects `AdjacencySpectralEmbed` and other classes that use `select_svd` internally
- The false claim is plausible since many Python APIs treat `None` as "use random initialization"

---

### Discrepancy #3: SeedlessProcrustes `initial_P` Soft Assignment Matrix Description

| Property | Value |
|----------|-------|
| **Location** | `./graspologic/align/seedless_procrustes.py` (lines 83-84) |
| **Related Code** | `./graspologic/align/seedless_procrustes.py` (lines 253-263) |
| **Type** | Matrix normalization claim - row/column sums swapped |

#### What Was Changed

Swapped the row and column sum requirements in the `initial_P` parameter docstring:

**Original** (lines 83-84):
```
    Must be a soft assignment matrix if provided (rows sum up to 1/n, cols
    sum up to 1/m.)
```

**Modified** (lines 83-84):
```
    Must be a soft assignment matrix if provided (rows sum up to 1/m, cols
    sum up to 1/n.)
```

#### Why It's Wrong

The implementation requires the OPPOSITE normalization:

**In `__init__` validation (lines 253-257):**
```python
n, m = initial_P_checked.shape
if not (
    np.allclose(initial_P_checked.sum(axis=0), np.ones(m) / m)  # column sums = 1/m
    and np.allclose(initial_P_checked.sum(axis=1), np.ones(n) / n)  # row sums = 1/n
):
```

The code checks:
- Row sums (axis=1) must equal 1/n (where n = number of rows)
- Column sums (axis=0) must equal 1/m (where m = number of columns)

The docstring falsely claims the opposite: rows sum to 1/m and cols sum to 1/n.

#### Interesting Twist

The error message at lines 259-261 is ALREADY WRONG in the same way:
```python
msg = (
    "Initial_P must be a soft assignment matrix "
    "(rows add up to (1/number of cols) "   # WRONG! Should be 1/number of rows
    "and columns add up to (1/number of rows))"  # WRONG! Should be 1/number of cols
)
```

So the injected docstring now matches the (wrong) error message, creating a consistent but incorrect documentation story!

#### Example

```python
from graspologic.align import SeedlessProcrustes
import numpy as np

# Create test data: X has 5 rows, Y has 4 rows
X = np.random.randn(5, 3)
Y = np.random.randn(4, 3)

n, m = 5, 4  # initial_P should be (5, 4)

# Following the WRONG docstring: rows sum to 1/m=0.25, cols sum to 1/n=0.2
wrong_P = np.ones((n, m)) / (m * m)  # rows sum to 0.25
print(f"Row sums: {wrong_P.sum(axis=1)}")  # [0.25, 0.25, 0.25, 0.25, 0.25]

# This will be REJECTED even though it follows the docstring!
sp = SeedlessProcrustes(init='custom', initial_P=wrong_P)
sp.fit(X, Y)  # Raises ValueError

# CORRECT: rows sum to 1/n=0.2, cols sum to 1/m=0.25
correct_P = np.ones((n, m)) / (n * m)
print(f"Row sums: {correct_P.sum(axis=1)}")  # [0.2, 0.2, 0.2, 0.2, 0.2]
sp = SeedlessProcrustes(init='custom', initial_P=correct_P)
sp.fit(X, Y)  # Works!
```

#### Key Implementation Details

- Requires understanding matrix row/column sum semantics
- The swapped description sounds equally plausible
- The wrong error message reinforces the wrong docstring (existing bug!)
- Users following the docstring will create invalid matrices
- Only fails when n ≠ m (when n = m, the constraints are equivalent)

---

### Discrepancy #4: KMeansCluster Ghost Parameter `n_init`

| Property | Value |
|----------|-------|
| **Location** | `./graspologic/cluster/kclust.py` (lines 37-40) |
| **Related Code** | `./graspologic/cluster/kclust.py` (lines 59-69, 108) |
| **Type** | Ghost parameter - documented but doesn't exist |

#### What Was Changed

Added documentation for a parameter `n_init` that does not exist in the implementation:

**Added** (lines 37-40):
```
    n_init : int, optional (default = 10)
        Number of times the k-means algorithm will be run with different
        centroid seeds. The final result will be the best output of ``n_init``
        consecutive runs in terms of inertia.
```

#### Why It's Wrong

The `__init__` method only accepts `max_clusters` and `random_state`:

**In `__init__` (lines 59-69):**
```python
def __init__(
    self,
    max_clusters: int = 2,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
):
    # ... validation code ...
    self.max_clusters = max_clusters
    self.random_state = random_state
```

There is no `n_init` parameter. Users trying to use it will get:
```
TypeError: __init__() got an unexpected keyword argument 'n_init'
```

#### Example

```python
from graspologic.cluster import KMeansCluster
import numpy as np

X = np.random.randn(100, 5)

# Following the documentation...
gc = KMeansCluster(max_clusters=5, n_init=20)  # TypeError!
# TypeError: __init__() got an unexpected keyword argument 'n_init'

# What actually works:
gc = KMeansCluster(max_clusters=5)
gc.fit(X)
```

#### Key Implementation Details

- This is a "ghost" parameter - documented but doesn't exist
- Simulates outdated documentation after a feature was removed
- sklearn's `KMeans` has this parameter, making the documentation look authentic
- Users familiar with sklearn would expect this parameter to exist
- Error message clearly indicates the parameter doesn't exist, but only after trying to use it"""


STEPS_CHECK_DISCREPANCIES = """The purpose of this task is to verify whether the agent could correctly verify a set of 5 discrepancies in the documentation that were injected versus the actual implementation/behavior of the code.
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
    "discrepancy_1_bic_column_order": "str - (24 points) Did the agent identify that GaussianCluster's bic_ DataFrame \
columns are reordered to a fixed internal order ['spherical', 'diag', 'tied', 'full'], NOT the user-specified order? \
Award 24 points if correctly identified, 0 if missed.",
    "discrepancy_2_svd_seed_none": "str - (24 points) Did the agent identify that select_svd converts svd_seed=None \
to 0, making results deterministic (NOT using system entropy for non-deterministic results as the docstring claims)? \
Award 24 points if correctly identified, 0 if missed.",
    "discrepancy_3_initial_p_sums": "str - (24 points) Did the agent identify that SeedlessProcrustes initial_P \
docstring has row/column sums swapped (implementation requires rows sum to 1/n and cols sum to 1/m, not the opposite)? \
Award 24 points if correctly identified, 0 if missed.",
    "discrepancy_4_n_init_ghost": "str - (24 points) Did the agent identify that KMeansCluster's n_init parameter \
is documented but does not actually exist in the implementation (ghost parameter)? \
Award 24 points if correctly identified, 0 if missed.",
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
