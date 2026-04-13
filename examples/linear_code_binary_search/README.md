# Binary Linear Code Feasibility Search

This example uses OpenEvolve to optimize a FunSearch-style static scoring function `priority(column_mask, n, k, d)` for constructing one binary matrix instance at a time.

## Problem Setup

For a binary `[n,k,d]` code with redundancy `r = n-k`, the evaluator works with a systematic parity-check matrix

`H = [P^T | I_r]`

and asks for `k` free columns in `F_2^r` such that every set of `d-1` columns in `H` is linearly independent.

The fixed search skeleton:

- enumerates free columns as integer bitmasks,
- filters out columns with weight `< d-1`,
- scores every candidate column exactly once,
- sorts the full candidate list by that static score,
- maintains exact forbidden xor layers for subsets of size up to `d-2`,
- greedily scans the sorted list and keeps every legal column it can add.

Only the static `priority()` function inside `initial_program.py` is evolved.

## Single-Instance Interface

The evaluator reads the target instance from environment variables:

- `LINEAR_CODE_N`
- `LINEAR_CODE_K`
- `LINEAR_CODE_D`
- optional: `LINEAR_CODE_RESTARTS`

If you do not set them, the default target is `[8,4,4]_2`.

The helper module also computes the exact best distance for the chosen `(n,k)` when the instance is small enough, so the artifacts show whether your requested `d` is optimal or too ambitious.

## Files

- `initial_program.py`: baseline priority heuristic, with a single EVOLVE-BLOCK.
- `search_core.py`: fixed legality checks, single-instance loader, greedy skeleton, and exact validation helpers.
- `evaluator.py`: thin OpenEvolve adapter.
- `config.yaml`: evolution config tuned for deterministic heuristic search.

## Method

This is intentionally closer to the FunSearch cap set pattern than to an interactive search policy:

- the LLM does not choose columns one by one using search state,
- the LLM only writes a static scoring function for a single candidate column,
- the fixed evaluator handles ordering, legality checking, and final construction.

## Run

```bash
cd examples/linear_code_binary_search
LINEAR_CODE_N=8 LINEAR_CODE_K=4 LINEAR_CODE_D=4 \
python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml --iterations 40
```

To inspect the baseline without running evolution:

```bash
LINEAR_CODE_N=7 LINEAR_CODE_K=4 LINEAR_CODE_D=3 python initial_program.py
```
