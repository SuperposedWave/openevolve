# Binary Linear Code Feasibility Search

This example uses OpenEvolve to optimize a `priority(candidate_mask, state)` function for constructing small binary linear codes with high minimum distance.

## Problem Setup

For a binary `[n,k,d]` code with redundancy `r = n-k`, the evaluator works with a systematic parity-check matrix

`H = [P^T | I_r]`

and asks for `k` free columns in `F_2^r` such that every set of `d-1` columns in `H` is linearly independent.

The fixed search skeleton:

- enumerates free columns as integer bitmasks,
- filters out columns with weight `< d-1`,
- maintains exact forbidden xor layers for subsets of size up to `d-2`,
- runs deterministic greedy search over a small benchmark suite.

Only the `priority()` function inside `initial_program.py` is evolved.

## Benchmark Suite

The evaluator uses a fixed suite of exact yes-instances:

- `[5,2,3]_2`
- `[6,3,3]_2`
- `[7,3,4]_2`
- `[7,4,3]_2`
- `[8,4,4]_2`
- `[9,4,4]_2`

Each target distance is validated by a small brute-force exact search inside the helper module.

## Files

- `initial_program.py`: baseline priority heuristic, with a single EVOLVE-BLOCK.
- `search_core.py`: fixed legality checks, benchmark catalog, greedy skeleton, and exact validation helpers.
- `evaluator.py`: thin OpenEvolve adapter.
- `config.yaml`: evolution config tuned for deterministic heuristic search.

## Run

```bash
cd examples/linear_code_binary_search
python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml --iterations 40
```

To inspect the baseline without running evolution:

```bash
python initial_program.py
```
