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
- optional: `LINEAR_CODE_CANDIDATE_WORKERS`
- optional: `LINEAR_CODE_RESTART_WORKERS`
- optional: `LINEAR_CODE_SEARCH_MODE=exact|sampled`
- optional sampled mode controls:
  - `LINEAR_CODE_SAMPLE_BUDGET`
  - `LINEAR_CODE_SAMPLE_SEED`
  - `LINEAR_CODE_STRATA_PER_WEIGHT`
  - `LINEAR_CODE_SAMPLE_OVERSAMPLE_FACTOR`

If you do not set them, the default target is `[8,4,4]_2`.

The greedy fill itself stays sequential because each accepted column updates the exact forbidden-state. The new worker settings only parallelize:

- candidate scoring inside one restart,
- and evaluation across independent restart indices.

For large instances, `LINEAR_CODE_SEARCH_MODE=sampled` enables an approximate path
that samples candidates by Hamming-weight strata before scoring and sorting them.
This avoids full candidate ranking and temporary run files, but it does not
reproduce the exact full-order greedy result.

## Files

- `initial_program.py`: baseline priority heuristic, with a single EVOLVE-BLOCK.
- `search_core.py`: fixed legality checks, single-instance loader, greedy skeleton, and exact validation helpers.
- `evaluator.py`: thin OpenEvolve adapter.
- `config.yaml`: evolution config tuned for deterministic heuristic search.
- `verify_distance.py`: prints the constructed `H`, the derived `G`, and the achieved `d`.
- `run_batch.py`: sweeps many `(n,k,d)` instances from `Misc/ECCRecord.json` into separate output directories.

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

To try the sampled search path on a large verification run:

```bash
LINEAR_CODE_N=50 LINEAR_CODE_K=20 LINEAR_CODE_D=13 \
LINEAR_CODE_SEARCH_MODE=sampled \
LINEAR_CODE_SAMPLE_BUDGET=2000000 \
LINEAR_CODE_RESTARTS=64 \
LINEAR_CODE_CANDIDATE_WORKERS=16 \
LINEAR_CODE_CANDIDATE_EXECUTOR=process \
python verify_distance.py --no-progress --target-only path/to/best_program.py
```

The `--target-only` flag skips exhaustive `d_actual` enumeration and reports the
target-distance certificate from the greedy construction.

To batch-run all valid `(n,k)` entries with `10 < n <= 40`, using `d = lower` from `Misc/ECCRecord.json`:

```bash
python run_batch.py \
  --record Misc/ECCRecord.json \
  --n-min 11 \
  --n-max 40 \
  --d-field lower \
  --iterations 40 \
  --output-root batch_runs
```

The batch runner automatically skips instances whose selected target distance is `d <= 2`. Those entries are not searched, but they are still written to the summary files as skipped rows.

Each instance is written to its own directory such as `batch_runs/n18_k7_d7/`, including:

- the OpenEvolve output,
- `resolved_config.yaml`,
- `run_metadata.json`,
- `matrix_verification.txt`,
- and root-level `summary.jsonl` / `summary.csv`.
