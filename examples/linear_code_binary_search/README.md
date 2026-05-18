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
- optional: `LINEAR_CODE_SEARCH_MODE` (`full` by default, `sampled_refill`, or `sampled_beam`)
- optional: `LINEAR_CODE_PROGRESS` (`1` enables restart progress and sampled-step output)
- optional: `LINEAR_CODE_LEGALITY_ENGINE` (`python` by default, or `native`)
- optional: `LINEAR_CODE_CANDIDATE_WORKERS`
- optional: `LINEAR_CODE_RESTART_WORKERS`

If you do not set them, the default target is `[8,4,4]_2`.

The default `full` mode scores every candidate and sorts the complete candidate
list. `sampled_refill` mode avoids full enumeration: each restart samples
candidate pools from Hamming-weight layers proportional to `C(r, w)`, filters
them through the current exact forbidden state, scores only legal sampled
candidates, and greedily accepts from that small pool before refilling.
`sampled_beam` keeps several partial constructions per restart and expands them
with the same sampled legality checks.

For larger exact-search runs, `LINEAR_CODE_LEGALITY_ENGINE=native` switches the
forbidden-state engine to the optional CPython C extension. Build it first:

```bash
python setup.py build_ext --inplace
```

The native engine is explicit opt-in, supports `r <= 60`, and fails loudly if it
is requested without a built extension. It stores exact reachable/forbidden
membership in sparse C hash sets and accelerates `can_add`, `add`, `undo`, and
beam-state `clone` operations while leaving sampling, priority scoring, and beam
ranking in Python. It still exactly materializes the low-weight reachable layers;
for large `r,d`, initialization can be intrinsically too large. The safety guard
`LINEAR_CODE_NATIVE_MAX_INITIAL_VALUES` defaults to `200000000`.

Useful sampled-search controls:

- `LINEAR_CODE_RANDOM_SEED`: base seed for reproducible randomized restarts.
- `LINEAR_CODE_SAMPLE_POOL_SIZE`: target legal sampled candidates per refill.
- `LINEAR_CODE_SAMPLE_ATTEMPTS_PER_REFILL`: random draws allowed for one refill.
- `LINEAR_CODE_SAMPLE_MAX_REFILLS`: maximum refills per restart.
- `LINEAR_CODE_SAMPLE_MAX_STALE_REFILLS`: no-progress refills before abandoning a restart.
- `LINEAR_CODE_BACKTRACK_DEPTH`: recent columns removed when sampled refill stalls.
- `LINEAR_CODE_BACKTRACK_MAX_EVENTS`: maximum sampled-refill backtracking events per restart.
- `LINEAR_CODE_BEAM_WIDTH`: number of partial constructions kept in sampled beam mode.
- `LINEAR_CODE_BEAM_BRANCHES_PER_STATE`: legal branches kept from each beam state.
- `LINEAR_CODE_BEAM_ATTEMPTS_PER_STATE`: random draws used to expand each beam state.
- `LINEAR_CODE_BEAM_FORBIDDEN_PENALTY`: penalty for growing the exact forbidden set.
- `LINEAR_CODE_LEGALITY_ENGINE`: use `native` to enable the C exact-legality engine.
- `LINEAR_CODE_NATIVE_MAX_INITIAL_VALUES`: safety cap for exact native initialization.
- `LINEAR_CODE_PROGRESS`: show a restart progress bar plus per-refill sampled-search steps.

The greedy fill itself stays sequential because each accepted column updates the exact forbidden-state. The worker settings only parallelize:

- candidate scoring inside one restart,
- and evaluation across independent restart indices.

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
