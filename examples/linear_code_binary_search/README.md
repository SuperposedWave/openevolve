# Binary Linear Code Feasibility Search

This example uses OpenEvolve to optimize a C scoring function for constructing
one binary matrix instance at a time. The Python search path has been removed;
the evaluator now accepts C priority files only.

## Problem Setup

For a binary `[n,k,d]` code with redundancy `r = n-k`, the evaluator works with a systematic parity-check matrix

`H = [P^T | I_r]`

and asks for `k` free columns in `F_2^r` such that every set of `d-1` columns in `H` is linearly independent.

The fixed C search skeleton:

- enumerates free columns as integer bitmasks,
- filters out columns with weight `< d-1`,
- scores every candidate column once to build a static ordering,
- reranks a bounded window of currently legal candidates with dynamic features,
- maintains exact forbidden xor layers for subsets of size up to `d-2`,
- accepts the best legal dynamic candidate,
- and can repair a stuck partial construction by dropping selected columns that
  release the most search space.

Only `oe_linear_code_priority()` inside `initial_program.c` is evolved.

## Single-Instance Interface

The evaluator reads the target instance from environment variables:

- `LINEAR_CODE_N`
- `LINEAR_CODE_K`
- `LINEAR_CODE_D`
- optional: `LINEAR_CODE_RESTARTS`
- optional: `LINEAR_CODE_RANDOM_SEED`
- optional: `LINEAR_CODE_DYNAMIC_WINDOW`
- optional: `LINEAR_CODE_REPAIR_EVENTS`
- optional: `LINEAR_CODE_REPAIR_DROP_COUNT`
- optional: `LINEAR_CODE_REPAIR_CANDIDATE_WINDOW`
- optional: `LINEAR_CODE_REPAIR_TABU_TENURE`
- optional: `LINEAR_CODE_REPAIR_MODE`
- optional: `LINEAR_CODE_REPAIR_MCTS_SIMULATIONS`
- optional: `LINEAR_CODE_REPAIR_MCTS_DEPTH`
- optional: `LINEAR_CODE_REPAIR_MCTS_WORKERS`
- optional: `LINEAR_CODE_REPAIR_MCTS_DROP_TOPK`
- optional: `LINEAR_CODE_C_COMPILE_TIMEOUT`
- optional: `LINEAR_CODE_C_RUN_TIMEOUT`

If you do not set them, the default target is `[8,4,4]_2`.

Useful C-kernel controls:

- `LINEAR_CODE_RANDOM_SEED`: base seed for reproducible restarts.
- `LINEAR_CODE_DYNAMIC_WINDOW`: C-kernel legal-candidate window reranked with
  dynamic forbidden-growth features; use `0` for the old static sorted-greedy path.
- `LINEAR_CODE_REPAIR_EVENTS`: C-kernel repair events per restart. When stuck,
  the skeleton drops a selected column that releases the most search space, then
  rebuilds the exact state and continues. Use `0` to disable.
- `LINEAR_CODE_REPAIR_DROP_COUNT`: selected columns dropped per repair event.
- `LINEAR_CODE_REPAIR_CANDIDATE_WINDOW`: sorted candidate prefix used to estimate
  how many candidates a possible drop would release.
- `LINEAR_CODE_REPAIR_TABU_TENURE`: recently dropped columns kept out of refill.
- `LINEAR_CODE_REPAIR_MODE`: `greedy` by default, or `mcts` for bounded local
  Monte Carlo repair when the dynamic fill gets stuck.
- `LINEAR_CODE_REPAIR_MCTS_SIMULATIONS`: total root-drop rollouts used by MCTS repair.
- `LINEAR_CODE_REPAIR_MCTS_DEPTH`: maximum local rollout actions after the first drop.
- `LINEAR_CODE_REPAIR_MCTS_WORKERS`: pthread worker count for parallel MCTS
  rollouts. The default is `1` to preserve old behavior; use `0` to auto-select
  CPU count, capped internally at 64 and by the simulation count.
- `LINEAR_CODE_REPAIR_MCTS_DROP_TOPK`: when an MCTS rollout gets stuck after the
  first drop, rank possible follow-up drops by released search space and sample
  from the top-k choices. The default is `2`; use `0` for the old random drop.
- `LINEAR_CODE_C_COMPILE_TIMEOUT`: compilation timeout in seconds.
- `LINEAR_CODE_C_RUN_TIMEOUT`: isolated C-kernel runtime timeout in seconds.

## Files

- `initial_program.c`: baseline semi-dynamic C priority heuristic, with a single EVOLVE-BLOCK.
- `c_search_skeleton.c`: fixed C candidate enumeration, static ordering, dynamic window reranking, legality checks, and ABI entry point.
- `c_kernel_runner.py`: compiles `initial_program.c` with `c_search_skeleton.c` and reads C metrics/output.
- `search_core.py`: shared instance parsing, exact validation helpers, and matrix formatting.
- `evaluator.py`: C-only OpenEvolve adapter.
- `Configs/config_c_kernel.yaml`: evolution config for C priority-only experiments.
- `Configs/config_c_kernel_large.yaml`: larger C-kernel evolution config.
- `Configs/llm_config.yaml`: shared LLM/provider configuration used by the configs above.
- `verify_distance.py`: prints the constructed `H`, the derived `G`, and the achieved `d`.
- `run_batch.py`: sweeps many `(n,k,d)` instances from `Misc/ECCRecord.json` into separate output directories.

## Method

This is intentionally closer to the FunSearch cap set pattern than to an
interactive search policy:

- the LLM does not choose columns one by one using search state,
- the LLM only writes a C scoring function for a single candidate column,
- the fixed evaluator handles ordering, legality checking, and final construction.

The fixed skeleton first builds a static ordering, then reranks a bounded window
of currently legal candidates before each accepted column. The evolved C
priority receives dynamic damage features such as current forbidden-set size and
the candidate's exact new forbidden growth. If the dynamic fill gets stuck, the
fixed skeleton can run a repair step: it evaluates which selected column releases
the most high-ranked candidates / forbidden-set mass, drops that column, rebuilds
the exact state, and continues filling. With `LINEAR_CODE_REPAIR_MODE=mcts`,
that drop choice is replaced by a bounded local rollout search. Root drops are
compared with aggregated rollout statistics, and stuck follow-up rollout drops
can be sampled from a small heuristic top-k rather than uniformly at random. The
rollouts can be evaluated in parallel with `LINEAR_CODE_REPAIR_MCTS_WORKERS`.
The rollout objective is lexicographic: construct more columns first, then
prefer fewer drops, fewer rollout steps, and a smaller forbidden set.

## Run

```bash
cd examples/linear_code_binary_search
LINEAR_CODE_N=8 LINEAR_CODE_K=4 LINEAR_CODE_D=4 \
python ../../openevolve-run.py initial_program.c evaluator.py --config Configs/config_c_kernel.yaml --iterations 40
```

To inspect the baseline without running evolution:

```bash
LINEAR_CODE_N=7 LINEAR_CODE_K=4 LINEAR_CODE_D=3 python verify_distance.py initial_program.c
```

Single-run inspection writes the constructed parity-check and generator matrices
to `matrix_verification.txt` by default. Override the path with
`LINEAR_CODE_MATRIX_OUTPUT`; exhaustive distance reporting is skipped when the
filled dimension exceeds `LINEAR_CODE_MATRIX_MAX_EXHAUSTIVE_K` whose default is
`24`.

The evaluator also returns matrix artifacts directly:

- `parity_check_matrix`: JSON list of `H` row strings.
- `generator_matrix`: JSON list of `G` row strings.
- `matrix_summary`: JSON metadata with matrix form, shapes, and selected columns.

To batch-run all valid `(n,k)` entries with `10 < n <= 40`, using `d = lower` from `Misc/ECCRecord.json`:

```bash
python run_batch.py \
  --record Misc/ECCRecord.json \
  --n-min 11 \
  --n-max 40 \
  --d-field lower \
  --iterations 40 \
  --output-root outputs
```

The batch runner automatically skips instances whose selected target distance is `d <= 2`. Those entries are not searched, but they are still written to the summary files as skipped rows.

Each instance is written under a target directory such as `outputs/n18_k7_d7/batch_20260629T120000Z/`, including:

- the OpenEvolve output,
- `resolved_config.yaml`,
- `run_metadata.json`,
- `matrix_verification.txt`,
- and batch-level `outputs/_summaries/<run-name>/summary.jsonl` / `summary.csv`.

## Persistent Code Table Records

The static viewer still reads `code_table_viewer/code_table_data.json`, but
ongoing experiment records can be kept in SQLite with `code_table_db.py`.

Initialize the local record store from the current viewer JSON:

```bash
python code_table_db.py import-viewer \
  --db code_table_records.sqlite \
  --input code_table_viewer/code_table_data.json
```

After new experiments finish under `outputs/`, import the new runs and refresh
the viewer JSON:

```bash
python code_table_db.py import-runs \
  --db code_table_records.sqlite \
  --search-root outputs

python code_table_db.py export-viewer \
  --db code_table_records.sqlite \
  --output code_table_viewer/code_table_data.json
```

For a clean rebuild from `Misc/ECCRecord.json` plus run directories:

```bash
python code_table_db.py rebuild-viewer \
  --db code_table_records.sqlite \
  --record Misc/ECCRecord.json \
  --search-root outputs \
  --output code_table_viewer/code_table_data.json
```

The database file is ignored by git, so it can grow as a local experiment log
while the exported JSON remains the portable artifact for the browser viewer.

`run_batch.py` updates `code_table_records.sqlite` and refreshes the viewer JSON
after each completed instance by default. Pass `--no-sqlite-update` to disable
that for a temporary run.

The shell scripts in `Scripts/` source `Scripts/record_sqlite.sh` and call it
after OpenEvolve finishes. Set `LINEAR_CODE_SQLITE_RECORD=0` to skip the automatic
record update, or override `LINEAR_CODE_SQLITE_DB`, `LINEAR_CODE_VIEWER_JSON`, and
`LINEAR_CODE_RECORD_PATH` when writing to a different local record store.

Best-program saves include evaluator artifacts when available, including
`search_result`, `matrix_summary`, `parity_check_matrix`, and
`generator_matrix`. The SQLite importer uses those saved matrix artifacts, or an
existing `matrix_verification.txt`, to populate the viewer without rerunning the
search. Set `LINEAR_CODE_VERIFY_ON_RECORD=1` only when you explicitly want the
recorder to run `verify_distance.py` and write a fresh `matrix_verification.txt`.
