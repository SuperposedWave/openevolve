# Generator-Matrix Binary Linear Code Search

This experimental example explores binary `[n,k,d]` code construction from the
systematic generator-matrix view

```text
G = [I_k | p_1 p_2 ... p_r],  r = n-k
```

Each parity column `p_j` is a `k`-bit vector. Instead of maintaining forbidden
xor layers over parity-check columns, this prototype maintains message deficits:

```text
deficit[m] = d - wt(m) - parity_hits(m)
```

for every non-zero message `m` with `wt(m) < d`. Adding a parity column covers a
message when `dot(m, p_j) = 1`, decreasing that message's deficit by one. The
construction succeeds when every deficit is non-positive and exact generator
verification confirms `d_actual >= d`.

## Why This Folder Exists

The sibling `examples/linear_code_binary_search` project is the current
parity-check route based on

```text
H = [P^T | I_r]
```

and a C kernel that chooses `k` free columns in `F_2^r`. This folder keeps the
generator-matrix route separate: it chooses `r` parity columns in `F_2^k` and
uses deficit multicover state. That separation makes the benchmark and future
LLM-priority experiments easier to reason about.

## Files

- `initial_program.py`: baseline Python priority function with one EVOLVE-BLOCK.
- `evaluator.py`: OpenEvolve adapter that loads `priority()` and returns metrics
  plus matrix artifacts.
- `search_core.py`: fixed generator-column skeleton, exact verification,
  metrics, artifacts, and the CLI implementation.
- `verify_distance.py`: single-program inspection tool that prints metrics,
  `G`, `H`, and `d_actual`.
- `render_config.py`: writes a resolved OpenEvolve config whose prompt contains
  the active `GEN_MATRIX_CODE_N/K/D` target.
- `Configs/config.yaml`: default OpenEvolve config for quick experiments.
- `Configs/config_large.yaml`: larger evolution config for harder targets.
- `Configs/llm_config.yaml`: provider template using `OPENAI_API_KEY`.
- `Scripts/run.sh`: OpenEvolve launch helper.
- `Scripts/run_benchmark.sh`: deterministic benchmark helper.
- `tests/test_generator_matrix_code_search.py`: regression tests for this
  example.

## Fixed Skeleton

`search_core.py` provides:

- `[n,k,d]` instance validation;
- dense low-weight message deficits;
- full `2^k` candidate coverage scoring with a Walsh-Hadamard transform;
- hard feasibility filtering: after a candidate is added, no message may need
  more future covers than the number of remaining parity columns;
- `feasible_next_count` lookahead: an estimate of how many next-step columns
  remain hard-feasible if the current candidate is accepted;
- shortlist reranking with row-balance, duplicate, correlation, and critical
  message features;
- optional hook for an evolved `priority_fn`;
- exact exhaustive verification of `min_{m != 0} wt(mG)` for `k <= 24`;
- JSON artifacts for `G`, `H`, per-step diagnostics, and benchmark metrics.

The Walsh-Hadamard scoring computes

```text
score[p] = sum deficit[m] for dot(m,p)=1
```

for every candidate `p` at once. For the key `[50,20,13]` experiment this makes
the full `2^20` column space cheap enough for Python-level prototyping.

## Evaluation Signal

The primary OpenEvolve metric is `combined_score`. With hard feasibility
filtering enabled, incomplete searches are scored column-first:

```text
success:
    combined_score = 1.0

otherwise:
    constructed_columns / target_columns is the main term
    coverage_progress only breaks ties within the same column count
```

This means a priority that reaches 20/30 columns should outrank one that reaches
19/30 columns, even if the 19-column partial matrix has prettier local deficit
statistics. `d_actual` is still returned as a diagnostic, but it is not the main
fitness signal for partial constructions.

## Interface

The evaluator reads the target instance and search controls from environment
variables:

- `GEN_MATRIX_CODE_N`
- `GEN_MATRIX_CODE_K`
- `GEN_MATRIX_CODE_D`
- optional: `GEN_MATRIX_CODE_RESTARTS`
- optional: `GEN_MATRIX_CODE_SHORTLIST_SIZE`
- optional: `GEN_MATRIX_CODE_RANDOM_POOL_SIZE`
- optional: `GEN_MATRIX_CODE_RANDOM_SEED`
- optional: `GEN_MATRIX_CODE_MAX_K`
- optional: `GEN_MATRIX_CODE_MAX_LOW_MESSAGES`
- optional: `GEN_MATRIX_CODE_EXACT_VERIFY_MAX_K`
- optional scoring weights:
  `GEN_MATRIX_CODE_DEFICIT_WEIGHT`,
  `GEN_MATRIX_CODE_PRESSURE_WEIGHT`,
  `GEN_MATRIX_CODE_CRITICAL_WEIGHT`,
  `GEN_MATRIX_CODE_ROW_NEED_WEIGHT`,
  `GEN_MATRIX_CODE_ROW_BALANCE_WEIGHT`,
  `GEN_MATRIX_CODE_DUPLICATE_PENALTY`,
  `GEN_MATRIX_CODE_CORRELATION_PENALTY`

If unset, the default target is `[20,10,5]`.

The priority function interface is:

```python
priority(
    column_mask,
    n,
    k,
    d,
    step,
    column_weight,
    covered_deficit_sum,
    covered_critical_count,
    uncovered_critical_count,
    feasible_next_count,
    min_row_weight_after,
    max_row_weight_after,
    avg_pair_balance_after,
)
```

Only code inside `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END` in
`initial_program.py` should be evolved.

## Run Search

From the repository root:

```bash
python examples/generator_matrix_code_search/search_core.py \
  --N 20 --K 10 --D 5
```

From this folder:

```bash
python search_core.py --N 20 --K 10 --D 5
```

Run the first benchmark set for this coordinate system:

```bash
python search_core.py --benchmark --shortlist-size 2048
```

Emit full JSON, including matrices:

```bash
python search_core.py --N 5 --K 2 --D 3 --json
```

Try additional pressure terms for sweeps:

```bash
python search_core.py --N 50 --K 20 --D 13 \
  --shortlist-size 4096 \
  --pressure-weight 0.05 \
  --row-need-weight 5.0
```

## Run OpenEvolve

From this folder:

```bash
GEN_MATRIX_CODE_N=20 GEN_MATRIX_CODE_K=10 GEN_MATRIX_CODE_D=5 \
python ../../openevolve-run.py initial_program.py evaluator.py \
  --config Configs/config.yaml \
  --iterations 40
```

Or use the helper:

```bash
GEN_MATRIX_CODE_N=20 GEN_MATRIX_CODE_K=10 GEN_MATRIX_CODE_D=5 \
GEN_MATRIX_CODE_ITERATIONS=40 \
Scripts/run.sh
```

`Scripts/run.sh` renders `Configs/.resolved_config.yaml` before launching
OpenEvolve, so the LLM prompt contains the concrete `n`, `k`, `d`, and `r`
values from the environment. If you call `openevolve-run` directly with a config
file, edit the `Current target` block yourself or run `render_config.py` first.

For harder targets:

```bash
GEN_MATRIX_CODE_N=50 GEN_MATRIX_CODE_K=20 GEN_MATRIX_CODE_D=13 \
GEN_MATRIX_CODE_SHORTLIST_SIZE=4096 \
GEN_MATRIX_CODE_CONFIG=Configs/config_large.yaml \
GEN_MATRIX_CODE_ITERATIONS=80 \
Scripts/run.sh
```

## Verify

Inspect the baseline priority on one instance:

```bash
python verify_distance.py initial_program.py --N 20 --K 10 --D 5
```

Run the deterministic benchmark helper:

```bash
Scripts/run_benchmark.sh
```

## Early Benchmark Snapshot

With the conservative default scorer:

```text
[20,10,5]   success=true,  d_actual=5
[31,21,5]   success=false, d_actual=2
[50,20,13]  success=false, d_actual=2
```

The useful signal is cost, not final quality yet: the `[50,20,13]` default run
evaluates the full `k=20` candidate space in a few seconds. The next step is to
improve the priority policy and add incremental repair.
