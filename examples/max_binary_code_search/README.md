# Binary Maximum-Code Search

This example uses OpenEvolve to optimize a FunSearch-style scoring function for
the binary maximum-code problem.

Given `n` and `d`, the goal is to find a large binary code `C` such that every
two distinct codewords have Hamming distance at least `d`. In coding-theory
notation this searches for good lower bounds on `A(n,d)`.

## Search Setup

Only the priority function in `initial_program.py` is evolved:

```python
priority(
    word_mask: int,
    n: int,
    d: int,
    step: int = 0,
    word_weight: int | None = None,
    new_forbidden_count: int = 0,
    overlap_forbidden_count: int = 0,
    local_available_count: int = 0,
) -> float
```

For target distance `d = 4`, the fixed evaluator uses

```text
A_2(n,4) = A_2(n-1,3)
```

It searches length `n - 1` centers with minimum distance `3`, then appends an
overall parity bit to each center to get a length `n`, distance `4` code.

The evaluator:

- forces the all-zero center into the internal code;
- samples currently legal candidate centers;
- computes dynamic damage features for each candidate;
- sorts sampled pools by `priority(...)`;
- greedily accepts candidates whose distance from the internal code is at least
  `3`;
- parity-extends the result and reports the length-`n` distance-`4` code.

For non-`d=4` targets, the evaluator falls back to the original full static
greedy path.

The transformed legality engine maintains exact forbidden centers

```text
forbidden_centers = union_{c in C} {c xor e : weight(e) <= 2}
```

so `can_add(x)` is an O(1) set lookup after each accepted center updates its
radius-2 forbidden region.

## Interface

The evaluator reads the target instance from environment variables:

- `MAX_CODE_N`
- `MAX_CODE_D`
- optional: `MAX_CODE_RESTARTS`
- optional: `MAX_CODE_MAX_CANDIDATES`
- optional: `MAX_CODE_PARITY_POOL_SIZE`
- optional: `MAX_CODE_PARITY_ATTEMPTS_PER_REFILL`
- optional: `MAX_CODE_PARITY_MAX_REFILLS`
- optional: `MAX_CODE_PARITY_MAX_STALE_REFILLS`
- optional: `MAX_CODE_LOCAL_SAMPLE_SIZE`

If unset, the default target is `A(17,4)` with four deterministic restarts.

`MAX_CODE_MAX_CANDIDATES` defaults to `2^22` as a safety guard for the stage-one
full-enumeration path. Later sampled/C-kernel stages should lift this limit by
avoiding complete enumeration.

## Run

```bash
cd examples/max_binary_code_search
MAX_CODE_N=17 MAX_CODE_D=4 \
python ../../openevolve-run.py initial_program.py evaluator.py --config Configs/config.yaml --iterations 40
```

To inspect the baseline without running evolution:

```bash
MAX_CODE_N=17 MAX_CODE_D=4 python initial_program.py
```

The evaluator returns these main metrics:

- `code_size`: verified number of codewords;
- `minimum_distance`: exact pairwise minimum distance of the final code;
- `combined_score`: primary OpenEvolve score, exactly equal to verified `code_size`.

Artifacts include JSON `codewords` and a detailed `search_result`.
