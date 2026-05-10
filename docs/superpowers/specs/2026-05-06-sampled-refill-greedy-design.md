# Sampled Search Design

## Goal

Improve the inner search for a fixed `(n, k, d)` instance when the full candidate
space is too large to enumerate, score, and sort. The goal is to find a feasible
binary linear code quickly while preserving the existing full-sort greedy path as
a baseline for small instances and regression tests.

## Approach

Add opt-in sampled search modes controlled by `LINEAR_CODE_SEARCH_MODE`. Each
restart owns a deterministic random seed derived from `LINEAR_CODE_RANDOM_SEED`
and the restart index.

In `sampled_refill` mode, one restart:

1. Maintain the existing exact `IncrementalForbiddenState`.
2. Repeatedly sample candidate masks from Hamming-weight layers with probability
   proportional to `C(r, w)`.
3. Reject duplicates and candidates currently in the forbidden set.
4. Score only legal sampled candidates with the existing
   `priority(column_mask, n, k, d)` function.
5. Sort the small refill pool and greedily accept legal columns.
6. Refill again when the current pool cannot add more columns.
7. On repeated no-progress refills, backtrack a bounded number of recent columns
   and add them to a restart-local tabu set.

In `sampled_beam` mode, one restart keeps several partial constructions. Each
beam state samples legal extensions, scores them with static priority minus a
small forbidden-growth penalty, and the restart keeps the best beam states for
the next depth.

This avoids materializing the full candidate list. The expensive forbidden update
still happens exactly when a column is accepted, so correctness of accepted
columns remains unchanged.

## Parameters

- `LINEAR_CODE_SAMPLE_POOL_SIZE`: target legal sampled candidates per refill.
- `LINEAR_CODE_SAMPLE_ATTEMPTS_PER_REFILL`: random draws allowed for one refill.
- `LINEAR_CODE_SAMPLE_MAX_REFILLS`: maximum refill rounds per restart.
- `LINEAR_CODE_SAMPLE_MAX_STALE_REFILLS`: no-progress refills before abandoning a restart.
- `LINEAR_CODE_RANDOM_SEED`: base seed for reproducibility.
- `LINEAR_CODE_BACKTRACK_DEPTH`: recent columns removed when sampled refill stalls.
- `LINEAR_CODE_BACKTRACK_MAX_EVENTS`: maximum sampled-refill backtracking events.
- `LINEAR_CODE_BEAM_WIDTH`: beam states kept per sampled beam restart.
- `LINEAR_CODE_BEAM_BRANCHES_PER_STATE`: legal branches kept from each beam state.
- `LINEAR_CODE_BEAM_ATTEMPTS_PER_STATE`: random draws used to expand one beam state.
- `LINEAR_CODE_BEAM_FORBIDDEN_PENALTY`: penalty applied to forbidden-set growth.

## Artifacts

Existing artifacts remain present. `search_result` additionally records
`search_mode`, `sample_attempts`, `sampled_candidates`, `scored_candidates`,
`backtrack_events`, `backtracked_columns`, `beam_width`, and
`beam_expanded_states`.
Successful vector records include `rank_scope`; in full mode ranks are global,
while sampled refill and beam ranks are local to the pool that accepted or
expanded the vector.

## Testing

Keep existing full-mode tests unchanged. Add tests that sampled mode:

- solves a small deterministic instance;
- scores fewer candidates than full scoring would across all restarts;
- emits sampled rank scope in successful-vector artifacts;
- is reproducible for a fixed seed and budget.
- uses exact binomial layer counts for weight sampling;
- can backtrack and still emit final-path-only successful vectors;
- solves a small instance in sampled beam mode.
