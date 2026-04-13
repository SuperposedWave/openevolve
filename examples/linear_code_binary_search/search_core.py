"""FunSearch-style static-priority search for binary matrix construction."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import uuid
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

try:
    from openevolve.evaluation_result import EvaluationResult
except Exception:
    _EVAL_RESULT_PATH = (
        Path(__file__).resolve().parents[2] / "openevolve" / "evaluation_result.py"
    )
    _EVAL_RESULT_SPEC = importlib.util.spec_from_file_location(
        "openevolve_evaluation_result_fallback",
        _EVAL_RESULT_PATH,
    )
    if _EVAL_RESULT_SPEC is None or _EVAL_RESULT_SPEC.loader is None:
        raise ImportError("Failed to load EvaluationResult fallback")
    _EVAL_RESULT_MODULE = importlib.util.module_from_spec(_EVAL_RESULT_SPEC)
    _EVAL_RESULT_SPEC.loader.exec_module(_EVAL_RESULT_MODULE)
    EvaluationResult = _EVAL_RESULT_MODULE.EvaluationResult


PriorityFn = Callable[[int, int, int, int], float]


@dataclass(frozen=True)
class BenchmarkInstance:
    """Single binary feasibility instance for a systematic parity-check search."""

    name: str
    n: int
    k: int
    target_distance: int
    restarts: int = 3
    description: str = ""

    @property
    def r(self) -> int:
        return self.n - self.k


@dataclass
class SearchAttemptResult:
    """Result of one deterministic sorted-greedy run."""

    success: bool
    selected_free_columns: Tuple[int, ...]
    added_free_columns: int
    restart_index: int
    sorted_candidates: Tuple[int, ...]
    sorted_scores: Tuple[Tuple[int, float], ...]
    blocked_candidate_count: int
    illegal_weight_histogram: Tuple[Tuple[int, int], ...]
    chosen_weights: Tuple[int, ...]


class IncrementalForbiddenState:
    """Maintains exact low-order xor layers for binary legality checks."""

    def __init__(self, r: int, distance: int):
        self.r = r
        self.distance = distance
        self.max_subset_size = max(distance - 2, 0)
        self.reachable = _initialize_reachable_layers(r, distance)
        self.forbidden = set().union(*self.reachable)
        self.selected_free_columns: List[int] = []

    def can_add(self, column_mask: int) -> bool:
        return column_mask not in self.forbidden

    def add(self, column_mask: int) -> None:
        if not self.can_add(column_mask):
            raise ValueError(f"Illegal free column {column_mask}")
        _add_column_to_reachable(self.reachable, column_mask, self.max_subset_size)
        self.forbidden = set().union(*self.reachable)
        self.selected_free_columns.append(column_mask)


DEFAULT_INSTANCE = BenchmarkInstance(
    name="default_[8,4,4]",
    n=8,
    k=4,
    target_distance=4,
    restarts=3,
    description="Default single-instance target used by the example.",
)


def popcount(mask: int) -> int:
    """Return the Hamming weight of a binary mask."""
    return mask.bit_count()


@lru_cache(maxsize=None)
def basis_columns(r: int) -> Tuple[int, ...]:
    """Systematic identity columns."""
    return tuple(1 << bit_index for bit_index in range(r))


@lru_cache(maxsize=None)
def candidate_masks(r: int, distance: int) -> Tuple[int, ...]:
    """All non-zero free columns that pass the initial weight filter."""
    min_weight = max(distance - 1, 1)
    return tuple(mask for mask in range(1, 1 << r) if popcount(mask) >= min_weight)


def _initialize_reachable_layers(r: int, distance: int) -> List[set[int]]:
    """Build exact xor layers for the initial systematic columns."""
    max_subset_size = max(distance - 2, 0)
    reachable = [set() for _ in range(max_subset_size + 1)]
    reachable[0].add(0)
    for column_mask in basis_columns(r):
        _add_column_to_reachable(reachable, column_mask, max_subset_size)
    return reachable


def _add_column_to_reachable(
    reachable: List[set[int]], column_mask: int, max_subset_size: int
) -> None:
    """Incrementally update xor layers after accepting a new column."""
    for subset_size in range(max_subset_size, 0, -1):
        previous_layer = reachable[subset_size - 1]
        if previous_layer:
            reachable[subset_size].update(
                xor_value ^ column_mask for xor_value in previous_layer
            )


def rebuild_reachable_layers(r: int, distance: int, free_columns: Sequence[int]) -> List[set[int]]:
    """Recompute xor layers from scratch for validation and tests."""
    reachable = _initialize_reachable_layers(r, distance)
    max_subset_size = max(distance - 2, 0)
    for column_mask in free_columns:
        _add_column_to_reachable(reachable, column_mask, max_subset_size)
    return reachable


def forbidden_masks_from_layers(reachable: Sequence[set[int]]) -> set[int]:
    """Flatten xor layers into the exact forbidden set."""
    return set().union(*reachable)


def initial_forbidden_masks(r: int, distance: int) -> set[int]:
    """Initial forbidden set induced by the identity columns."""
    return forbidden_masks_from_layers(_initialize_reachable_layers(r, distance))


def all_columns(r: int, free_columns: Sequence[int]) -> Tuple[int, ...]:
    """All columns in the systematic parity-check matrix."""
    return tuple(free_columns) + basis_columns(r)


def columns_meet_distance_requirement(columns: Sequence[int], distance: int) -> bool:
    """Exact distance check via exhaustive xor tests up to distance - 1."""
    for subset_size in range(1, distance):
        for subset in combinations(columns, subset_size):
            xor_value = 0
            for column_mask in subset:
                xor_value ^= column_mask
            if xor_value == 0:
                return False
    return True


def validate_free_columns(r: int, free_columns: Sequence[int], distance: int) -> bool:
    """Validate a systematic construction independently of the greedy state."""
    return columns_meet_distance_requirement(all_columns(r, free_columns), distance)


def actual_minimum_distance_from_columns(columns: Sequence[int]) -> int:
    """Return the exact minimum distance induced by a parity-check matrix column set."""
    total_columns = len(columns)
    for subset_size in range(1, total_columns + 1):
        for subset in combinations(columns, subset_size):
            xor_value = 0
            for column_mask in subset:
                xor_value ^= column_mask
            if xor_value == 0:
                return subset_size
    return total_columns + 1


def actual_minimum_distance(r: int, free_columns: Sequence[int]) -> int:
    """Return the exact minimum distance for H = [P^T | I_r]."""
    return actual_minimum_distance_from_columns(all_columns(r, free_columns))


def exact_find_feasible_free_columns(r: int, k: int, distance: int) -> Tuple[int, ...] | None:
    """Exact brute-force witness search for small binary instances."""
    for free_columns in combinations(candidate_masks(r, distance), k):
        if validate_free_columns(r, free_columns, distance):
            return tuple(free_columns)
    return None


def exact_best_distance(n: int, k: int) -> Tuple[int, Tuple[int, ...]]:
    """Exact best-distance search for small binary instances."""
    r = n - k
    for distance in range(n, 1, -1):
        witness = exact_find_feasible_free_columns(r, k, distance)
        if witness is not None:
            return distance, witness
    return 1, tuple()


def make_instance(
    n: int,
    k: int,
    distance: int,
    restarts: int = 3,
    name: str | None = None,
) -> BenchmarkInstance:
    """Create and validate a single search instance."""
    if n <= 0 or k <= 0 or distance <= 0:
        raise ValueError("n, k, and d must be positive")
    if k >= n:
        raise ValueError("Require 0 < k < n")
    if distance > n:
        raise ValueError("Require d <= n")
    r = n - k
    if distance - 1 > r and k > 1:
        # This is not mathematically impossible in every case, but for this binary
        # systematic free-column search it leaves no eligible candidates beyond trivial cases.
        raise ValueError("Requested d is too large for this binary free-column search regime")
    return BenchmarkInstance(
        name=name or f"instance_[{n},{k},{distance}]",
        n=n,
        k=k,
        target_distance=distance,
        restarts=restarts,
    )


def instance_from_env(prefix: str = "LINEAR_CODE_") -> BenchmarkInstance:
    """Build one instance from environment variables."""
    n = int(os.environ.get(f"{prefix}N", DEFAULT_INSTANCE.n))
    k = int(os.environ.get(f"{prefix}K", DEFAULT_INSTANCE.k))
    distance = int(os.environ.get(f"{prefix}D", DEFAULT_INSTANCE.target_distance))
    restarts = int(os.environ.get(f"{prefix}RESTARTS", DEFAULT_INSTANCE.restarts))
    return make_instance(n=n, k=k, distance=distance, restarts=restarts)


def format_mask(mask: int, r: int) -> str:
    """Binary string representation with fixed width."""
    return format(mask, f"0{r}b")


def parity_check_matrix_rows(r: int, free_columns: Sequence[int]) -> Tuple[str, ...]:
    """Return the parity-check matrix as row strings over F_2."""
    columns = all_columns(r, free_columns)
    rows = []
    for row_index in range(r):
        row_bits = []
        for column_mask in columns:
            row_bits.append("1" if (column_mask >> row_index) & 1 else "0")
        rows.append("".join(row_bits))
    return tuple(rows)


def generator_matrix_rows(r: int, free_columns: Sequence[int]) -> Tuple[str, ...]:
    """Return the systematic generator matrix G = [I_k | P] as row strings over F_2."""
    k = len(free_columns)
    rows = []
    for row_index, column_mask in enumerate(free_columns):
        identity_bits = ["1" if i == row_index else "0" for i in range(k)]
        parity_bits = [
            "1" if (column_mask >> bit_index) & 1 else "0"
            for bit_index in range(r)
        ]
        rows.append("".join(identity_bits + parity_bits))
    return tuple(rows)


def _safe_priority(priority_fn: PriorityFn, candidate_mask: int, n: int, k: int, d: int) -> float:
    """Protect the fixed skeleton from bad static priority implementations."""
    try:
        value = priority_fn(candidate_mask, n, k, d)
    except Exception:
        value = popcount(candidate_mask)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = float(popcount(candidate_mask))
    if not math.isfinite(value):
        return float(popcount(candidate_mask))
    return value


def _deterministic_tiebreak(candidate_mask: int, restart_index: int) -> int:
    """Restart-specific tie-break used only to perturb equal-score columns."""
    return (
        candidate_mask * 1103515245
        + restart_index * 2654435761
        + 12345
    ) & 0xFFFFFFFF


def _iterate_with_progress(
    items: Sequence[int],
    description: str,
    show_progress: bool,
):
    """Yield items, optionally wrapped in a tqdm progress bar."""
    if not show_progress:
        return items

    try:
        from tqdm import tqdm
    except Exception:
        return items

    return tqdm(items, desc=description, leave=False)


def ranked_candidates(
    instance: BenchmarkInstance,
    priority_fn: PriorityFn,
    restart_index: int,
    show_progress: bool = False,
) -> Tuple[Tuple[int, ...], Tuple[Tuple[int, float], ...]]:
    """Compute a single static ordering of all candidate columns."""
    scored_candidates = []
    candidates = candidate_masks(instance.r, instance.target_distance)
    for candidate_mask in _iterate_with_progress(
        candidates,
        f"ranking restart {restart_index}",
        show_progress,
    ):
        score = _safe_priority(
            priority_fn,
            candidate_mask,
            instance.n,
            instance.k,
            instance.target_distance,
        )
        tie_break = _deterministic_tiebreak(candidate_mask, restart_index)
        scored_candidates.append((score, tie_break, candidate_mask))
    scored_candidates.sort(reverse=True)
    ordered_candidates = tuple(candidate_mask for _, _, candidate_mask in scored_candidates)
    ordered_scores = tuple((candidate_mask, score) for score, _, candidate_mask in scored_candidates)
    return ordered_candidates, ordered_scores


def greedy_construct(
    instance: BenchmarkInstance,
    priority_fn: PriorityFn,
    restart_index: int,
    show_progress: bool = False,
) -> SearchAttemptResult:
    """Run one fixed sorted-greedy pass for a single benchmark instance."""
    search_state = IncrementalForbiddenState(instance.r, instance.target_distance)
    ordered_candidates, ordered_scores = ranked_candidates(
        instance,
        priority_fn,
        restart_index,
        show_progress=show_progress,
    )
    blocked_candidate_count = 0
    illegal_weight_histogram: Counter[int] = Counter()

    for candidate_mask in _iterate_with_progress(
        ordered_candidates,
        f"greedy restart {restart_index}",
        show_progress,
    ):
        if len(search_state.selected_free_columns) >= instance.k:
            break
        if search_state.can_add(candidate_mask):
            search_state.add(candidate_mask)
        else:
            blocked_candidate_count += 1
            illegal_weight_histogram[popcount(candidate_mask)] += 1

    selected = tuple(search_state.selected_free_columns)
    success = len(selected) == instance.k and validate_free_columns(
        instance.r,
        selected,
        instance.target_distance,
    )
    return SearchAttemptResult(
        success=success,
        selected_free_columns=selected,
        added_free_columns=len(selected),
        restart_index=restart_index,
        sorted_candidates=ordered_candidates,
        sorted_scores=ordered_scores,
        blocked_candidate_count=blocked_candidate_count,
        illegal_weight_histogram=tuple(sorted(illegal_weight_histogram.items())),
        chosen_weights=tuple(popcount(mask) for mask in selected),
    )


def best_restart_for_instance(
    instance: BenchmarkInstance,
    priority_fn: PriorityFn,
    show_progress: bool = False,
) -> SearchAttemptResult:
    """Evaluate all fixed restarts and keep the best deterministic attempt."""
    attempts = [
        greedy_construct(
            instance,
            priority_fn,
            restart_index,
            show_progress=show_progress,
        )
        for restart_index in range(instance.restarts)
    ]
    return max(
        attempts,
        key=lambda attempt: (
            int(attempt.success),
            attempt.added_free_columns,
            -attempt.restart_index,
        ),
    )


def search_best_distance(
    n: int, k: int, priority_fn: PriorityFn, max_restarts: int = 3
) -> Tuple[int, Tuple[int, ...]]:
    """Phase-2 wrapper retained for future use."""
    r = n - k
    for distance in range(n, 1, -1):
        instance = BenchmarkInstance(
            name=f"search_[{n},{k},{distance}]",
            n=n,
            k=k,
            target_distance=distance,
            restarts=max_restarts,
        )
        attempt = best_restart_for_instance(instance, priority_fn)
        if attempt.success:
            return distance, attempt.selected_free_columns
    return 1, tuple()


def load_priority_function(program_path: str) -> PriorityFn:
    """Load the evolved priority function from a program file."""
    module_name = f"linear_code_priority_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to import program from {program_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "priority"):
        raise AttributeError("Program must define a priority(column_mask, n, k, d) function")
    return module.priority


def evaluate_priority_function(
    priority_fn: PriorityFn, instance: BenchmarkInstance | None = None
) -> EvaluationResult:
    """Run the fixed greedy search on one configurable instance."""
    active_instance = instance or DEFAULT_INSTANCE
    attempt = best_restart_for_instance(active_instance, priority_fn)
    progress = attempt.added_free_columns / active_instance.k
    combined_score = 1.0 if attempt.success else progress

    blocked_counter: Counter[int] = Counter()
    for weight, count in attempt.illegal_weight_histogram:
        blocked_counter[weight] += count
    top_ranked_columns = [
        {
            "column": format_mask(mask, active_instance.r),
            "score": score,
        }
        for mask, score in attempt.sorted_scores[: min(10, len(attempt.sorted_scores))]
    ]

    return EvaluationResult(
        metrics={
            "combined_score": combined_score,
            "success_rate": float(attempt.success),
            "avg_progress": progress,
            "constructed_columns": attempt.added_free_columns,
            "target_columns": active_instance.k,
            "target_distance": active_instance.target_distance,
            "n": active_instance.n,
            "k": active_instance.k,
        },
        artifacts={
            "instance": json.dumps(
                {
                    "name": active_instance.name,
                    "n": active_instance.n,
                    "k": active_instance.k,
                    "d": active_instance.target_distance,
                    "restarts": active_instance.restarts,
                },
                sort_keys=True,
            ),
            "search_result": json.dumps(
                {
                    "success": attempt.success,
                    "restart": attempt.restart_index,
                    "added_free_columns": attempt.added_free_columns,
                    "candidate_count": len(attempt.sorted_candidates),
                    "blocked_candidates": attempt.blocked_candidate_count,
                    "target_free_columns": active_instance.k,
                    "selected_free_columns": [
                        format_mask(mask, active_instance.r)
                        for mask in attempt.selected_free_columns
                    ],
                    "chosen_weights": list(attempt.chosen_weights),
                },
                sort_keys=True,
            ),
            "top_ranked_columns": json.dumps(top_ranked_columns, sort_keys=True),
            "blocked_weight_histogram": json.dumps(dict(sorted(blocked_counter.items()))),
        },
    )


def evaluate_program_path(program_path: str) -> EvaluationResult:
    """Convenience wrapper used by the OpenEvolve evaluator."""
    priority_fn = load_priority_function(program_path)
    return evaluate_priority_function(priority_fn, instance_from_env())
