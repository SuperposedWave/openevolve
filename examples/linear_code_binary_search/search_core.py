"""Fixed search skeleton for binary linear-code feasibility benchmarks."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import uuid
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

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


PriorityFn = Callable[[int, Dict[str, object]], float]


@dataclass(frozen=True)
class BenchmarkInstance:
    """Small exact benchmark with a known optimal target distance."""

    name: str
    n: int
    k: int
    target_distance: int
    optimal_distance: int
    witness_free_columns: Tuple[int, ...]
    restarts: int = 3
    description: str = ""

    @property
    def r(self) -> int:
        return self.n - self.k


@dataclass
class SearchAttemptResult:
    """Result of one deterministic greedy restart."""

    success: bool
    selected_free_columns: Tuple[int, ...]
    added_free_columns: int
    restart_index: int
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
        self.coordinate_usage = [0] * r

    def can_add(self, column_mask: int) -> bool:
        return column_mask not in self.forbidden

    def add(self, column_mask: int) -> None:
        if not self.can_add(column_mask):
            raise ValueError(f"Illegal free column {column_mask}")
        _add_column_to_reachable(self.reachable, column_mask, self.max_subset_size)
        self.forbidden = set().union(*self.reachable)
        self.selected_free_columns.append(column_mask)
        for bit_index in range(self.r):
            if column_mask & (1 << bit_index):
                self.coordinate_usage[bit_index] += 1


BENCHMARKS: Tuple[BenchmarkInstance, ...] = (
    BenchmarkInstance(
        name="short_[5,2,3]",
        n=5,
        k=2,
        target_distance=3,
        optimal_distance=3,
        witness_free_columns=(3, 5),
        description="A short code where distinct weight-2 columns already suffice.",
    ),
    BenchmarkInstance(
        name="parity_[6,3,3]",
        n=6,
        k=3,
        target_distance=3,
        optimal_distance=3,
        witness_free_columns=(3, 5, 6),
        description="The [6,3,3] single-parity style benchmark.",
    ),
    BenchmarkInstance(
        name="simplex_[7,3,4]",
        n=7,
        k=3,
        target_distance=4,
        optimal_distance=4,
        witness_free_columns=(7, 11, 13),
        description="A simplex-style instance with distance 4.",
    ),
    BenchmarkInstance(
        name="hamming_[7,4,3]",
        n=7,
        k=4,
        target_distance=3,
        optimal_distance=3,
        witness_free_columns=(3, 5, 6, 7),
        description="The classic [7,4,3] Hamming-code benchmark.",
    ),
    BenchmarkInstance(
        name="extended_hamming_[8,4,4]",
        n=8,
        k=4,
        target_distance=4,
        optimal_distance=4,
        witness_free_columns=(7, 11, 13, 14),
        description="A compact [8,4,4] benchmark.",
    ),
    BenchmarkInstance(
        name="lifted_[9,4,4]",
        n=9,
        k=4,
        target_distance=4,
        optimal_distance=4,
        witness_free_columns=(7, 11, 13, 14),
        description="A rank-5 variant that still admits distance 4.",
    ),
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
    return basis_columns(r) + tuple(free_columns)


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


def exact_find_feasible_free_columns(
    r: int, k: int, distance: int
) -> Tuple[int, ...] | None:
    """Exact brute-force witness search for small benchmarks."""
    for free_columns in combinations(candidate_masks(r, distance), k):
        if validate_free_columns(r, free_columns, distance):
            return tuple(free_columns)
    return None


def exact_best_distance(n: int, k: int) -> Tuple[int, Tuple[int, ...]]:
    """Exact best-distance search for small binary benchmarks."""
    r = n - k
    for distance in range(n, 1, -1):
        witness = exact_find_feasible_free_columns(r, k, distance)
        if witness is not None:
            return distance, witness
    return 1, tuple()


@lru_cache(maxsize=1)
def validate_benchmark_catalog() -> bool:
    """Fail fast if the fixed benchmark table is inconsistent."""
    for instance in BENCHMARKS:
        if not validate_free_columns(instance.r, instance.witness_free_columns, instance.target_distance):
            raise ValueError(f"Invalid witness for benchmark {instance.name}")
        exact_distance, _ = exact_best_distance(instance.n, instance.k)
        if exact_distance != instance.optimal_distance:
            raise ValueError(
                f"Benchmark {instance.name} expected distance {instance.optimal_distance}, "
                f"but exact search found {exact_distance}"
            )
        if instance.target_distance != instance.optimal_distance:
            raise ValueError(f"Benchmark {instance.name} target distance must be optimal")
    return True


def format_mask(mask: int, r: int) -> str:
    """Binary string representation with fixed width."""
    return format(mask, f"0{r}b")


def _safe_priority(priority_fn: PriorityFn, candidate_mask: int, state: Dict[str, object]) -> float:
    """Protect the fixed skeleton from bad priority implementations."""
    try:
        value = priority_fn(candidate_mask, state)
    except Exception:
        value = popcount(candidate_mask)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = float(popcount(candidate_mask))
    if not math.isfinite(value):
        return float(popcount(candidate_mask))
    return value


def _deterministic_tiebreak(candidate_mask: int, restart_index: int, step_index: int) -> int:
    """Restart-specific tie-break to create deterministic diversity."""
    return (
        candidate_mask * 1103515245
        + restart_index * 2654435761
        + step_index * 97531
        + 12345
    ) & 0xFFFFFFFF


def _build_priority_state(
    instance: BenchmarkInstance,
    search_state: IncrementalForbiddenState,
    restart_index: int,
    remaining_candidates: int,
) -> Dict[str, object]:
    """Stable public state exposed to the evolved priority function."""
    selected_weights = tuple(popcount(mask) for mask in search_state.selected_free_columns)
    weight_histogram = tuple(
        selected_weights.count(weight) for weight in range(instance.r + 1)
    )
    return {
        "n": instance.n,
        "k": instance.k,
        "r": instance.r,
        "D": instance.target_distance,
        "restart_index": restart_index,
        "selected_count": len(search_state.selected_free_columns),
        "remaining_slots": instance.k - len(search_state.selected_free_columns),
        "selected_free_columns": tuple(search_state.selected_free_columns),
        "selected_weights": selected_weights,
        "selected_weight_histogram": weight_histogram,
        "coordinate_usage": tuple(search_state.coordinate_usage),
        "xor_layer_sizes": tuple(len(layer) for layer in search_state.reachable),
        "forbidden_size": len(search_state.forbidden),
        "candidate_pool_remaining": remaining_candidates,
    }


def greedy_construct(
    instance: BenchmarkInstance, priority_fn: PriorityFn, restart_index: int
) -> SearchAttemptResult:
    """Run one fixed greedy pass for a single benchmark instance."""
    search_state = IncrementalForbiddenState(instance.r, instance.target_distance)
    remaining_candidates = list(candidate_masks(instance.r, instance.target_distance))
    blocked_candidate_count = 0
    illegal_weight_histogram: Counter[int] = Counter()

    while (
        len(search_state.selected_free_columns) < instance.k
        and remaining_candidates
    ):
        legal_candidates: List[Tuple[float, int, int]] = []
        next_remaining: List[int] = []
        public_state = _build_priority_state(
            instance,
            search_state,
            restart_index=restart_index,
            remaining_candidates=len(remaining_candidates),
        )

        for candidate_mask in remaining_candidates:
            if search_state.can_add(candidate_mask):
                score = _safe_priority(priority_fn, candidate_mask, public_state)
                tie_break = _deterministic_tiebreak(
                    candidate_mask,
                    restart_index,
                    len(search_state.selected_free_columns),
                )
                legal_candidates.append((score, tie_break, candidate_mask))
                next_remaining.append(candidate_mask)
            else:
                blocked_candidate_count += 1
                illegal_weight_histogram[popcount(candidate_mask)] += 1

        if not legal_candidates:
            break

        legal_candidates.sort(reverse=True)
        best_candidate = legal_candidates[0][2]
        search_state.add(best_candidate)
        remaining_candidates = [
            candidate_mask
            for candidate_mask in next_remaining
            if candidate_mask != best_candidate
        ]

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
        blocked_candidate_count=blocked_candidate_count,
        illegal_weight_histogram=tuple(sorted(illegal_weight_histogram.items())),
        chosen_weights=tuple(popcount(mask) for mask in selected),
    )


def best_restart_for_instance(
    instance: BenchmarkInstance, priority_fn: PriorityFn
) -> SearchAttemptResult:
    """Evaluate all fixed restarts and keep the best deterministic attempt."""
    attempts = [
        greedy_construct(instance, priority_fn, restart_index)
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
            optimal_distance=distance,
            witness_free_columns=tuple(),
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
        raise AttributeError("Program must define a priority(candidate_mask, state) function")
    return module.priority


def evaluate_priority_function(
    priority_fn: PriorityFn, benchmarks: Sequence[BenchmarkInstance] | None = None
) -> EvaluationResult:
    """Run the fixed benchmark suite against an evolved priority function."""
    validate_benchmark_catalog()
    benchmark_set = tuple(benchmarks or BENCHMARKS)
    best_attempts = [best_restart_for_instance(instance, priority_fn) for instance in benchmark_set]

    successes = 0
    total_progress = 0.0
    total_columns = 0
    blocked_counter: Counter[int] = Counter()
    benchmark_summaries = []

    for instance, attempt in zip(benchmark_set, best_attempts):
        progress = attempt.added_free_columns / instance.k
        total_progress += progress
        total_columns += attempt.added_free_columns
        if attempt.success:
            successes += 1
        for weight, count in attempt.illegal_weight_histogram:
            blocked_counter[weight] += count
        benchmark_summaries.append(
            {
                "name": instance.name,
                "success": attempt.success,
                "restart": attempt.restart_index,
                "added_free_columns": attempt.added_free_columns,
                "target_free_columns": instance.k,
                "selected_free_columns": [
                    format_mask(mask, instance.r) for mask in attempt.selected_free_columns
                ],
                "chosen_weights": list(attempt.chosen_weights),
            }
        )

    success_rate = successes / len(benchmark_set)
    avg_progress = total_progress / len(benchmark_set)
    combined_score = 0.75 * success_rate + 0.25 * avg_progress

    return EvaluationResult(
        metrics={
            "combined_score": combined_score,
            "success_rate": success_rate,
            "avg_progress": avg_progress,
            "solved_instances": successes,
            "total_instances": len(benchmark_set),
            "constructed_columns": total_columns,
        },
        artifacts={
            "benchmark_summaries": json.dumps(benchmark_summaries, sort_keys=True),
            "successful_instances": ", ".join(
                summary["name"] for summary in benchmark_summaries if summary["success"]
            )
            or "none",
            "failed_instances": ", ".join(
                summary["name"] for summary in benchmark_summaries if not summary["success"]
            )
            or "none",
            "blocked_weight_histogram": json.dumps(dict(sorted(blocked_counter.items()))),
            "benchmark_catalog": json.dumps(
                [
                    {
                        "name": instance.name,
                        "n": instance.n,
                        "k": instance.k,
                        "target_distance": instance.target_distance,
                        "optimal_distance": instance.optimal_distance,
                    }
                    for instance in benchmark_set
                ],
                sort_keys=True,
            ),
        },
    )


def evaluate_program_path(program_path: str) -> EvaluationResult:
    """Convenience wrapper used by the OpenEvolve evaluator."""
    priority_fn = load_priority_function(program_path)
    return evaluate_priority_function(priority_fn)


validate_benchmark_catalog()
