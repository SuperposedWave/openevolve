"""FunSearch-style static-priority search for ternary matrix construction."""

from __future__ import annotations

from contextlib import ExitStack
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import heapq
import importlib.util
import json
import logging
import math
import multiprocessing
import os
import struct
import sys
import tempfile
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import BinaryIO, Callable, Iterable, Iterator, List, Sequence, Tuple

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


logger = logging.getLogger(__name__)
PriorityFn = Callable[[int, int, int, int], float]
FIELD_ORDER = 3
NONZERO_FIELD_ELEMENTS = tuple(range(1, FIELD_ORDER))
_SCORED_RECORD_STRUCT = struct.Struct(">dQQ")
_PROCESS_PRIORITY_FN: PriorityFn | None = None


@dataclass(frozen=True)
class ParallelismPlan:
    """Resolved worker budget for restart and candidate-level parallelism."""

    restart_workers: int
    candidate_workers: int
    chunk_prefetch_depth: int


@dataclass(frozen=True)
class BenchmarkInstance:
    """Single GF(3) feasibility instance for a systematic parity-check search."""

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
    candidate_count: int
    restart_index: int
    sorted_candidates: Tuple[int, ...]
    sorted_scores: Tuple[Tuple[int, float], ...]
    blocked_candidate_count: int
    illegal_weight_histogram: Tuple[Tuple[int, int], ...]
    chosen_weights: Tuple[int, ...]


@dataclass(frozen=True)
class ScoredChunkRun:
    """One sorted on-disk run used by the streaming candidate path."""

    path: str
    record_count: int


class IncrementalForbiddenState:
    """Maintains exact low-order GF(3) combination layers for legality checks."""

    def __init__(self, r: int, distance: int):
        self.r = r
        self.distance = distance
        self.max_subset_size = max(distance - 2, 0)
        self.reachable = _initialize_reachable_layers(r, distance)
        self.forbidden = forbidden_masks_from_layers(self.reachable)
        self.selected_free_columns: List[int] = []

    def can_add(self, column_code: int) -> bool:
        return normalize_column_code(column_code, self.r) not in self.forbidden

    def add(self, column_code: int) -> None:
        normalized_code = normalize_column_code(column_code, self.r)
        if not self.can_add(normalized_code):
            raise ValueError(f"Illegal free column {column_code}")
        _add_column_to_reachable(
            self.reachable,
            normalized_code,
            self.r,
            self.max_subset_size,
        )
        self.forbidden = forbidden_masks_from_layers(self.reachable)
        self.selected_free_columns.append(normalized_code)


DEFAULT_INSTANCE = BenchmarkInstance(
    name="default_[7,3,4]_3",
    n=7,
    k=3,
    target_distance=4,
    restarts=3,
    description="Default single-instance target used by the example.",
)


def encode_column(digits: Sequence[int]) -> int:
    """Encode a GF(3) column vector as a little-endian base-3 integer."""
    code = 0
    factor = 1
    for digit in digits:
        value = int(digit)
        if value < 0 or value >= FIELD_ORDER:
            raise ValueError(f"GF(3) digit out of range: {digit}")
        code += value * factor
        factor *= FIELD_ORDER
    return code


def decode_column(column_code: int, r: int) -> Tuple[int, ...]:
    """Decode a base-3 integer into an r-coordinate GF(3) column vector."""
    if column_code < 0:
        raise ValueError("column_code must be non-negative")
    digits = []
    remaining = column_code
    for _ in range(r):
        digits.append(remaining % FIELD_ORDER)
        remaining //= FIELD_ORDER
    if remaining:
        raise ValueError(f"column_code {column_code} does not fit in {r} GF(3) coordinates")
    return tuple(digits)


def _scale_digits(digits: Sequence[int], scalar: int) -> Tuple[int, ...]:
    """Scale one GF(3) vector by a field scalar."""
    return tuple((scalar * digit) % FIELD_ORDER for digit in digits)


def _add_digits(left: Sequence[int], right: Sequence[int]) -> Tuple[int, ...]:
    """Add two GF(3) vectors coordinate-wise."""
    return tuple((a + b) % FIELD_ORDER for a, b in zip(left, right))


def normalize_digits(digits: Sequence[int]) -> Tuple[int, ...]:
    """Return the projective representative whose first non-zero coordinate is 1."""
    normalized = tuple(digit % FIELD_ORDER for digit in digits)
    for digit in normalized:
        if digit:
            if digit == 1:
                return normalized
            return _scale_digits(normalized, digit)
    return normalized


def normalize_column_code(column_code: int, r: int) -> int:
    """Normalize a GF(3) column code up to non-zero scalar multiplication."""
    return encode_column(normalize_digits(decode_column(column_code, r)))


def support_weight(column_code: int, r: int) -> int:
    """Return the Hamming support weight of a GF(3) column."""
    return sum(1 for digit in decode_column(column_code, r) if digit)


def popcount(column_code: int) -> int:
    """Return support weight for a base-3 column without requiring trailing-zero width."""
    count = 0
    remaining = column_code
    while remaining:
        if remaining % FIELD_ORDER:
            count += 1
        remaining //= FIELD_ORDER
    return count


def format_column(column_code: int, r: int) -> str:
    """GF(3) string representation in row order."""
    return "".join(str(digit) for digit in decode_column(column_code, r))


def parse_column(column_text: str) -> int:
    """Parse a row-order GF(3) column string."""
    return encode_column(tuple(int(digit) for digit in column_text.strip()))


def _env_worker_override(name: str) -> int | None:
    """Parse an optional worker-count override from the environment."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return None
    try:
        return max(int(raw_value), 1)
    except ValueError:
        return None


def _resolve_worker_count(task_count: int, env_name: str, minimum_parallel_tasks: int) -> int:
    """Choose a bounded thread count for a pool-backed parallel section."""
    override = _env_worker_override(env_name)
    if override is not None:
        return min(override, task_count)
    if task_count < max(minimum_parallel_tasks, 2):
        return 1
    cpu_count = os.cpu_count() or 1
    return min(cpu_count, task_count)


def _env_positive_int(name: str) -> int | None:
    """Parse an optional strictly positive integer environment variable."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return None
    try:
        parsed = int(raw_value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _env_flag_enabled(name: str) -> bool:
    """Parse a conventional boolean environment flag."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return False
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _cpu_budget() -> int:
    """Return the total CPU budget available to automatic worker planning."""
    return max(os.cpu_count() or 1, 1)


def _profiling_enabled() -> bool:
    """Return whether opt-in stage profiling logs should be emitted."""
    return _env_flag_enabled("LINEAR_CODE_PROFILE")


def _log_profile(stage: str, **fields: object) -> None:
    """Emit one structured stage log line when profiling is enabled."""
    if not _profiling_enabled():
        return
    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info(f"linear_code_profile stage={stage} {payload}".strip())


def _format_eta(total_seconds: float) -> str:
    """Render a compact ETA string for progress logs."""
    rounded_seconds = max(int(round(total_seconds)), 0)
    minutes, seconds = divmod(rounded_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def _candidate_chunk_size() -> int:
    """Return the numeric code-range chunk size for streaming candidate generation."""
    return _env_positive_int("LINEAR_CODE_CANDIDATE_CHUNK_SIZE") or (1 << 20)


def _candidate_map_chunksize(task_count: int, worker_count: int) -> int:
    """Choose a coarse process-pool map batch size for large candidate chunks."""
    override = _env_positive_int("LINEAR_CODE_CANDIDATE_MAP_CHUNKSIZE")
    if override is not None:
        return min(max(override, 1), max(task_count, 1))
    if task_count <= 1 or worker_count <= 1:
        return 1
    target_batches = max(worker_count * 16, 1)
    return max(1, math.ceil(task_count / target_batches))


def _streaming_mask_threshold() -> int:
    """Return the code-space threshold above which candidate streaming is enabled by default."""
    return _env_positive_int("LINEAR_CODE_STREAMING_THRESHOLD_MASKS") or (1 << 22)


def _should_stream_candidates(r: int) -> bool:
    """Choose the candidate-processing strategy for the current redundancy."""
    if _env_flag_enabled("LINEAR_CODE_FORCE_STREAMING"):
        return True
    return (FIELD_ORDER**r) > _streaming_mask_threshold()


def _candidate_executor_mode() -> str:
    """Return the requested candidate-scoring executor mode."""
    raw_value = os.environ.get("LINEAR_CODE_CANDIDATE_EXECUTOR", "thread")
    normalized = raw_value.strip().lower()
    if normalized in {"process", "thread"}:
        return normalized
    return "thread"


def _candidate_chunk_prefetch_depth(candidate_workers: int | None) -> int:
    """Keep one extra chunk prefetched when candidate scoring is parallelized."""
    return 2 if (candidate_workers or 1) > 1 else 1


def _resolve_parallelism_plan(
    instance: BenchmarkInstance,
    show_progress: bool = False,
) -> ParallelismPlan:
    """Auto-balance restart and candidate-level worker budgets."""
    cpu_budget = _cpu_budget()
    restart_override = _env_worker_override("LINEAR_CODE_RESTART_WORKERS")
    candidate_override = _env_worker_override("LINEAR_CODE_CANDIDATE_WORKERS")
    streaming = _should_stream_candidates(instance.r)
    process_candidate_executor = _candidate_executor_mode() == "process"

    if show_progress or instance.restarts <= 1:
        restart_workers = 1
    elif restart_override is not None:
        restart_workers = min(restart_override, instance.restarts)
    elif process_candidate_executor:
        # Avoid nested fork-backed process pools from multiple restart threads.
        restart_workers = 1
    elif candidate_override is not None:
        restart_workers = min(
            instance.restarts,
            max(cpu_budget // max(candidate_override, 1), 1),
        )
    elif streaming and cpu_budget < instance.restarts * 8:
        restart_workers = 1
    else:
        minimum_candidate_workers = 8 if streaming else 4
        restart_workers = min(
            instance.restarts,
            max(cpu_budget // minimum_candidate_workers, 1),
        )

    if candidate_override is not None:
        candidate_workers = candidate_override
    elif restart_override is not None:
        candidate_workers = max(cpu_budget // max(restart_workers, 1), 1)
    else:
        candidate_workers = max(cpu_budget // max(restart_workers, 1), 1)

    if show_progress:
        restart_workers = 1
    restart_workers = min(max(restart_workers, 1), instance.restarts)
    candidate_workers = max(candidate_workers, 1)
    chunk_prefetch_depth = 2 if streaming and candidate_workers > 1 else 1
    return ParallelismPlan(
        restart_workers=restart_workers,
        candidate_workers=candidate_workers,
        chunk_prefetch_depth=chunk_prefetch_depth,
    )


@lru_cache(maxsize=None)
def basis_columns(r: int) -> Tuple[int, ...]:
    """Systematic identity columns."""
    return tuple(
        encode_column(1 if coordinate == bit_index else 0 for coordinate in range(r))
        for bit_index in range(r)
    )


@lru_cache(maxsize=None)
def candidate_masks(r: int, distance: int) -> Tuple[int, ...]:
    """All projective GF(3) free columns that pass the initial weight filter."""
    min_weight = max(distance - 1, 1)
    return tuple(
        column_code
        for column_code in range(1, FIELD_ORDER**r)
        if column_code == normalize_column_code(column_code, r)
        and support_weight(column_code, r) >= min_weight
    )


def candidate_mask_chunks(
    r: int,
    distance: int,
    chunk_size: int | None = None,
) -> Iterator[Tuple[int, ...]]:
    """Yield candidate codes in numeric-range chunks instead of one giant tuple."""
    min_weight = max(distance - 1, 1)
    resolved_chunk_size = chunk_size or _candidate_chunk_size()
    upper_bound = FIELD_ORDER**r
    for chunk_start in range(1, upper_bound, resolved_chunk_size):
        chunk_end = min(chunk_start + resolved_chunk_size, upper_bound)
        chunk = tuple(
            column_code
            for column_code in range(chunk_start, chunk_end)
            if column_code == normalize_column_code(column_code, r)
            and support_weight(column_code, r) >= min_weight
        )
        if chunk:
            yield chunk


def _next_candidate_chunk(
    iterator: Iterator[Tuple[int, ...]],
) -> Tuple[Tuple[int, ...], float] | None:
    """Fetch the next candidate chunk and measure generation time."""
    started_at = time.perf_counter()
    try:
        chunk = next(iterator)
    except StopIteration:
        return None
    return chunk, time.perf_counter() - started_at


def _iter_candidate_mask_chunks(
    r: int,
    distance: int,
    chunk_size: int,
    prefetch_depth: int = 1,
) -> Iterator[Tuple[int, Tuple[int, ...], float]]:
    """Yield timed candidate chunks, optionally prefetching one chunk ahead."""
    iterator = iter(candidate_mask_chunks(r, distance, chunk_size))
    if prefetch_depth <= 1:
        chunk_index = 0
        while True:
            next_chunk = _next_candidate_chunk(iterator)
            if next_chunk is None:
                break
            chunk_candidates, generation_elapsed_seconds = next_chunk
            yield chunk_index, chunk_candidates, generation_elapsed_seconds
            chunk_index += 1
        return

    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="linear-code-chunk-prefetch",
    ) as executor:
        chunk_index = 0
        next_chunk_future = executor.submit(_next_candidate_chunk, iterator)
        while True:
            next_chunk = next_chunk_future.result()
            if next_chunk is None:
                break
            next_chunk_future = executor.submit(_next_candidate_chunk, iterator)
            chunk_candidates, generation_elapsed_seconds = next_chunk
            yield chunk_index, chunk_candidates, generation_elapsed_seconds
            chunk_index += 1


def _initialize_reachable_layers(r: int, distance: int) -> List[set[int]]:
    """Build exact GF(3) projective span layers for the initial systematic columns."""
    max_subset_size = max(distance - 2, 0)
    reachable = [set() for _ in range(max_subset_size + 1)]
    reachable[0].add(0)
    for column_code in basis_columns(r):
        _add_column_to_reachable(reachable, column_code, r, max_subset_size)
    return reachable


def _add_column_to_reachable(
    reachable: List[set[int]], column_code: int, r: int, max_subset_size: int
) -> None:
    """Incrementally update GF(3) combination layers after accepting a new column."""
    column_digits = decode_column(normalize_column_code(column_code, r), r)
    for subset_size in range(max_subset_size, 0, -1):
        previous_layer = reachable[subset_size - 1]
        if previous_layer:
            additions = set()
            for prior_code in previous_layer:
                prior_digits = decode_column(prior_code, r)
                prior_scalars = (1,) if prior_code == 0 else NONZERO_FIELD_ELEMENTS
                for prior_scalar in prior_scalars:
                    scaled_prior = _scale_digits(prior_digits, prior_scalar)
                    for column_scalar in NONZERO_FIELD_ELEMENTS:
                        combined = _add_digits(
                            scaled_prior,
                            _scale_digits(column_digits, column_scalar),
                        )
                        combined_code = encode_column(combined)
                        if combined_code:
                            additions.add(normalize_column_code(combined_code, r))
            reachable[subset_size].update(additions)


def rebuild_reachable_layers(r: int, distance: int, free_columns: Sequence[int]) -> List[set[int]]:
    """Recompute GF(3) combination layers from scratch for validation and tests."""
    reachable = _initialize_reachable_layers(r, distance)
    max_subset_size = max(distance - 2, 0)
    for column_code in free_columns:
        _add_column_to_reachable(
            reachable,
            normalize_column_code(column_code, r),
            r,
            max_subset_size,
        )
    return reachable


def forbidden_masks_from_layers(reachable: Sequence[set[int]]) -> set[int]:
    """Flatten GF(3) combination layers into the exact forbidden set."""
    return set().union(*reachable).difference({0})


def initial_forbidden_masks(r: int, distance: int) -> set[int]:
    """Initial forbidden set induced by the identity columns."""
    return forbidden_masks_from_layers(_initialize_reachable_layers(r, distance))


def all_columns(r: int, free_columns: Sequence[int]) -> Tuple[int, ...]:
    """All columns in the systematic parity-check matrix."""
    return tuple(normalize_column_code(column_code, r) for column_code in free_columns) + basis_columns(r)


def gf3_rank(columns: Sequence[int], r: int) -> int:
    """Return the rank of GF(3) column vectors."""
    rows = [list(decode_column(column_code, r)) for column_code in columns if column_code]
    rank = 0
    for coordinate in range(r):
        pivot_index = None
        for row_index in range(rank, len(rows)):
            if rows[row_index][coordinate] % FIELD_ORDER:
                pivot_index = row_index
                break
        if pivot_index is None:
            continue
        rows[rank], rows[pivot_index] = rows[pivot_index], rows[rank]
        inverse = 1 if rows[rank][coordinate] == 1 else 2
        rows[rank] = [(value * inverse) % FIELD_ORDER for value in rows[rank]]
        for row_index in range(len(rows)):
            if row_index == rank:
                continue
            factor = rows[row_index][coordinate] % FIELD_ORDER
            if factor:
                rows[row_index] = [
                    (value - factor * pivot_value) % FIELD_ORDER
                    for value, pivot_value in zip(rows[row_index], rows[rank])
                ]
        rank += 1
        if rank == len(rows):
            break
    return rank


def columns_meet_distance_requirement(columns: Sequence[int], distance: int, r: int) -> bool:
    """Exact distance check via exhaustive GF(3) rank tests up to distance - 1."""
    for subset_size in range(1, distance):
        for subset in combinations(columns, subset_size):
            if gf3_rank(subset, r) < subset_size:
                return False
    return True


def validate_free_columns(r: int, free_columns: Sequence[int], distance: int) -> bool:
    """Validate a systematic construction independently of the greedy state."""
    return columns_meet_distance_requirement(all_columns(r, free_columns), distance, r)


def actual_minimum_distance_from_columns(columns: Sequence[int], r: int) -> int:
    """Return the exact minimum distance induced by a parity-check matrix column set."""
    total_columns = len(columns)
    for subset_size in range(1, total_columns + 1):
        for subset in combinations(columns, subset_size):
            if gf3_rank(subset, r) < subset_size:
                return subset_size
    return total_columns + 1


def actual_minimum_distance(r: int, free_columns: Sequence[int]) -> int:
    """Return the exact minimum distance for H = [P^T | I_r]."""
    return actual_minimum_distance_from_columns(all_columns(r, free_columns), r)


def exact_find_feasible_free_columns(r: int, k: int, distance: int) -> Tuple[int, ...] | None:
    """Exact brute-force witness search for small GF(3) instances."""
    for free_columns in combinations(candidate_masks(r, distance), k):
        if validate_free_columns(r, free_columns, distance):
            return tuple(free_columns)
    return None


def exact_best_distance(n: int, k: int) -> Tuple[int, Tuple[int, ...]]:
    """Exact best-distance search for small GF(3) instances."""
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
        # This is not mathematically impossible in every case, but for this ternary
        # systematic free-column search it leaves no eligible candidates beyond trivial cases.
        raise ValueError("Requested d is too large for this ternary free-column search regime")
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
    """GF(3) column string representation with fixed width."""
    return format_column(mask, r)


def parity_check_matrix_rows(r: int, free_columns: Sequence[int]) -> Tuple[str, ...]:
    """Return the parity-check matrix as row strings over GF(3)."""
    columns = all_columns(r, free_columns)
    rows = []
    for row_index in range(r):
        row_digits = []
        for column_code in columns:
            row_digits.append(str(decode_column(column_code, r)[row_index]))
        rows.append("".join(row_digits))
    return tuple(rows)


def generator_matrix_rows(r: int, free_columns: Sequence[int]) -> Tuple[str, ...]:
    """Return the systematic generator matrix G = [I_k | -P] as row strings over GF(3)."""
    k = len(free_columns)
    rows = []
    for row_index, column_code in enumerate(free_columns):
        identity_digits = ["1" if i == row_index else "0" for i in range(k)]
        parity_digits = [
            str((-digit) % FIELD_ORDER)
            for digit in decode_column(normalize_column_code(column_code, r), r)
        ]
        rows.append("".join(identity_digits + parity_digits))
    return tuple(rows)


def _safe_priority(priority_fn: PriorityFn, candidate_mask: int, n: int, k: int, d: int) -> float:
    """Protect the fixed skeleton from bad static priority implementations."""
    try:
        value = priority_fn(candidate_mask, n, k, d)
    except Exception:
        value = support_weight(candidate_mask, n - k)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = float(support_weight(candidate_mask, n - k))
    if not math.isfinite(value):
        return float(support_weight(candidate_mask, n - k))
    return value


def _score_candidate_job(
    job: Tuple[PriorityFn, int, int, int, int, int]
) -> Tuple[float, int, int]:
    """Score one candidate column and attach the restart-specific tie-break."""
    priority_fn, candidate_mask, n, k, d, restart_index = job
    score = _safe_priority(priority_fn, candidate_mask, n, k, d)
    tie_break = _deterministic_tiebreak(candidate_mask, restart_index)
    return score, tie_break, candidate_mask


def _initialize_process_priority_fn(program_path: str) -> None:
    """Load the priority function inside a process worker."""
    global _PROCESS_PRIORITY_FN
    _PROCESS_PRIORITY_FN = load_priority_function(program_path)


def _score_candidate_job_in_process(
    job: Tuple[int, int, int, int, int]
) -> Tuple[float, int, int]:
    """Score one candidate column using the process-local priority function."""
    candidate_mask, n, k, d, restart_index = job
    if _PROCESS_PRIORITY_FN is None:
        raise RuntimeError("Process scoring requested before initializing priority function")
    score = _safe_priority(_PROCESS_PRIORITY_FN, candidate_mask, n, k, d)
    tie_break = _deterministic_tiebreak(candidate_mask, restart_index)
    return score, tie_break, candidate_mask


def _priority_program_path(priority_fn: PriorityFn) -> str | None:
    """Return the source path backing a priority function when available."""
    code_object = getattr(priority_fn, "__code__", None)
    program_path = getattr(code_object, "co_filename", None)
    if isinstance(program_path, str) and os.path.exists(program_path):
        return program_path
    return None


def _create_process_candidate_pool(
    priority_fn: PriorityFn,
    worker_count: int,
) -> ProcessPoolExecutor | None:
    """Create a process pool for chunk scoring when the priority source path is available."""
    if worker_count <= 1:
        return None
    program_path = _priority_program_path(priority_fn)
    if program_path is None:
        logger.warning(
            "LINEAR_CODE_CANDIDATE_EXECUTOR=process requested, but priority source path is unavailable; "
            "falling back to thread scoring"
        )
        return None
    try:
        mp_context = multiprocessing.get_context("fork")
    except ValueError:
        logger.warning(
            "LINEAR_CODE_CANDIDATE_EXECUTOR=process requested, but this platform does not support "
            "'fork' multiprocessing; falling back to thread scoring"
        )
        return None
    return ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=mp_context,
        initializer=_initialize_process_priority_fn,
        initargs=(program_path,),
    )


def _score_candidate_chunk(
    chunk_candidates: Sequence[int],
    instance: BenchmarkInstance,
    priority_fn: PriorityFn,
    restart_index: int,
    worker_count: int,
    executor_mode: str = "thread",
    process_pool: ProcessPoolExecutor | None = None,
    thread_pool: ThreadPoolExecutor | None = None,
    map_chunksize: int = 1,
) -> List[Tuple[float, int, int]]:
    """Score one candidate chunk, optionally using a worker pool."""
    if worker_count <= 1:
        return [
            _score_candidate_job(
                (
                    priority_fn,
                    candidate_mask,
                    instance.n,
                    instance.k,
                    instance.target_distance,
                    restart_index,
                )
            )
            for candidate_mask in chunk_candidates
        ]

    if executor_mode == "process" and process_pool is not None:
        job_iter = (
            (
                candidate_mask,
                instance.n,
                instance.k,
                instance.target_distance,
                restart_index,
            )
            for candidate_mask in chunk_candidates
        )
        return list(
            process_pool.map(
                _score_candidate_job_in_process,
                job_iter,
                chunksize=map_chunksize,
            )
        )

    job_iter = (
        (
            priority_fn,
            candidate_mask,
            instance.n,
            instance.k,
            instance.target_distance,
            restart_index,
        )
        for candidate_mask in chunk_candidates
    )
    if thread_pool is not None:
        return list(thread_pool.map(_score_candidate_job, job_iter))
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="linear-code-score",
    ) as executor:
        return list(executor.map(_score_candidate_job, job_iter))


def _write_scored_chunk_run(records: Sequence[Tuple[float, int, int]]) -> ScoredChunkRun:
    """Persist one scored candidate chunk as a sorted binary file run on disk."""
    fd, path = tempfile.mkstemp(prefix="linear-code-run-", suffix=".bin")
    record_count = 0
    try:
        with os.fdopen(fd, "wb") as handle:
            for score, tie_break, candidate_mask in records:
                handle.write(_SCORED_RECORD_STRUCT.pack(score, tie_break, candidate_mask))
                record_count += 1
    except Exception:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        raise
    return ScoredChunkRun(path=path, record_count=record_count)


def _read_scored_record(handle: BinaryIO) -> Tuple[float, int, int] | None:
    """Read one scored candidate triple from a chunk-run file."""
    raw_record = handle.read(_SCORED_RECORD_STRUCT.size)
    if not raw_record:
        return None
    score, tie_break, candidate_mask = _SCORED_RECORD_STRUCT.unpack(raw_record)
    return score, int(tie_break), int(candidate_mask)


def _merge_scored_chunk_runs(
    runs: Sequence[ScoredChunkRun],
) -> Iterator[Tuple[float, int, int]]:
    """Merge sorted chunk runs into one exact global descending order."""
    with ExitStack() as stack:
        handles: List[BinaryIO] = []
        heap: List[Tuple[float, int, int, int, float, int, int]] = []
        for run_index, run in enumerate(runs):
            handle = stack.enter_context(open(run.path, "rb"))
            handles.append(handle)
            first_record = _read_scored_record(handle)
            if first_record is None:
                continue
            score, tie_break, candidate_mask = first_record
            heapq.heappush(
                heap,
                (-score, -tie_break, -candidate_mask, run_index, score, tie_break, candidate_mask),
            )

        while heap:
            _, _, _, run_index, score, tie_break, candidate_mask = heapq.heappop(heap)
            yield score, tie_break, candidate_mask
            next_record = _read_scored_record(handles[run_index])
            if next_record is None:
                continue
            next_score, next_tie_break, next_candidate_mask = next_record
            heapq.heappush(
                heap,
                (
                    -next_score,
                    -next_tie_break,
                    -next_candidate_mask,
                    run_index,
                    next_score,
                    next_tie_break,
                    next_candidate_mask,
                ),
            )


def _cleanup_scored_chunk_runs(runs: Sequence[ScoredChunkRun]) -> None:
    """Delete temporary chunk-run files after a streaming search completes."""
    for run in runs:
        try:
            os.unlink(run.path)
        except FileNotFoundError:
            continue


def _run_restart_job(
    job: Tuple[BenchmarkInstance, PriorityFn, int, int]
) -> SearchAttemptResult:
    """Run one restart in isolation using the resolved candidate-worker budget."""
    instance, priority_fn, restart_index, candidate_workers = job
    return greedy_construct(
        instance,
        priority_fn,
        restart_index,
        show_progress=False,
        candidate_workers=candidate_workers,
    )


def _deterministic_tiebreak(candidate_mask: int, restart_index: int) -> int:
    """Restart-specific tie-break used only to perturb equal-score columns."""
    return (
        candidate_mask * 1103515245
        + restart_index * 2654435761
        + 12345
    ) & 0xFFFFFFFF


def _iterate_with_progress(
    items: Iterable,
    description: str,
    show_progress: bool,
    total: int | None = None,
):
    """Yield items, optionally wrapped in a tqdm progress bar."""
    if not show_progress:
        return items

    try:
        from tqdm import tqdm
    except Exception:
        return items

    return tqdm(items, desc=description, leave=False, total=total)


def ranked_candidates(
    instance: BenchmarkInstance,
    priority_fn: PriorityFn,
    restart_index: int,
    show_progress: bool = False,
    candidate_workers: int | None = None,
) -> Tuple[Tuple[int, ...], Tuple[Tuple[int, float], ...]]:
    """Compute a single static ordering of all candidate columns."""
    scored_candidates = []
    started_at = time.perf_counter()
    candidates = candidate_masks(instance.r, instance.target_distance)
    _log_profile(
        "candidate_generation",
        n=instance.n,
        k=instance.k,
        d=instance.target_distance,
        r=instance.r,
        restart=restart_index,
        candidate_count=len(candidates),
        elapsed_seconds=f"{time.perf_counter() - started_at:.6f}",
    )
    worker_count = candidate_workers
    if worker_count is None:
        worker_count = _resolve_worker_count(
            len(candidates),
            env_name="LINEAR_CODE_CANDIDATE_WORKERS",
            minimum_parallel_tasks=64,
        )
    worker_count = min(max(worker_count, 1), max(len(candidates), 1))

    scoring_started_at = time.perf_counter()
    if worker_count <= 1:
        for candidate_mask in _iterate_with_progress(
            candidates,
            f"ranking restart {restart_index}",
            show_progress,
            total=len(candidates),
        ):
            scored_candidates.append(
                _score_candidate_job(
                    (
                        priority_fn,
                        candidate_mask,
                        instance.n,
                        instance.k,
                        instance.target_distance,
                        restart_index,
                    )
                )
            )
    else:
        jobs = [
            (
                priority_fn,
                candidate_mask,
                instance.n,
                instance.k,
                instance.target_distance,
                restart_index,
            )
            for candidate_mask in candidates
        ]
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="linear-code-score",
        ) as executor:
            scored_iterator = executor.map(_score_candidate_job, jobs)
            for scored_candidate in _iterate_with_progress(
                scored_iterator,
                f"ranking restart {restart_index}",
                show_progress,
                total=len(candidates),
            ):
                scored_candidates.append(scored_candidate)
    _log_profile(
        "candidate_scoring",
        n=instance.n,
        k=instance.k,
        d=instance.target_distance,
        r=instance.r,
        restart=restart_index,
        candidate_count=len(candidates),
        worker_count=worker_count,
        elapsed_seconds=f"{time.perf_counter() - scoring_started_at:.6f}",
    )
    sort_started_at = time.perf_counter()
    scored_candidates.sort(reverse=True)
    _log_profile(
        "candidate_sort",
        n=instance.n,
        k=instance.k,
        d=instance.target_distance,
        r=instance.r,
        restart=restart_index,
        candidate_count=len(scored_candidates),
        elapsed_seconds=f"{time.perf_counter() - sort_started_at:.6f}",
    )
    ordered_candidates = tuple(candidate_mask for _, _, candidate_mask in scored_candidates)
    ordered_scores = tuple((candidate_mask, score) for score, _, candidate_mask in scored_candidates)
    return ordered_candidates, ordered_scores


def _ranked_candidate_runs(
    instance: BenchmarkInstance,
    priority_fn: PriorityFn,
    restart_index: int,
    candidate_workers: int | None = None,
    chunk_prefetch_depth: int = 1,
) -> Tuple[List[ScoredChunkRun], int]:
    """Build sorted chunk runs for the exact streaming candidate path."""
    worker_count = candidate_workers
    runs: List[ScoredChunkRun] = []
    candidate_count = 0
    processed_chunks = 0
    chunk_size = _candidate_chunk_size()
    total_chunks = max(math.ceil(((FIELD_ORDER**instance.r) - 1) / chunk_size), 1)
    streaming_started_at = time.perf_counter()
    if worker_count is None:
        worker_count = _resolve_worker_count(
            chunk_size,
            env_name="LINEAR_CODE_CANDIDATE_WORKERS",
            minimum_parallel_tasks=64,
        )
    worker_count = min(max(worker_count, 1), max(chunk_size, 1))

    executor_mode = "thread"
    process_pool = None
    thread_pool = None
    if _candidate_executor_mode() == "process":
        process_pool = _create_process_candidate_pool(priority_fn, worker_count)
        if process_pool is not None:
            executor_mode = "process"
    if executor_mode == "thread" and worker_count > 1:
        thread_pool = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="linear-code-score",
        )

    try:
        for chunk_index, chunk_candidates, generation_elapsed_seconds in _iter_candidate_mask_chunks(
            instance.r,
            instance.target_distance,
            chunk_size,
            prefetch_depth=chunk_prefetch_depth,
        ):
            _log_profile(
                "candidate_generation",
                n=instance.n,
                k=instance.k,
                d=instance.target_distance,
                r=instance.r,
                restart=restart_index,
                chunk_index=chunk_index,
                chunk_candidate_count=len(chunk_candidates),
                elapsed_seconds=f"{generation_elapsed_seconds:.6f}",
            )
            candidate_count += len(chunk_candidates)

            scoring_started_at = time.perf_counter()
            map_chunksize = _candidate_map_chunksize(len(chunk_candidates), worker_count)
            scored_chunk = _score_candidate_chunk(
                chunk_candidates,
                instance,
                priority_fn,
                restart_index,
                worker_count,
                executor_mode=executor_mode,
                process_pool=process_pool,
                thread_pool=thread_pool,
                map_chunksize=map_chunksize,
            )
            scoring_elapsed_seconds = time.perf_counter() - scoring_started_at
            _log_profile(
                "candidate_scoring",
                n=instance.n,
                k=instance.k,
                d=instance.target_distance,
                r=instance.r,
                restart=restart_index,
                chunk_index=chunk_index,
                chunk_candidate_count=len(chunk_candidates),
                worker_count=worker_count,
                executor_mode=executor_mode,
                map_chunksize=map_chunksize,
                elapsed_seconds=f"{scoring_elapsed_seconds:.6f}",
            )

            sort_started_at = time.perf_counter()
            scored_chunk.sort(reverse=True)
            sort_elapsed_seconds = time.perf_counter() - sort_started_at
            _log_profile(
                "candidate_sort",
                n=instance.n,
                k=instance.k,
                d=instance.target_distance,
                r=instance.r,
                restart=restart_index,
                chunk_index=chunk_index,
                chunk_candidate_count=len(scored_chunk),
                elapsed_seconds=f"{sort_elapsed_seconds:.6f}",
            )
            write_started_at = time.perf_counter()
            runs.append(_write_scored_chunk_run(scored_chunk))
            write_elapsed_seconds = time.perf_counter() - write_started_at
            _log_profile(
                "candidate_write",
                n=instance.n,
                k=instance.k,
                d=instance.target_distance,
                r=instance.r,
                restart=restart_index,
                chunk_index=chunk_index,
                chunk_candidate_count=len(scored_chunk),
                elapsed_seconds=f"{write_elapsed_seconds:.6f}",
            )
            processed_chunks = chunk_index + 1
            if _profiling_enabled() or logger.isEnabledFor(logging.INFO):
                elapsed_seconds = time.perf_counter() - streaming_started_at
                average_chunk_seconds = elapsed_seconds / processed_chunks
                remaining_chunks = max(total_chunks - processed_chunks, 0)
                eta_seconds = average_chunk_seconds * remaining_chunks
                logger.info(
                    "streaming candidate ranking progress "
                    f"restart={restart_index} "
                    f"chunk={processed_chunks}/{total_chunks} "
                    f"chunk_candidates={len(chunk_candidates)} "
                    f"candidates_seen={candidate_count} "
                    f"candidate_workers={worker_count} "
                    f"executor_mode={executor_mode} "
                    f"map_chunksize={map_chunksize} "
                    f"inflight_chunks={max(chunk_prefetch_depth, 1)} "
                    f"elapsed={_format_eta(elapsed_seconds)} "
                    f"eta={_format_eta(eta_seconds)}"
                )
    finally:
        if process_pool is not None:
            process_pool.shutdown()
        if thread_pool is not None:
            thread_pool.shutdown()

    return runs, candidate_count


def greedy_construct(
    instance: BenchmarkInstance,
    priority_fn: PriorityFn,
    restart_index: int,
    show_progress: bool = False,
    candidate_workers: int | None = None,
) -> SearchAttemptResult:
    """Run one fixed sorted-greedy pass for a single benchmark instance."""
    if _should_stream_candidates(instance.r):
        return _greedy_construct_streaming(
            instance,
            priority_fn,
            restart_index,
            candidate_workers=candidate_workers,
        )

    search_state = IncrementalForbiddenState(instance.r, instance.target_distance)
    ordered_candidates, ordered_scores = ranked_candidates(
        instance,
        priority_fn,
        restart_index,
        show_progress=show_progress,
        candidate_workers=candidate_workers,
    )
    blocked_candidate_count = 0
    illegal_weight_histogram: Counter[int] = Counter()

    greedy_started_at = time.perf_counter()
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
            illegal_weight_histogram[support_weight(candidate_mask, instance.r)] += 1

    selected = tuple(search_state.selected_free_columns)
    _log_profile(
        "greedy_scan",
        n=instance.n,
        k=instance.k,
        d=instance.target_distance,
        r=instance.r,
        restart=restart_index,
        candidate_count=len(ordered_candidates),
        selected_count=len(selected),
        blocked_count=blocked_candidate_count,
        elapsed_seconds=f"{time.perf_counter() - greedy_started_at:.6f}",
    )
    success = len(selected) == instance.k and validate_free_columns(
        instance.r,
        selected,
        instance.target_distance,
    )
    return SearchAttemptResult(
        success=success,
        selected_free_columns=selected,
        added_free_columns=len(selected),
        candidate_count=len(ordered_candidates),
        restart_index=restart_index,
        sorted_candidates=ordered_candidates,
        sorted_scores=ordered_scores,
        blocked_candidate_count=blocked_candidate_count,
        illegal_weight_histogram=tuple(sorted(illegal_weight_histogram.items())),
        chosen_weights=tuple(support_weight(mask, instance.r) for mask in selected),
    )


def _greedy_construct_streaming(
    instance: BenchmarkInstance,
    priority_fn: PriorityFn,
    restart_index: int,
    candidate_workers: int | None = None,
) -> SearchAttemptResult:
    """Run the exact greedy search while streaming sorted candidate runs from disk."""
    search_state = IncrementalForbiddenState(instance.r, instance.target_distance)
    runs, candidate_count = _ranked_candidate_runs(
        instance,
        priority_fn,
        restart_index,
        candidate_workers=candidate_workers,
        chunk_prefetch_depth=_candidate_chunk_prefetch_depth(candidate_workers),
    )
    blocked_candidate_count = 0
    illegal_weight_histogram: Counter[int] = Counter()
    top_ranked_scores: List[Tuple[int, float]] = []

    greedy_started_at = time.perf_counter()
    try:
        for score, _, candidate_mask in _merge_scored_chunk_runs(runs):
            if len(top_ranked_scores) < 10:
                top_ranked_scores.append((candidate_mask, score))
            if len(search_state.selected_free_columns) >= instance.k:
                if len(top_ranked_scores) >= 10:
                    break
                continue
            if search_state.can_add(candidate_mask):
                search_state.add(candidate_mask)
            else:
                blocked_candidate_count += 1
                illegal_weight_histogram[support_weight(candidate_mask, instance.r)] += 1
            if len(search_state.selected_free_columns) >= instance.k and len(top_ranked_scores) >= 10:
                break
    finally:
        _cleanup_scored_chunk_runs(runs)

    selected = tuple(search_state.selected_free_columns)
    _log_profile(
        "greedy_scan",
        n=instance.n,
        k=instance.k,
        d=instance.target_distance,
        r=instance.r,
        restart=restart_index,
        candidate_count=candidate_count,
        selected_count=len(selected),
        blocked_count=blocked_candidate_count,
        elapsed_seconds=f"{time.perf_counter() - greedy_started_at:.6f}",
    )
    success = len(selected) == instance.k and validate_free_columns(
        instance.r,
        selected,
        instance.target_distance,
    )
    return SearchAttemptResult(
        success=success,
        selected_free_columns=selected,
        added_free_columns=len(selected),
        candidate_count=candidate_count,
        restart_index=restart_index,
        sorted_candidates=tuple(mask for mask, _ in top_ranked_scores),
        sorted_scores=tuple(top_ranked_scores),
        blocked_candidate_count=blocked_candidate_count,
        illegal_weight_histogram=tuple(sorted(illegal_weight_histogram.items())),
        chosen_weights=tuple(support_weight(mask, instance.r) for mask in selected),
    )


def best_restart_for_instance(
    instance: BenchmarkInstance,
    priority_fn: PriorityFn,
    show_progress: bool = False,
) -> SearchAttemptResult:
    """Evaluate all fixed restarts and keep the best deterministic attempt."""
    parallelism_plan = _resolve_parallelism_plan(
        instance,
        show_progress=show_progress,
    )
    _log_profile(
        "parallelism_plan",
        n=instance.n,
        k=instance.k,
        d=instance.target_distance,
        r=instance.r,
        restarts=instance.restarts,
        restart_workers=parallelism_plan.restart_workers,
        candidate_workers=parallelism_plan.candidate_workers,
        chunk_prefetch_depth=parallelism_plan.chunk_prefetch_depth,
        cpu_budget=_cpu_budget(),
    )
    if show_progress or parallelism_plan.restart_workers <= 1:
        attempts = [
            greedy_construct(
                instance,
                priority_fn,
                restart_index,
                show_progress=show_progress,
                candidate_workers=parallelism_plan.candidate_workers,
            )
            for restart_index in range(instance.restarts)
        ]
    else:
        restart_jobs = [
            (
                instance,
                priority_fn,
                restart_index,
                parallelism_plan.candidate_workers,
            )
            for restart_index in range(instance.restarts)
        ]
        with ThreadPoolExecutor(
            max_workers=parallelism_plan.restart_workers,
            thread_name_prefix="linear-code-restart",
        ) as executor:
            attempts = list(executor.map(_run_restart_job, restart_jobs))
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
        raise AttributeError("Program must define a priority(column_code, n, k, d) function")
    return module.priority


def evaluate_priority_function(
    priority_fn: PriorityFn, instance: BenchmarkInstance | None = None
) -> EvaluationResult:
    """Run the fixed greedy search on one configurable instance."""
    active_instance = instance or DEFAULT_INSTANCE
    _log_profile(
        "evaluation_start",
        n=active_instance.n,
        k=active_instance.k,
        d=active_instance.target_distance,
        r=active_instance.r,
        restarts=active_instance.restarts,
    )
    started_at = time.perf_counter()
    attempt = best_restart_for_instance(active_instance, priority_fn)
    elapsed_seconds = time.perf_counter() - started_at
    progress = attempt.added_free_columns / active_instance.k
    combined_score = 1.0 if attempt.success else progress
    _log_profile(
        "evaluation_summary",
        n=active_instance.n,
        k=active_instance.k,
        d=active_instance.target_distance,
        r=active_instance.r,
        restarts=active_instance.restarts,
        success=int(attempt.success),
        constructed_columns=attempt.added_free_columns,
        blocked_candidates=attempt.blocked_candidate_count,
        elapsed_seconds=f"{elapsed_seconds:.6f}",
    )

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
            "evaluation_time_seconds": elapsed_seconds,
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
                    "candidate_count": attempt.candidate_count,
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
