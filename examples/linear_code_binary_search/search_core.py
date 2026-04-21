"""FunSearch-style static-priority search for binary matrix construction."""

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
_SCORED_RECORD_STRUCT = struct.Struct(">dQQ")
_PROCESS_PRIORITY_FN: PriorityFn | None = None


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


def _profiling_enabled() -> bool:
    """Return whether opt-in stage profiling logs should be emitted."""
    return _env_flag_enabled("LINEAR_CODE_PROFILE")


def _log_profile(stage: str, **fields: object) -> None:
    """Emit one structured stage log line when profiling is enabled."""
    if not _profiling_enabled():
        return
    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info(f"linear_code_profile stage={stage} {payload}".strip())


def _candidate_chunk_size() -> int:
    """Return the numeric mask-range chunk size for streaming candidate generation."""
    return _env_positive_int("LINEAR_CODE_CANDIDATE_CHUNK_SIZE") or (1 << 20)


def _streaming_mask_threshold() -> int:
    """Return the mask-space threshold above which candidate streaming is enabled by default."""
    return _env_positive_int("LINEAR_CODE_STREAMING_THRESHOLD_MASKS") or (1 << 22)


def _should_stream_candidates(r: int) -> bool:
    """Choose the candidate-processing strategy for the current redundancy."""
    if _env_flag_enabled("LINEAR_CODE_FORCE_STREAMING"):
        return True
    return (1 << r) > _streaming_mask_threshold()


def _candidate_executor_mode() -> str:
    """Return the requested candidate-scoring executor mode."""
    raw_value = os.environ.get("LINEAR_CODE_CANDIDATE_EXECUTOR", "thread")
    normalized = raw_value.strip().lower()
    if normalized in {"process", "thread"}:
        return normalized
    return "thread"


@lru_cache(maxsize=None)
def basis_columns(r: int) -> Tuple[int, ...]:
    """Systematic identity columns."""
    return tuple(1 << bit_index for bit_index in range(r))


@lru_cache(maxsize=None)
def candidate_masks(r: int, distance: int) -> Tuple[int, ...]:
    """All non-zero free columns that pass the initial weight filter."""
    min_weight = max(distance - 1, 1)
    return tuple(mask for mask in range(1, 1 << r) if popcount(mask) >= min_weight)


def candidate_mask_chunks(
    r: int,
    distance: int,
    chunk_size: int | None = None,
) -> Iterator[Tuple[int, ...]]:
    """Yield candidate masks in numeric-range chunks instead of one giant tuple."""
    min_weight = max(distance - 1, 1)
    resolved_chunk_size = chunk_size or _candidate_chunk_size()
    upper_bound = 1 << r
    for chunk_start in range(1, upper_bound, resolved_chunk_size):
        chunk_end = min(chunk_start + resolved_chunk_size, upper_bound)
        chunk = tuple(
            mask for mask in range(chunk_start, chunk_end) if popcount(mask) >= min_weight
        )
        if chunk:
            yield chunk


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
        return list(process_pool.map(_score_candidate_job_in_process, job_iter))

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
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="linear-code-score",
    ) as executor:
        return list(executor.map(_score_candidate_job, job_iter))


def _write_scored_chunk_run(records: Sequence[Tuple[float, int, int]]) -> ScoredChunkRun:
    """Persist one scored candidate chunk as a sorted binary run on disk."""
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
    job: Tuple[BenchmarkInstance, PriorityFn, int]
) -> SearchAttemptResult:
    """Run one restart in isolation, keeping inner candidate scoring serial."""
    instance, priority_fn, restart_index = job
    return greedy_construct(
        instance,
        priority_fn,
        restart_index,
        show_progress=False,
        candidate_workers=1,
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
) -> Tuple[List[ScoredChunkRun], int]:
    """Build sorted chunk runs for the exact streaming candidate path."""
    worker_count = candidate_workers
    runs: List[ScoredChunkRun] = []
    candidate_count = 0
    if worker_count is None:
        worker_count = _resolve_worker_count(
            _candidate_chunk_size(),
            env_name="LINEAR_CODE_CANDIDATE_WORKERS",
            minimum_parallel_tasks=64,
        )

    executor_mode = "thread"
    process_pool = None
    if _candidate_executor_mode() == "process":
        process_pool = _create_process_candidate_pool(priority_fn, worker_count)
        if process_pool is not None:
            executor_mode = "process"

    try:
        for chunk_index, chunk_candidates in enumerate(
            candidate_mask_chunks(instance.r, instance.target_distance, _candidate_chunk_size())
        ):
            generation_elapsed_seconds = 0.0
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
            scored_chunk = _score_candidate_chunk(
                chunk_candidates,
                instance,
                priority_fn,
                restart_index,
                worker_count,
                executor_mode=executor_mode,
                process_pool=process_pool,
            )
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
                elapsed_seconds=f"{time.perf_counter() - scoring_started_at:.6f}",
            )

            sort_started_at = time.perf_counter()
            scored_chunk.sort(reverse=True)
            _log_profile(
                "candidate_sort",
                n=instance.n,
                k=instance.k,
                d=instance.target_distance,
                r=instance.r,
                restart=restart_index,
                chunk_index=chunk_index,
                chunk_candidate_count=len(scored_chunk),
                elapsed_seconds=f"{time.perf_counter() - sort_started_at:.6f}",
            )
            runs.append(_write_scored_chunk_run(scored_chunk))
    finally:
        if process_pool is not None:
            process_pool.shutdown()

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
            illegal_weight_histogram[popcount(candidate_mask)] += 1

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
        chosen_weights=tuple(popcount(mask) for mask in selected),
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
                break
            if search_state.can_add(candidate_mask):
                search_state.add(candidate_mask)
            else:
                blocked_candidate_count += 1
                illegal_weight_histogram[popcount(candidate_mask)] += 1
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
        chosen_weights=tuple(popcount(mask) for mask in selected),
    )


def best_restart_for_instance(
    instance: BenchmarkInstance,
    priority_fn: PriorityFn,
    show_progress: bool = False,
) -> SearchAttemptResult:
    """Evaluate all fixed restarts and keep the best deterministic attempt."""
    restart_worker_count = _resolve_worker_count(
        instance.restarts,
        env_name="LINEAR_CODE_RESTART_WORKERS",
        minimum_parallel_tasks=2,
    )
    if show_progress or restart_worker_count <= 1:
        attempts = [
            greedy_construct(
                instance,
                priority_fn,
                restart_index,
                show_progress=show_progress,
            )
            for restart_index in range(instance.restarts)
        ]
    else:
        restart_jobs = [
            (instance, priority_fn, restart_index)
            for restart_index in range(instance.restarts)
        ]
        with ThreadPoolExecutor(
            max_workers=restart_worker_count,
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
        raise AttributeError("Program must define a priority(column_mask, n, k, d) function")
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
