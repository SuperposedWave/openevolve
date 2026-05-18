"""FunSearch-style static-priority search for binary matrix construction."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import importlib.machinery
import importlib.util
import json
import logging
import math
import multiprocessing
import os
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import random
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


logger = logging.getLogger(__name__)
PriorityFn = Callable[[int, int, int, int], float]
_PROCESS_PRIORITY_FN: PriorityFn | None = None
_NATIVE_MODULE = None


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
class AcceptedVectorRecord:
    """Analysis details for one free column accepted by the greedy scan."""

    fill_index: int
    rank: int
    column: str
    weight: int
    score: float
    rank_scope: str = "global"


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
    accepted_vectors: Tuple[AcceptedVectorRecord, ...]
    search_mode: str = "full"
    sample_attempt_count: int = 0
    sampled_candidate_count: int = 0
    scored_candidate_count: int = 0
    backtrack_events: int = 0
    backtracked_columns: int = 0
    beam_width: int = 0
    beam_expanded_states: int = 0
    legality_engine: str = "python"
    native_r_limit: int | None = None
    final_forbidden_count: int | None = None


@dataclass
class BeamSearchState:
    """One partial solution maintained by sampled beam search."""

    search_state: "IncrementalForbiddenState"
    accepted_vectors: Tuple[AcceptedVectorRecord, ...]
    adjusted_score: float
    priority_score: float
    tie_break: float


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

    def add(self, column_mask: int) -> int:
        if not self.can_add(column_mask):
            raise ValueError(f"Illegal free column {column_mask}")
        before_count = len(self.forbidden)
        _add_column_to_reachable(self.reachable, column_mask, self.max_subset_size)
        self.forbidden = set().union(*self.reachable)
        self.selected_free_columns.append(column_mask)
        return len(self.forbidden) - before_count


class NativeForbiddenStateAdapter:
    """Python adapter around the optional C exact legality engine."""

    engine_name = "native"

    def __init__(self, r: int, distance: int, native_state=None):
        self.r = r
        self.distance = distance
        self.max_subset_size = max(distance - 2, 0)
        self._native = native_state
        if self._native is None:
            native_module = _load_native_module()
            self._native = native_module.NativeForbiddenState(r, distance)
        self.selected_free_columns: List[int] = list(self._native.selected_columns())

    def can_add(self, column_mask: int) -> bool:
        return bool(self._native.can_add(column_mask))

    def add(self, column_mask: int) -> int:
        growth = int(self._native.add(column_mask))
        self.selected_free_columns.append(column_mask)
        return growth

    def undo(self, count: int) -> None:
        if count <= 0:
            return
        self._native.undo(count)
        del self.selected_free_columns[-count:]

    def clone(self) -> "NativeForbiddenStateAdapter":
        return NativeForbiddenStateAdapter(
            self.r,
            self.distance,
            native_state=self._native.clone(),
        )

    def forbidden_count(self) -> int:
        return int(self._native.forbidden_count())

    def layer_counts(self) -> Tuple[int, ...]:
        return tuple(int(value) for value in self._native.layer_counts())


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


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    """Parse an integer environment override with a lower bound."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return max(int(raw_value), minimum)
    except ValueError:
        return default


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    """Parse a float environment override with a lower bound."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return max(float(raw_value), minimum)
    except ValueError:
        return default


def _resolve_worker_count(task_count: int, env_name: str, minimum_parallel_tasks: int) -> int:
    """Choose a bounded thread count for a pool-backed parallel section."""
    override = _env_worker_override(env_name)
    if override is not None:
        return min(override, task_count)
    if task_count < max(minimum_parallel_tasks, 2):
        return 1
    cpu_count = os.cpu_count() or 1
    return min(cpu_count, task_count)


def _env_flag_enabled(name: str) -> bool:
    """Parse a conventional boolean environment flag."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return False
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _profiling_enabled() -> bool:
    """Return whether opt-in stage profiling logs should be emitted."""
    return _env_flag_enabled("LINEAR_CODE_PROFILE")


def _progress_enabled() -> bool:
    """Return whether user-facing progress bars should be shown."""
    return _env_flag_enabled("LINEAR_CODE_PROGRESS")


def _search_mode() -> str:
    """Return the requested inner search mode."""
    raw_value = os.environ.get("LINEAR_CODE_SEARCH_MODE", "full")
    normalized = raw_value.strip().lower().replace("-", "_")
    if normalized in {"sampled", "sampled_refill", "refill"}:
        return "sampled_refill"
    if normalized in {"sampled_beam", "beam"}:
        return "sampled_beam"
    return "full"


def _legality_engine_mode() -> str:
    """Return the requested exact legality engine."""
    raw_value = os.environ.get("LINEAR_CODE_LEGALITY_ENGINE", "python")
    normalized = raw_value.strip().lower().replace("-", "_")
    if normalized == "native":
        return "native"
    return "python"


def _load_native_module():
    """Load the optional CPython native legality module."""
    global _NATIVE_MODULE
    if _NATIVE_MODULE is not None:
        return _NATIVE_MODULE
    try:
        import _linear_code_native as native_module

        _NATIVE_MODULE = native_module
        return native_module
    except ImportError:
        pass

    search_roots = [Path(__file__).resolve().parents[2], Path(__file__).resolve().parent]
    for root in search_roots:
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            candidate = root / f"_linear_code_native{suffix}"
            if not candidate.exists():
                continue
            spec = importlib.util.spec_from_file_location("_linear_code_native", candidate)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules["_linear_code_native"] = module
            spec.loader.exec_module(module)
            _NATIVE_MODULE = module
            return module
    raise ImportError(
        "LINEAR_CODE_LEGALITY_ENGINE=native requested, but _linear_code_native is not built. "
        "Run `python setup.py build_ext --inplace` first."
    )


def _log_profile(stage: str, **fields: object) -> None:
    """Emit one structured stage log line when profiling is enabled."""
    if not _profiling_enabled():
        return
    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info(f"linear_code_profile stage={stage} {payload}".strip())


def _progress_message(show_progress: bool, message: str) -> None:
    """Emit a short user-facing progress message without breaking tqdm bars."""
    if not show_progress:
        return
    try:
        from tqdm import tqdm
    except Exception:
        print(message)
        return
    tqdm.write(message)


@lru_cache(maxsize=None)
def basis_columns(r: int) -> Tuple[int, ...]:
    """Systematic identity columns."""
    return tuple(1 << bit_index for bit_index in range(r))


@lru_cache(maxsize=None)
def candidate_masks(r: int, distance: int) -> Tuple[int, ...]:
    """All non-zero free columns that pass the initial weight filter."""
    min_weight = max(distance - 1, 1)
    return tuple(mask for mask in range(1, 1 << r) if popcount(mask) >= min_weight)


@lru_cache(maxsize=None)
def candidate_count(r: int, distance: int) -> int:
    """Count initially weight-eligible free columns without materializing them."""
    min_weight = max(distance - 1, 1)
    return sum(math.comb(r, weight) for weight in range(min_weight, r + 1))


@lru_cache(maxsize=None)
def candidate_weight_layer_counts(r: int, distance: int) -> Tuple[Tuple[int, int], ...]:
    """Return exact candidate counts for each initially eligible Hamming weight."""
    min_weight = max(distance - 1, 1)
    return tuple((weight, math.comb(r, weight)) for weight in range(min_weight, r + 1))


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


def _accepted_vector_record(
    candidate_mask: int,
    score: float,
    rank: int,
    fill_index: int,
    r: int,
    rank_scope: str = "global",
) -> AcceptedVectorRecord:
    """Create the shared accepted-vector analysis record for both scan paths."""
    return AcceptedVectorRecord(
        fill_index=fill_index,
        rank=rank,
        column=format_mask(candidate_mask, r),
        weight=popcount(candidate_mask),
        score=float(score),
        rank_scope=rank_scope,
    )


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


def _score_static_candidate_job(
    job: Tuple[PriorityFn, int, int, int, int]
) -> Tuple[int, float]:
    """Score one candidate column without restart-specific tie-break."""
    priority_fn, candidate_mask, n, k, d = job
    score = _safe_priority(priority_fn, candidate_mask, n, k, d)
    return candidate_mask, score


def _initialize_process_priority_fn(program_path: str) -> None:
    """Load the priority function inside a process worker."""
    global _PROCESS_PRIORITY_FN
    _PROCESS_PRIORITY_FN = load_priority_function(program_path)


def _score_static_candidate_job_in_process(
    job: Tuple[int, int, int, int]
) -> Tuple[int, float]:
    """Score one candidate column with the process-local priority function."""
    candidate_mask, n, k, d = job
    if _PROCESS_PRIORITY_FN is None:
        raise RuntimeError("Process scoring requested before initializing priority function")
    score = _safe_priority(_PROCESS_PRIORITY_FN, candidate_mask, n, k, d)
    return candidate_mask, score


def _priority_program_path(priority_fn: PriorityFn) -> str | None:
    """Return the source path backing a priority function when available."""
    code_object = getattr(priority_fn, "__code__", None)
    program_path = getattr(code_object, "co_filename", None)
    if isinstance(program_path, str) and os.path.exists(program_path):
        return program_path
    return None


def _candidate_executor_mode() -> str:
    """Return the requested candidate-scoring executor mode."""
    raw_value = os.environ.get("LINEAR_CODE_CANDIDATE_EXECUTOR", "thread")
    normalized = raw_value.strip().lower()
    if normalized in {"process", "thread"}:
        return normalized
    return "thread"


def _create_process_candidate_pool(
    priority_fn: PriorityFn,
    worker_count: int,
) -> ProcessPoolExecutor | None:
    """Create a process pool for static scoring when the priority source path is available."""
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


def _run_scored_restart_job(
    job: Tuple[BenchmarkInstance, Tuple[Tuple[int, float], ...], int]
) -> SearchAttemptResult:
    """Run one restart from shared static candidate scores."""
    instance, static_scores, restart_index = job
    return greedy_construct(
        instance,
        priority_fn=None,
        restart_index=restart_index,
        show_progress=False,
        candidate_workers=1,
        static_scores=static_scores,
    )


def _deterministic_tiebreak(candidate_mask: int, restart_index: int) -> int:
    """Restart-specific tie-break used only to perturb equal-score columns."""
    return (
        candidate_mask * 1103515245
        + restart_index * 2654435761
        + 12345
    ) & 0xFFFFFFFF


def _sampled_seed(restart_index: int) -> int:
    """Return the deterministic seed for one sampled restart."""
    base_seed = _env_int("LINEAR_CODE_RANDOM_SEED", 0, minimum=0)
    return (base_seed + restart_index * 1000003) & 0xFFFFFFFF


def _sample_weight(r: int, distance: int, rng: random.Random) -> int:
    """Draw a Hamming weight layer proportional to the layer's candidate count."""
    layer_counts = candidate_weight_layer_counts(r, distance)
    weights = [weight for weight, _ in layer_counts]
    counts = [count for _, count in layer_counts]
    return rng.choices(weights, weights=counts, k=1)[0]


def _random_mask_with_weight(r: int, weight: int, rng: random.Random) -> int:
    """Generate a uniformly random binary mask with exactly `weight` ones."""
    mask = 0
    for bit_index in rng.sample(range(r), weight):
        mask |= 1 << bit_index
    return mask


def _sampled_refill_pool_size(instance: BenchmarkInstance) -> int:
    """Return the target legal-candidate pool size for sampled refills."""
    default_size = max(256, min(8192, instance.k * 256))
    return _env_int("LINEAR_CODE_SAMPLE_POOL_SIZE", default_size)


def _sampled_refill_attempt_budget(pool_size: int) -> int:
    """Return the maximum random draws used to refill one sampled pool."""
    return _env_int("LINEAR_CODE_SAMPLE_ATTEMPTS_PER_REFILL", pool_size * 20)


def _sampled_max_refills(instance: BenchmarkInstance) -> int:
    """Return the maximum number of sampled pool refills per restart."""
    return _env_int("LINEAR_CODE_SAMPLE_MAX_REFILLS", max(16, instance.k * 8))


def _sampled_max_stale_refills() -> int:
    """Return how many no-progress refills are allowed before abandoning a restart."""
    return _env_int("LINEAR_CODE_SAMPLE_MAX_STALE_REFILLS", 4)


def _backtrack_depth() -> int:
    """Return how many recent accepted columns to remove on sampled-refill backtracking."""
    return _env_int("LINEAR_CODE_BACKTRACK_DEPTH", 2, minimum=0)


def _backtrack_max_events() -> int:
    """Return the maximum number of sampled-refill backtracking events per restart."""
    return _env_int("LINEAR_CODE_BACKTRACK_MAX_EVENTS", 4, minimum=0)


def _beam_width() -> int:
    """Return the sampled beam width."""
    return _env_int("LINEAR_CODE_BEAM_WIDTH", 8)


def _beam_branches_per_state() -> int:
    """Return the maximum number of branch candidates kept from each beam state."""
    return _env_int("LINEAR_CODE_BEAM_BRANCHES_PER_STATE", 128)


def _beam_attempts_per_state() -> int:
    """Return the maximum random draws used to expand one beam state."""
    return _env_int("LINEAR_CODE_BEAM_ATTEMPTS_PER_STATE", 2560)


def _beam_forbidden_penalty() -> float:
    """Return the forbidden-growth penalty used by sampled beam scoring."""
    return _env_float("LINEAR_CODE_BEAM_FORBIDDEN_PENALTY", 1.0)


def _native_r_limit() -> int | None:
    """Return the native engine r limit when available and requested."""
    if _legality_engine_mode() != "native":
        return None
    native_module = _load_native_module()
    return int(native_module.NATIVE_R_LIMIT)


def create_forbidden_state(r: int, distance: int):
    """Create the configured exact forbidden-state engine."""
    if _legality_engine_mode() == "native":
        return NativeForbiddenStateAdapter(r, distance)
    return IncrementalForbiddenState(r, distance)


def _state_engine_name(search_state) -> str:
    """Return the implementation name for a forbidden state."""
    return getattr(search_state, "engine_name", "python")


def _state_forbidden_count(search_state) -> int | None:
    """Return forbidden count from either Python or native state."""
    if hasattr(search_state, "forbidden_count"):
        return int(search_state.forbidden_count())
    forbidden = getattr(search_state, "forbidden", None)
    if forbidden is not None:
        return len(forbidden)
    return None


def _state_add(search_state, column_mask: int) -> int:
    """Add a column and return the exact forbidden-set growth."""
    result = search_state.add(column_mask)
    return int(result) if result is not None else 0


def _clone_search_state(search_state):
    """Deep-copy an incremental forbidden state."""
    if hasattr(search_state, "clone"):
        return search_state.clone()
    cloned = IncrementalForbiddenState(search_state.r, search_state.distance)
    cloned.reachable = [set(layer) for layer in search_state.reachable]
    cloned.forbidden = set(search_state.forbidden)
    cloned.selected_free_columns = list(search_state.selected_free_columns)
    return cloned


def _rebuild_search_state(
    r: int,
    distance: int,
    selected_free_columns: Sequence[int],
):
    """Rebuild an incremental forbidden state from selected free columns."""
    rebuilt = create_forbidden_state(r, distance)
    for column_mask in selected_free_columns:
        _state_add(rebuilt, column_mask)
    return rebuilt


def _validate_selected_free_columns(
    r: int,
    free_columns: Sequence[int],
    distance: int,
) -> bool:
    """Validate free columns using the configured exact engine."""
    if _legality_engine_mode() == "native":
        native_module = _load_native_module()
        return bool(native_module.validate_columns(r, distance, tuple(free_columns)))
    return validate_free_columns(r, free_columns, distance)


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


def score_static_candidates(
    instance: BenchmarkInstance,
    priority_fn: PriorityFn,
    show_progress: bool = False,
    candidate_workers: int | None = None,
) -> Tuple[Tuple[int, float], ...]:
    """Compute restart-independent priority scores once for all candidate columns."""
    started_at = time.perf_counter()
    candidates = candidate_masks(instance.r, instance.target_distance)
    _log_profile(
        "candidate_generation",
        n=instance.n,
        k=instance.k,
        d=instance.target_distance,
        r=instance.r,
        restart="shared",
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

    executor_mode = "thread"
    process_pool = None
    if _candidate_executor_mode() == "process":
        process_pool = _create_process_candidate_pool(priority_fn, worker_count)
        if process_pool is not None:
            executor_mode = "process"

    scoring_started_at = time.perf_counter()
    try:
        if worker_count <= 1:
            scored_candidates = tuple(
                _score_static_candidate_job(
                    (
                        priority_fn,
                        candidate_mask,
                        instance.n,
                        instance.k,
                        instance.target_distance,
                    )
                )
                for candidate_mask in _iterate_with_progress(
                    candidates,
                    "shared candidate scoring",
                    show_progress,
                    total=len(candidates),
                )
            )
        elif executor_mode == "process" and process_pool is not None:
            jobs = [
                (
                    candidate_mask,
                    instance.n,
                    instance.k,
                    instance.target_distance,
                )
                for candidate_mask in candidates
            ]
            scored_candidates = tuple(
                _iterate_with_progress(
                    process_pool.map(_score_static_candidate_job_in_process, jobs),
                    "shared candidate scoring",
                    show_progress,
                    total=len(candidates),
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
                )
                for candidate_mask in candidates
            ]
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="linear-code-score",
            ) as executor:
                scored_candidates = tuple(
                    _iterate_with_progress(
                        executor.map(_score_static_candidate_job, jobs),
                        "shared candidate scoring",
                        show_progress,
                        total=len(candidates),
                    )
                )
    finally:
        if process_pool is not None:
            process_pool.shutdown()
    _log_profile(
        "candidate_scoring",
        n=instance.n,
        k=instance.k,
        d=instance.target_distance,
        r=instance.r,
        restart="shared",
        candidate_count=len(candidates),
        worker_count=worker_count,
        executor_mode=executor_mode,
        elapsed_seconds=f"{time.perf_counter() - scoring_started_at:.6f}",
    )
    return scored_candidates


def ranked_candidates_from_scores(
    instance: BenchmarkInstance,
    static_scores: Sequence[Tuple[int, float]],
    restart_index: int,
) -> Tuple[Tuple[int, ...], Tuple[Tuple[int, float], ...]]:
    """Apply restart-specific tie-breaks to cached static candidate scores."""
    sort_started_at = time.perf_counter()
    scored_candidates = [
        (
            score,
            _deterministic_tiebreak(candidate_mask, restart_index),
            candidate_mask,
        )
        for candidate_mask, score in static_scores
    ]
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


def ranked_candidates(
    instance: BenchmarkInstance,
    priority_fn: PriorityFn,
    restart_index: int,
    show_progress: bool = False,
    candidate_workers: int | None = None,
) -> Tuple[Tuple[int, ...], Tuple[Tuple[int, float], ...]]:
    """Compute a single restart-specific ordering of all candidate columns."""
    static_scores = score_static_candidates(
        instance,
        priority_fn,
        show_progress=show_progress,
        candidate_workers=candidate_workers,
    )
    return ranked_candidates_from_scores(instance, static_scores, restart_index)


def greedy_construct(
    instance: BenchmarkInstance,
    priority_fn: PriorityFn | None,
    restart_index: int,
    show_progress: bool = False,
    candidate_workers: int | None = None,
    static_scores: Sequence[Tuple[int, float]] | None = None,
) -> SearchAttemptResult:
    """Run one fixed sorted-greedy pass for a single benchmark instance."""
    search_state = create_forbidden_state(instance.r, instance.target_distance)
    if static_scores is None:
        if priority_fn is None:
            raise ValueError("priority_fn is required when static_scores are not provided")
        ordered_candidates, ordered_scores = ranked_candidates(
            instance,
            priority_fn,
            restart_index,
            show_progress=show_progress,
            candidate_workers=candidate_workers,
        )
    else:
        ordered_candidates, ordered_scores = ranked_candidates_from_scores(
            instance,
            static_scores,
            restart_index,
        )
    blocked_candidate_count = 0
    illegal_weight_histogram: Counter[int] = Counter()
    accepted_vectors: List[AcceptedVectorRecord] = []

    greedy_started_at = time.perf_counter()
    for rank, (candidate_mask, score) in enumerate(
        _iterate_with_progress(
            ordered_scores,
            f"greedy restart {restart_index}",
            show_progress,
        ),
        start=1,
    ):
        if len(search_state.selected_free_columns) >= instance.k:
            break
        if search_state.can_add(candidate_mask):
            _state_add(search_state, candidate_mask)
            accepted_vectors.append(
                _accepted_vector_record(
                    candidate_mask,
                    score,
                    rank,
                    len(search_state.selected_free_columns),
                    instance.r,
                )
            )
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
    success = len(selected) == instance.k and _validate_selected_free_columns(
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
        accepted_vectors=tuple(accepted_vectors),
        search_mode="full",
        sample_attempt_count=0,
        sampled_candidate_count=0,
        scored_candidate_count=len(ordered_scores),
        backtrack_events=0,
        backtracked_columns=0,
        beam_width=0,
        beam_expanded_states=0,
        legality_engine=_state_engine_name(search_state),
        native_r_limit=_native_r_limit(),
        final_forbidden_count=_state_forbidden_count(search_state),
    )


def sampled_refill_greedy_construct(
    instance: BenchmarkInstance,
    priority_fn: PriorityFn,
    restart_index: int,
    show_progress: bool = False,
) -> SearchAttemptResult:
    """Run priority-guided randomized greedy search without enumerating all candidates."""
    search_state = create_forbidden_state(instance.r, instance.target_distance)
    rng = random.Random(_sampled_seed(restart_index))
    pool_size = _sampled_refill_pool_size(instance)
    attempt_budget = _sampled_refill_attempt_budget(pool_size)
    max_refills = _sampled_max_refills(instance)
    max_stale_refills = _sampled_max_stale_refills()
    total_candidate_count = candidate_count(instance.r, instance.target_distance)

    seen: set[int] = set()
    blocked_candidate_count = 0
    sample_attempt_count = 0
    illegal_weight_histogram: Counter[int] = Counter()
    accepted_vectors: List[AcceptedVectorRecord] = []
    scored_candidates_seen: List[Tuple[int, float]] = []
    tabu_columns: set[int] = set()
    stale_refills = 0
    backtrack_events = 0
    backtracked_columns = 0
    backtrack_depth = _backtrack_depth()
    max_backtrack_events = _backtrack_max_events()

    greedy_started_at = time.perf_counter()
    _progress_message(
        show_progress,
        (
            f"restart {restart_index}: start sampled_refill "
            f"pool_size={pool_size} attempts_per_refill={attempt_budget} "
            f"max_refills={max_refills}"
        ),
    )
    for refill_index in range(1, max_refills + 1):
        if len(search_state.selected_free_columns) >= instance.k:
            break
        if len(seen) >= total_candidate_count:
            break

        _progress_message(
            show_progress,
            (
                f"restart {restart_index}: refill {refill_index}/{max_refills} "
                f"sample selected={len(search_state.selected_free_columns)}/{instance.k} "
                f"seen={len(seen)}/{total_candidate_count}"
            ),
        )
        pool: List[Tuple[float, float, int]] = []
        refill_attempts = 0
        while (
            len(pool) < pool_size
            and refill_attempts < attempt_budget
            and len(seen) < total_candidate_count
        ):
            refill_attempts += 1
            sample_attempt_count += 1
            weight = _sample_weight(instance.r, instance.target_distance, rng)
            candidate_mask = _random_mask_with_weight(instance.r, weight, rng)
            if candidate_mask in tabu_columns:
                continue
            if candidate_mask in seen:
                continue
            seen.add(candidate_mask)
            if not search_state.can_add(candidate_mask):
                blocked_candidate_count += 1
                illegal_weight_histogram[popcount(candidate_mask)] += 1
                continue
            score = _safe_priority(
                priority_fn,
                candidate_mask,
                instance.n,
                instance.k,
                instance.target_distance,
            )
            scored_candidates_seen.append((candidate_mask, score))
            pool.append((score, rng.random(), candidate_mask))

        _progress_message(
            show_progress,
            (
                f"restart {restart_index}: refill {refill_index}/{max_refills} "
                f"sampled pool={len(pool)} attempts={refill_attempts} "
                f"seen={len(seen)} blocked={blocked_candidate_count}"
            ),
        )
        if not pool:
            stale_refills += 1
            if stale_refills >= max_stale_refills:
                if (
                    backtrack_depth > 0
                    and backtrack_events < max_backtrack_events
                    and search_state.selected_free_columns
                ):
                    remove_count = min(
                        backtrack_depth,
                        len(search_state.selected_free_columns),
                        len(accepted_vectors),
                    )
                    removed = tuple(search_state.selected_free_columns[-remove_count:])
                    tabu_columns.update(removed)
                    if hasattr(search_state, "undo"):
                        search_state.undo(remove_count)
                    else:
                        remaining = tuple(search_state.selected_free_columns[:-remove_count])
                        search_state = _rebuild_search_state(
                            instance.r,
                            instance.target_distance,
                            remaining,
                        )
                    del accepted_vectors[-remove_count:]
                    backtrack_events += 1
                    backtracked_columns += remove_count
                    stale_refills = 0
                    _progress_message(
                        show_progress,
                        (
                            f"restart {restart_index}: backtrack removed={remove_count} "
                            f"selected={len(search_state.selected_free_columns)}/{instance.k} "
                            f"events={backtrack_events}/{max_backtrack_events}"
                        ),
                    )
                    continue
                break
            continue

        _progress_message(
            show_progress,
            f"restart {restart_index}: refill {refill_index}/{max_refills} sort_and_greedy pool={len(pool)}",
        )
        pool.sort(reverse=True)
        accepted_this_refill = 0
        for pool_rank, (score, _, candidate_mask) in enumerate(pool, start=1):
            if len(search_state.selected_free_columns) >= instance.k:
                break
            if search_state.can_add(candidate_mask):
                _state_add(search_state, candidate_mask)
                accepted_this_refill += 1
                accepted_vectors.append(
                    _accepted_vector_record(
                        candidate_mask,
                        score,
                        pool_rank,
                        len(search_state.selected_free_columns),
                        instance.r,
                        rank_scope="sampled_pool",
                    )
                )
            else:
                blocked_candidate_count += 1
                illegal_weight_histogram[popcount(candidate_mask)] += 1

        _progress_message(
            show_progress,
            (
                f"restart {restart_index}: refill {refill_index}/{max_refills} "
                f"accepted={accepted_this_refill} "
                f"selected={len(search_state.selected_free_columns)}/{instance.k} "
                f"stale={stale_refills if accepted_this_refill == 0 else 0}"
            ),
        )
        if accepted_this_refill:
            stale_refills = 0
        else:
            stale_refills += 1
            if stale_refills >= max_stale_refills:
                if (
                    backtrack_depth > 0
                    and backtrack_events < max_backtrack_events
                    and search_state.selected_free_columns
                ):
                    remove_count = min(
                        backtrack_depth,
                        len(search_state.selected_free_columns),
                        len(accepted_vectors),
                    )
                    removed = tuple(search_state.selected_free_columns[-remove_count:])
                    tabu_columns.update(removed)
                    if hasattr(search_state, "undo"):
                        search_state.undo(remove_count)
                    else:
                        remaining = tuple(search_state.selected_free_columns[:-remove_count])
                        search_state = _rebuild_search_state(
                            instance.r,
                            instance.target_distance,
                            remaining,
                        )
                    del accepted_vectors[-remove_count:]
                    backtrack_events += 1
                    backtracked_columns += remove_count
                    stale_refills = 0
                    _progress_message(
                        show_progress,
                        (
                            f"restart {restart_index}: backtrack removed={remove_count} "
                            f"selected={len(search_state.selected_free_columns)}/{instance.k} "
                            f"events={backtrack_events}/{max_backtrack_events}"
                        ),
                    )
                    continue
                break

    selected = tuple(search_state.selected_free_columns)
    top_sampled_scores = sorted(
        scored_candidates_seen,
        key=lambda item: (
            item[1],
            _deterministic_tiebreak(item[0], restart_index),
            item[0],
        ),
        reverse=True,
    )
    success = len(selected) == instance.k and _validate_selected_free_columns(
        instance.r,
        selected,
        instance.target_distance,
    )
    _log_profile(
        "sampled_refill_greedy_scan",
        n=instance.n,
        k=instance.k,
        d=instance.target_distance,
        r=instance.r,
        restart=restart_index,
        candidate_count=total_candidate_count,
        sampled_candidates=len(seen),
        scored_candidates=len(scored_candidates_seen),
        sample_attempts=sample_attempt_count,
        selected_count=len(selected),
        blocked_count=blocked_candidate_count,
        backtrack_events=backtrack_events,
        backtracked_columns=backtracked_columns,
        elapsed_seconds=f"{time.perf_counter() - greedy_started_at:.6f}",
    )
    _progress_message(
        show_progress,
        (
            f"restart {restart_index}: finish success={int(success)} "
            f"selected={len(selected)}/{instance.k} sampled={len(seen)} "
            f"scored={len(scored_candidates_seen)} attempts={sample_attempt_count} "
            f"backtracks={backtrack_events}"
        ),
    )
    return SearchAttemptResult(
        success=success,
        selected_free_columns=selected,
        added_free_columns=len(selected),
        candidate_count=total_candidate_count,
        restart_index=restart_index,
        sorted_candidates=tuple(mask for mask, _ in top_sampled_scores),
        sorted_scores=tuple(top_sampled_scores),
        blocked_candidate_count=blocked_candidate_count,
        illegal_weight_histogram=tuple(sorted(illegal_weight_histogram.items())),
        chosen_weights=tuple(popcount(mask) for mask in selected),
        accepted_vectors=tuple(accepted_vectors),
        search_mode="sampled_refill",
        sample_attempt_count=sample_attempt_count,
        sampled_candidate_count=len(seen),
        scored_candidate_count=len(scored_candidates_seen),
        backtrack_events=backtrack_events,
        backtracked_columns=backtracked_columns,
        beam_width=0,
        beam_expanded_states=0,
        legality_engine=_state_engine_name(search_state),
        native_r_limit=_native_r_limit(),
        final_forbidden_count=_state_forbidden_count(search_state),
    )


def _run_sampled_restart_job(
    job: Tuple[BenchmarkInstance, PriorityFn, int]
) -> SearchAttemptResult:
    """Run one sampled restart."""
    instance, priority_fn, restart_index = job
    return sampled_refill_greedy_construct(
        instance,
        priority_fn,
        restart_index=restart_index,
        show_progress=False,
    )


def _beam_state_key(state: BeamSearchState) -> Tuple[int, float, float, float]:
    """Sort key for keeping the most promising beam states."""
    return (
        len(state.search_state.selected_free_columns),
        state.adjusted_score,
        state.priority_score,
        state.tie_break,
    )


def sampled_beam_construct(
    instance: BenchmarkInstance,
    priority_fn: PriorityFn,
    restart_index: int,
    show_progress: bool = False,
) -> SearchAttemptResult:
    """Run priority-guided sampled beam search without enumerating all candidates."""
    rng = random.Random(_sampled_seed(restart_index))
    width = _beam_width()
    branches_per_state = _beam_branches_per_state()
    attempts_per_state = _beam_attempts_per_state()
    forbidden_penalty = _beam_forbidden_penalty()
    total_candidate_count = candidate_count(instance.r, instance.target_distance)
    forbidden_normalizer = float(max(1, 1 << instance.r))

    beam_states = [
        BeamSearchState(
            search_state=create_forbidden_state(instance.r, instance.target_distance),
            accepted_vectors=tuple(),
            adjusted_score=0.0,
            priority_score=0.0,
            tie_break=rng.random(),
        )
    ]
    seen_masks: set[int] = set()
    scored_candidates_seen: List[Tuple[int, float]] = []
    illegal_weight_histogram: Counter[int] = Counter()
    blocked_candidate_count = 0
    sample_attempt_count = 0
    expanded_states = 0
    started_at = time.perf_counter()

    _progress_message(
        show_progress,
        (
            f"restart {restart_index}: start sampled_beam width={width} "
            f"branches_per_state={branches_per_state} attempts_per_state={attempts_per_state}"
        ),
    )
    for depth in range(1, instance.k + 1):
        if any(len(state.search_state.selected_free_columns) >= instance.k for state in beam_states):
            break
        next_states: List[BeamSearchState] = []
        _progress_message(
            show_progress,
            (
                f"restart {restart_index}: beam depth {depth}/{instance.k} "
                f"states={len(beam_states)} seen={len(seen_masks)}/{total_candidate_count}"
            ),
        )
        for state_index, state in enumerate(beam_states):
            if len(state.search_state.selected_free_columns) >= instance.k:
                next_states.append(state)
                continue

            pool: List[Tuple[float, float, float, int, IncrementalForbiddenState, int]] = []
            local_seen: set[int] = set()
            attempts = 0
            while len(pool) < branches_per_state and attempts < attempts_per_state:
                attempts += 1
                sample_attempt_count += 1
                weight = _sample_weight(instance.r, instance.target_distance, rng)
                candidate_mask = _random_mask_with_weight(instance.r, weight, rng)
                if candidate_mask in local_seen:
                    continue
                local_seen.add(candidate_mask)
                seen_masks.add(candidate_mask)
                if not state.search_state.can_add(candidate_mask):
                    blocked_candidate_count += 1
                    illegal_weight_histogram[popcount(candidate_mask)] += 1
                    continue

                priority_score = _safe_priority(
                    priority_fn,
                    candidate_mask,
                    instance.n,
                    instance.k,
                    instance.target_distance,
                )
                scored_candidates_seen.append((candidate_mask, priority_score))
                next_search_state = _clone_search_state(state.search_state)
                before_forbidden = _state_forbidden_count(next_search_state) or 0
                add_growth = _state_add(next_search_state, candidate_mask)
                after_forbidden = _state_forbidden_count(next_search_state)
                forbidden_growth = (
                    add_growth
                    if after_forbidden is None
                    else after_forbidden - before_forbidden
                )
                extension_score = (
                    priority_score
                    - forbidden_penalty * (forbidden_growth / forbidden_normalizer)
                )
                pool.append(
                    (
                        extension_score,
                        priority_score,
                        rng.random(),
                        candidate_mask,
                        next_search_state,
                        forbidden_growth,
                    )
                )

            pool.sort(reverse=True)
            _progress_message(
                show_progress,
                (
                    f"restart {restart_index}: beam depth {depth}/{instance.k} "
                    f"state={state_index + 1}/{len(beam_states)} "
                    f"pool={len(pool)} attempts={attempts}"
                ),
            )
            for pool_rank, (
                extension_score,
                priority_score,
                tie_break,
                candidate_mask,
                next_search_state,
                _,
            ) in enumerate(pool, start=1):
                accepted_vectors = state.accepted_vectors + (
                    _accepted_vector_record(
                        candidate_mask,
                        priority_score,
                        pool_rank,
                        len(next_search_state.selected_free_columns),
                        instance.r,
                        rank_scope="sampled_beam_pool",
                    ),
                )
                next_states.append(
                    BeamSearchState(
                        search_state=next_search_state,
                        accepted_vectors=accepted_vectors,
                        adjusted_score=state.adjusted_score + extension_score,
                        priority_score=state.priority_score + priority_score,
                        tie_break=tie_break,
                    )
                )
                expanded_states += 1

        if not next_states:
            break
        next_states.sort(key=_beam_state_key, reverse=True)
        beam_states = next_states[:width]
        _progress_message(
            show_progress,
            (
                f"restart {restart_index}: beam depth {depth}/{instance.k} "
                f"kept={len(beam_states)} best_selected="
                f"{len(beam_states[0].search_state.selected_free_columns)}/{instance.k}"
            ),
        )

    best_state = max(beam_states, key=_beam_state_key)
    selected = tuple(best_state.search_state.selected_free_columns)
    success = len(selected) == instance.k and _validate_selected_free_columns(
        instance.r,
        selected,
        instance.target_distance,
    )
    top_sampled_scores = sorted(
        scored_candidates_seen,
        key=lambda item: (
            item[1],
            _deterministic_tiebreak(item[0], restart_index),
            item[0],
        ),
        reverse=True,
    )
    _log_profile(
        "sampled_beam_scan",
        n=instance.n,
        k=instance.k,
        d=instance.target_distance,
        r=instance.r,
        restart=restart_index,
        candidate_count=total_candidate_count,
        sampled_candidates=len(seen_masks),
        scored_candidates=len(scored_candidates_seen),
        sample_attempts=sample_attempt_count,
        selected_count=len(selected),
        blocked_count=blocked_candidate_count,
        beam_width=width,
        beam_expanded_states=expanded_states,
        elapsed_seconds=f"{time.perf_counter() - started_at:.6f}",
    )
    _progress_message(
        show_progress,
        (
            f"restart {restart_index}: finish sampled_beam success={int(success)} "
            f"selected={len(selected)}/{instance.k} sampled={len(seen_masks)} "
            f"scored={len(scored_candidates_seen)} expanded={expanded_states}"
        ),
    )
    return SearchAttemptResult(
        success=success,
        selected_free_columns=selected,
        added_free_columns=len(selected),
        candidate_count=total_candidate_count,
        restart_index=restart_index,
        sorted_candidates=tuple(mask for mask, _ in top_sampled_scores),
        sorted_scores=tuple(top_sampled_scores),
        blocked_candidate_count=blocked_candidate_count,
        illegal_weight_histogram=tuple(sorted(illegal_weight_histogram.items())),
        chosen_weights=tuple(popcount(mask) for mask in selected),
        accepted_vectors=best_state.accepted_vectors,
        search_mode="sampled_beam",
        sample_attempt_count=sample_attempt_count,
        sampled_candidate_count=len(seen_masks),
        scored_candidate_count=len(scored_candidates_seen),
        backtrack_events=0,
        backtracked_columns=0,
        beam_width=width,
        beam_expanded_states=expanded_states,
        legality_engine=_state_engine_name(best_state.search_state),
        native_r_limit=_native_r_limit(),
        final_forbidden_count=_state_forbidden_count(best_state.search_state),
    )


def _run_beam_restart_job(
    job: Tuple[BenchmarkInstance, PriorityFn, int]
) -> SearchAttemptResult:
    """Run one sampled beam restart."""
    instance, priority_fn, restart_index = job
    return sampled_beam_construct(
        instance,
        priority_fn,
        restart_index=restart_index,
        show_progress=False,
    )


def best_restart_for_instance(
    instance: BenchmarkInstance,
    priority_fn: PriorityFn,
    show_progress: bool = False,
) -> SearchAttemptResult:
    """Evaluate all fixed restarts and keep the best deterministic attempt."""
    search_mode = _search_mode()
    if search_mode == "sampled_beam":
        restart_worker_count = _resolve_worker_count(
            instance.restarts,
            env_name="LINEAR_CODE_RESTART_WORKERS",
            minimum_parallel_tasks=2,
        )
        if show_progress or restart_worker_count <= 1:
            attempts = [
                sampled_beam_construct(
                    instance,
                    priority_fn,
                    restart_index=restart_index,
                    show_progress=show_progress,
                )
                for restart_index in _iterate_with_progress(
                    range(instance.restarts),
                    "sampled beam restarts",
                    show_progress,
                    total=instance.restarts,
                )
            ]
        else:
            restart_jobs = [
                (instance, priority_fn, restart_index)
                for restart_index in range(instance.restarts)
            ]
            with ThreadPoolExecutor(
                max_workers=restart_worker_count,
                thread_name_prefix="linear-code-beam-restart",
            ) as executor:
                attempts = list(executor.map(_run_beam_restart_job, restart_jobs))
        return max(
            attempts,
            key=lambda attempt: (
                int(attempt.success),
                attempt.added_free_columns,
                attempt.beam_expanded_states,
                -attempt.sample_attempt_count,
                -attempt.restart_index,
            ),
        )

    if search_mode == "sampled_refill":
        restart_worker_count = _resolve_worker_count(
            instance.restarts,
            env_name="LINEAR_CODE_RESTART_WORKERS",
            minimum_parallel_tasks=2,
        )
        if show_progress or restart_worker_count <= 1:
            attempts = [
                sampled_refill_greedy_construct(
                    instance,
                    priority_fn,
                    restart_index=restart_index,
                    show_progress=show_progress,
                )
                for restart_index in _iterate_with_progress(
                    range(instance.restarts),
                    "sampled restarts",
                    show_progress,
                    total=instance.restarts,
                )
            ]
        else:
            restart_jobs = [
                (instance, priority_fn, restart_index)
                for restart_index in range(instance.restarts)
            ]
            with ThreadPoolExecutor(
                max_workers=restart_worker_count,
                thread_name_prefix="linear-code-sampled-restart",
            ) as executor:
                attempts = list(executor.map(_run_sampled_restart_job, restart_jobs))
        return max(
            attempts,
            key=lambda attempt: (
                int(attempt.success),
                attempt.added_free_columns,
                -attempt.sample_attempt_count,
                -attempt.restart_index,
            ),
        )

    restart_worker_count = _resolve_worker_count(
        instance.restarts,
        env_name="LINEAR_CODE_RESTART_WORKERS",
        minimum_parallel_tasks=2,
    )
    static_scores = score_static_candidates(
        instance,
        priority_fn,
        show_progress=show_progress,
    )
    if show_progress or restart_worker_count <= 1:
        attempts = [
            greedy_construct(
                instance,
                priority_fn=None,
                restart_index=restart_index,
                show_progress=show_progress,
                static_scores=static_scores,
            )
            for restart_index in _iterate_with_progress(
                range(instance.restarts),
                "full restarts",
                show_progress,
                total=instance.restarts,
            )
        ]
    else:
        restart_jobs = [
            (instance, static_scores, restart_index)
            for restart_index in range(instance.restarts)
        ]
        with ThreadPoolExecutor(
            max_workers=restart_worker_count,
            thread_name_prefix="linear-code-restart",
        ) as executor:
            attempts = list(executor.map(_run_scored_restart_job, restart_jobs))
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


def _accepted_vector_to_dict(record: AcceptedVectorRecord) -> dict[str, int | float | str]:
    """Convert an accepted-vector record to artifact JSON shape."""
    return {
        "fill_index": record.fill_index,
        "rank": record.rank,
        "rank_scope": record.rank_scope,
        "column": record.column,
        "weight": record.weight,
        "score": record.score,
    }


def _successful_code_summary(
    active_instance: BenchmarkInstance,
    attempt: SearchAttemptResult,
) -> dict[str, int | float | str | dict[int, int]]:
    """Summarize accepted-vector rank and weight statistics for a successful code."""
    ranks = [record.rank for record in attempt.accepted_vectors]
    weight_counter = Counter(record.weight for record in attempt.accepted_vectors)
    return {
        "n": active_instance.n,
        "k": active_instance.k,
        "d": active_instance.target_distance,
        "r": active_instance.r,
        "restart": attempt.restart_index,
        "search_mode": attempt.search_mode,
        "vector_count": len(attempt.accepted_vectors),
        "rank_min": min(ranks),
        "rank_max": max(ranks),
        "rank_avg": sum(ranks) / len(ranks),
        "weight_histogram": dict(sorted(weight_counter.items())),
    }


def evaluate_priority_function(
    priority_fn: PriorityFn,
    instance: BenchmarkInstance | None = None,
    show_progress: bool | None = None,
) -> EvaluationResult:
    """Run the fixed greedy search on one configurable instance."""
    active_instance = instance or DEFAULT_INSTANCE
    active_show_progress = _progress_enabled() if show_progress is None else show_progress
    _log_profile(
        "evaluation_start",
        n=active_instance.n,
        k=active_instance.k,
        d=active_instance.target_distance,
        r=active_instance.r,
        restarts=active_instance.restarts,
    )
    started_at = time.perf_counter()
    attempt = best_restart_for_instance(
        active_instance,
        priority_fn,
        show_progress=active_show_progress,
    )
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
    parity_rows = parity_check_matrix_rows(
        active_instance.r,
        attempt.selected_free_columns,
    )
    generator_rows = generator_matrix_rows(
        active_instance.r,
        attempt.selected_free_columns,
    )
    matrix_summary = {
        "form": "H=[P^T|I_r], G=[I_k|P]",
        "complete": attempt.added_free_columns == active_instance.k,
        "n": active_instance.n,
        "k": active_instance.k,
        "d": active_instance.target_distance,
        "r": active_instance.r,
        "filled_free_columns": attempt.added_free_columns,
        "target_free_columns": active_instance.k,
        "h_shape": [active_instance.r, attempt.added_free_columns + active_instance.r],
        "g_shape": [
            attempt.added_free_columns,
            attempt.added_free_columns + active_instance.r,
        ],
        "selected_free_columns": [
            format_mask(mask, active_instance.r)
            for mask in attempt.selected_free_columns
        ],
    }

    artifacts = {
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
                "search_mode": attempt.search_mode,
                "legality_engine": attempt.legality_engine,
                "native_r_limit": attempt.native_r_limit,
                "forbidden_count": attempt.final_forbidden_count,
                "restart": attempt.restart_index,
                "added_free_columns": attempt.added_free_columns,
                "candidate_count": attempt.candidate_count,
                "sample_attempts": attempt.sample_attempt_count,
                "sampled_candidates": attempt.sampled_candidate_count,
                "scored_candidates": attempt.scored_candidate_count,
                "backtrack_events": attempt.backtrack_events,
                "backtracked_columns": attempt.backtracked_columns,
                "beam_width": attempt.beam_width,
                "beam_expanded_states": attempt.beam_expanded_states,
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
        "parity_check_matrix": json.dumps(list(parity_rows)),
        "generator_matrix": json.dumps(list(generator_rows)),
        "matrix_summary": json.dumps(matrix_summary, sort_keys=True),
    }
    if attempt.success:
        artifacts["successful_code_vectors"] = json.dumps(
            [_accepted_vector_to_dict(record) for record in attempt.accepted_vectors],
            sort_keys=True,
        )
        artifacts["successful_code_summary"] = json.dumps(
            _successful_code_summary(active_instance, attempt),
            sort_keys=True,
        )

    return EvaluationResult(
        metrics={
            "combined_score": combined_score,
            "success_rate": float(attempt.success),
            "avg_progress": progress,
            "constructed_columns": attempt.added_free_columns,
            "sample_attempts": attempt.sample_attempt_count,
            "sampled_candidates": attempt.sampled_candidate_count,
            "scored_candidates": attempt.scored_candidate_count,
            "backtrack_events": attempt.backtrack_events,
            "backtracked_columns": attempt.backtracked_columns,
            "beam_width": attempt.beam_width,
            "beam_expanded_states": attempt.beam_expanded_states,
            "native_r_limit": attempt.native_r_limit or 0,
            "forbidden_count": attempt.final_forbidden_count or 0,
            "target_columns": active_instance.k,
            "target_distance": active_instance.target_distance,
            "n": active_instance.n,
            "k": active_instance.k,
            "evaluation_time_seconds": elapsed_seconds,
        },
        artifacts=artifacts,
    )


def evaluate_program_path(program_path: str) -> EvaluationResult:
    """Convenience wrapper used by the OpenEvolve evaluator."""
    priority_fn = load_priority_function(program_path)
    return evaluate_priority_function(priority_fn, instance_from_env())
