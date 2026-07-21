"""Generator-matrix column-fill search for binary linear codes.

This is an experimental Python skeleton for constructing a systematic
generator matrix

    G = [I_k | p_1 p_2 ... p_r]

by choosing the parity columns p_j in F_2^k.  It treats every non-zero message
with weight below the target distance as a multicover constraint: each selected
parity column covers message m when dot(m, p_j) = 1.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

try:
    from openevolve.evaluation_result import EvaluationResult
except Exception:
    _EVAL_RESULT_PATH = (
        Path(__file__).resolve().parents[2] / "openevolve" / "evaluation_result.py"
    )
    _EVAL_RESULT_SPEC = importlib.util.spec_from_file_location(
        "openevolve_evaluation_result_generator_matrix_fallback",
        _EVAL_RESULT_PATH,
    )
    if _EVAL_RESULT_SPEC is None or _EVAL_RESULT_SPEC.loader is None:
        raise ImportError("Failed to load EvaluationResult fallback")
    _EVAL_RESULT_MODULE = importlib.util.module_from_spec(_EVAL_RESULT_SPEC)
    _EVAL_RESULT_SPEC.loader.exec_module(_EVAL_RESULT_MODULE)
    EvaluationResult = _EVAL_RESULT_MODULE.EvaluationResult


ColumnPriorityFn = Callable[..., float]


@dataclass(frozen=True)
class GeneratorColumnInstance:
    """Single binary [n,k,d] generator-column feasibility target."""

    n: int
    k: int
    d: int
    name: str = ""

    @property
    def r(self) -> int:
        return self.n - self.k


@dataclass(frozen=True)
class ColumnSearchConfig:
    """Controls for the deterministic Python column-fill skeleton."""

    restarts: int = 1
    shortlist_size: int = 1024
    random_pool_size: int = 0
    seed: int = 1
    max_k: int = 24
    max_low_messages: int = 8_000_000
    exact_verify_max_k: int = 24
    target_column_weight_fraction: float = 0.5
    deficit_weight: float = 1.0
    pressure_weight: float = 0.0
    critical_weight: float = 4.0
    row_need_weight: float = 0.0
    impossible_critical_penalty: float = 1000.0
    row_balance_weight: float = 0.2
    duplicate_penalty: float = 0.5
    correlation_penalty: float = 0.05


@dataclass(frozen=True)
class ColumnStepRecord:
    """Diagnostics for one accepted parity column."""

    step: int
    column: str
    column_weight: int
    covered_deficit_sum: int
    covered_critical_count: int
    uncovered_critical_count: int
    feasible_next_count: int
    remaining_unsatisfied: int
    remaining_deficit_sum: int
    min_row_weight_after: int
    max_row_weight_after: int
    score: float


@dataclass(frozen=True)
class ColumnSearchResult:
    """Output from one generator-column construction attempt."""

    instance: GeneratorColumnInstance
    config: ColumnSearchConfig
    success: bool
    columns: tuple[int, ...]
    column_bits: tuple[str, ...]
    row_weights: tuple[int, ...]
    unsatisfied_count: int
    remaining_deficit_sum: int
    min_margin: int
    d_actual: int | None
    step_records: tuple[ColumnStepRecord, ...]
    restart_index: int
    candidate_scoring_time: float
    exact_verification_time: float
    total_time: float


def make_instance(n: int, k: int, d: int, name: str | None = None) -> GeneratorColumnInstance:
    """Create and validate one generator-column instance."""
    if n <= 0 or k <= 0 or d <= 0:
        raise ValueError("n, k, and d must be positive")
    if k >= n:
        raise ValueError("Require 0 < k < n")
    if d > n:
        raise ValueError("Require d <= n")
    return GeneratorColumnInstance(n=n, k=k, d=d, name=name or f"[{n},{k},{d}]")


DEFAULT_INSTANCE = make_instance(20, 10, 5, name="default_[20,10,5]")


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        return default
    if minimum is not None:
        value = max(value, minimum)
    return value


def _env_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError:
        return default
    if not math.isfinite(value):
        return default
    return value


def instance_from_env(prefix: str = "GEN_MATRIX_CODE_") -> GeneratorColumnInstance:
    """Build one generator-matrix instance from environment variables."""
    n = _env_int(f"{prefix}N", DEFAULT_INSTANCE.n, minimum=1)
    k = _env_int(f"{prefix}K", DEFAULT_INSTANCE.k, minimum=1)
    d = _env_int(f"{prefix}D", DEFAULT_INSTANCE.d, minimum=1)
    return make_instance(n=n, k=k, d=d)


def config_from_env(prefix: str = "GEN_MATRIX_CODE_") -> ColumnSearchConfig:
    """Build search controls from environment variables."""
    return ColumnSearchConfig(
        restarts=_env_int(f"{prefix}RESTARTS", 1, minimum=1),
        shortlist_size=_env_int(f"{prefix}SHORTLIST_SIZE", 1024, minimum=1),
        random_pool_size=_env_int(f"{prefix}RANDOM_POOL_SIZE", 0, minimum=0),
        seed=_env_int(f"{prefix}RANDOM_SEED", 1),
        max_k=_env_int(f"{prefix}MAX_K", 24, minimum=1),
        max_low_messages=_env_int(f"{prefix}MAX_LOW_MESSAGES", 8_000_000, minimum=1),
        exact_verify_max_k=_env_int(f"{prefix}EXACT_VERIFY_MAX_K", 24, minimum=1),
        target_column_weight_fraction=_env_float(
            f"{prefix}TARGET_COLUMN_WEIGHT_FRACTION",
            0.5,
        ),
        deficit_weight=_env_float(f"{prefix}DEFICIT_WEIGHT", 1.0),
        pressure_weight=_env_float(f"{prefix}PRESSURE_WEIGHT", 0.0),
        critical_weight=_env_float(f"{prefix}CRITICAL_WEIGHT", 4.0),
        row_need_weight=_env_float(f"{prefix}ROW_NEED_WEIGHT", 0.0),
        impossible_critical_penalty=_env_float(
            f"{prefix}IMPOSSIBLE_CRITICAL_PENALTY",
            1000.0,
        ),
        row_balance_weight=_env_float(f"{prefix}ROW_BALANCE_WEIGHT", 0.2),
        duplicate_penalty=_env_float(f"{prefix}DUPLICATE_PENALTY", 0.5),
        correlation_penalty=_env_float(f"{prefix}CORRELATION_PENALTY", 0.05),
    )


def popcount(mask: int) -> int:
    """Return Hamming weight of a binary mask."""
    return int(mask.bit_count())


def format_mask(mask: int, width: int) -> str:
    """Fixed-width binary representation."""
    return format(mask, f"0{width}b")


def _mask_count(k: int) -> int:
    if k < 1:
        raise ValueError("k must be positive")
    return 1 << k


def _weights_for_all_masks(k: int) -> np.ndarray:
    masks = np.arange(_mask_count(k), dtype=np.uint32)
    return np.bitwise_count(masks).astype(np.int16)


@lru_cache(maxsize=16)
def low_message_masks(k: int, d: int) -> tuple[int, ...]:
    """All non-zero messages with weight below d."""
    weights = _weights_for_all_masks(k)
    low = np.flatnonzero((weights > 0) & (weights < d))
    return tuple(int(mask) for mask in low)


def initial_deficits(k: int, d: int) -> np.ndarray:
    """Dense deficit vector indexed by message mask; zero means already satisfied."""
    weights = _weights_for_all_masks(k)
    deficits = np.maximum(0, d - weights).astype(np.int16)
    deficits[0] = 0
    deficits[weights >= d] = 0
    return deficits


def _fwht(values: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard transform with sequency indexed by mask."""
    transformed = values.astype(np.int64, copy=True)
    size = transformed.shape[0]
    step = 1
    while step < size:
        blocks = transformed.reshape(-1, step * 2)
        left = blocks[:, :step].copy()
        right = blocks[:, step : step * 2]
        blocks[:, :step] = left + right
        blocks[:, step : step * 2] = left - right
        step *= 2
    return transformed


def _odd_dot_sums(values: np.ndarray) -> np.ndarray:
    """For every p, return sum(values[m]) over messages with dot(m,p)=1."""
    total = int(values.sum())
    spectrum = _fwht(values)
    return ((total - spectrum) // 2).astype(np.int64, copy=False)


def _row_need_sums(all_masks: np.ndarray, row_need: np.ndarray) -> np.ndarray:
    """For every candidate p, sum row_need[i] over set bits of p."""
    scores = np.zeros_like(all_masks, dtype=np.float64)
    for bit_index, need in enumerate(row_need):
        need_value = float(need)
        if need_value <= 0.0:
            continue
        scores[(all_masks & np.uint32(1 << bit_index)) != 0] += need_value
    return scores


def _dot_parity_mask(all_masks: np.ndarray, column_mask: int) -> np.ndarray:
    parity = np.bitwise_count(np.bitwise_and(all_masks, np.uint32(column_mask))) & 1
    return parity.astype(bool)


def _safe_priority(
    priority_fn: ColumnPriorityFn | None,
    column_mask: int,
    instance: GeneratorColumnInstance,
    step: int,
    column_weight: int,
    covered_deficit_sum: int,
    covered_critical_count: int,
    uncovered_critical_count: int,
    feasible_next_count: int,
    min_row_weight_after: int,
    max_row_weight_after: int,
    avg_pair_balance_after: float,
    default: float,
) -> float:
    if priority_fn is None:
        return default
    try:
        value = priority_fn(
            column_mask,
            instance.n,
            instance.k,
            instance.d,
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
    except TypeError:
        try:
            value = priority_fn(
                column_mask,
                instance.n,
                instance.k,
                instance.d,
                step,
                column_weight,
                covered_deficit_sum,
                covered_critical_count,
                uncovered_critical_count,
                min_row_weight_after,
                max_row_weight_after,
                avg_pair_balance_after,
            )
        except Exception:
            try:
                value = priority_fn(column_mask, instance.n, instance.k, instance.d)
            except Exception:
                return default
    except Exception:
        return default
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(score):
        return default
    return score


def _candidate_pool(
    base_scores: np.ndarray,
    rng: random.Random,
    config: ColumnSearchConfig,
) -> np.ndarray:
    valid_scores = base_scores.copy()
    valid_scores[0] = -np.inf
    finite_count = int(np.isfinite(valid_scores).sum())
    if finite_count <= 0:
        return np.array([], dtype=np.uint32)

    shortlist_size = min(config.shortlist_size, finite_count)
    if shortlist_size > 0 and shortlist_size < len(valid_scores):
        top_indices = np.argpartition(valid_scores, -shortlist_size)[-shortlist_size:]
    else:
        top_indices = np.flatnonzero(np.isfinite(valid_scores))

    if config.random_pool_size <= 0:
        return top_indices.astype(np.uint32, copy=False)

    max_mask = len(valid_scores) - 1
    random_indices = np.array(
        [rng.randint(1, max_mask) for _ in range(config.random_pool_size)],
        dtype=np.uint32,
    )
    return np.unique(np.concatenate([top_indices.astype(np.uint32), random_indices]))


def _hard_feasible_pool(
    pool: np.ndarray,
    covered_critical_counts: np.ndarray,
    current_critical_count: int,
    base_scores: np.ndarray,
) -> np.ndarray:
    """Keep only candidates that cover every message that must be covered now."""
    if current_critical_count <= 0:
        return pool
    feasible_pool = pool[covered_critical_counts[pool] == current_critical_count]
    if len(feasible_pool) > 0:
        return feasible_pool
    all_feasible = np.flatnonzero(
        (covered_critical_counts == current_critical_count)
        & np.isfinite(base_scores)
        & (np.arange(len(base_scores), dtype=np.uint32) != 0)
    )
    return all_feasible.astype(np.uint32, copy=False)


def _sample_mask_array(masks: np.ndarray, limit: int = 512) -> np.ndarray:
    """Deterministic compact sample used for forward-feasibility estimates."""
    if len(masks) <= limit:
        return masks.astype(np.uint32, copy=False)
    positions = np.linspace(0, len(masks) - 1, num=limit, dtype=np.int64)
    return masks[positions].astype(np.uint32, copy=False)


def _add_augmented_constraint(basis: list[int], row_mask: int, k: int) -> bool:
    """Add one equation dot(row_mask, x)=1 to an augmented GF(2) basis."""
    row = int(row_mask) | (1 << k)
    variable_mask = (1 << k) - 1
    for bit_index in range(k - 1, -1, -1):
        if ((row >> bit_index) & 1) == 0:
            continue
        if basis[bit_index]:
            row ^= basis[bit_index]
        else:
            basis[bit_index] = row
            return True
    return (row & ~variable_mask) == 0


def _basis_from_constraints(k: int, masks: np.ndarray) -> tuple[list[int], bool]:
    basis = [0] * k
    for mask in masks:
        if not _add_augmented_constraint(basis, int(mask), k):
            return basis, False
    return basis, True


def _solution_count_from_basis(k: int, basis: Sequence[int], consistent: bool) -> int:
    if not consistent:
        return 0
    rank = sum(1 for row in basis if row)
    if rank == 0:
        return (1 << k) - 1
    return 1 << max(0, k - rank)


def _estimate_feasible_next_count(
    column_mask: int,
    k: int,
    remaining_after: int,
    deficits: np.ndarray,
    current_critical_sample: np.ndarray,
    near_critical_sample: np.ndarray,
) -> int:
    """Estimate how many next columns satisfy the next hard constraints."""
    if remaining_after <= 0:
        return 0 if bool((deficits > 0).any()) else 1
    basis, consistent = _basis_from_constraints(k, current_critical_sample)
    if not consistent:
        return 0
    if len(near_critical_sample) > 0:
        covered_near = _dot_parity_mask(near_critical_sample, column_mask)
        for mask in near_critical_sample[~covered_near]:
            if not _add_augmented_constraint(basis, int(mask), k):
                return 0
    return _solution_count_from_basis(k, basis, consistent=True)


def _avg_pair_balance(row_weights_after: np.ndarray) -> float:
    if row_weights_after.size <= 1:
        return 0.0
    centered = row_weights_after.astype(np.float64) - float(row_weights_after.mean())
    return float(np.mean(np.abs(centered)))


def _column_row_bits(column_mask: int, k: int) -> np.ndarray:
    row_indices = np.arange(k, dtype=np.uint32)
    return (((np.uint32(column_mask) >> row_indices) & np.uint32(1))).astype(np.int16)


def _score_shortlist_candidate(
    column_mask: int,
    instance: GeneratorColumnInstance,
    config: ColumnSearchConfig,
    priority_fn: ColumnPriorityFn | None,
    step: int,
    row_weights: np.ndarray,
    selected_counts: dict[int, int],
    selected_columns: Sequence[int],
    covered_deficit_sums: np.ndarray,
    covered_pressure_sums: np.ndarray,
    covered_critical_counts: np.ndarray,
    current_critical_count: int,
    feasible_next_count: int,
    row_need_scores: np.ndarray,
    all_masks: np.ndarray,
) -> tuple[float, ColumnStepRecord]:
    column_weight = popcount(column_mask)
    row_bits = _column_row_bits(column_mask, instance.k)
    row_weights_after = row_weights + row_bits
    min_row_after = int(row_weights_after.min())
    max_row_after = int(row_weights_after.max())
    avg_pair_balance_after = _avg_pair_balance(row_weights_after)
    covered_deficit_sum = int(covered_deficit_sums[column_mask])
    covered_pressure_sum = int(covered_pressure_sums[column_mask])
    covered_critical_count = int(covered_critical_counts[column_mask])
    uncovered_critical_count = current_critical_count - covered_critical_count
    target_weight = instance.k * config.target_column_weight_fraction
    future_space = feasible_next_count / (feasible_next_count + 1024.0)

    duplicate_cost = selected_counts.get(column_mask, 0)
    correlation_cost = 0.0
    if selected_columns:
        sample = selected_columns[-min(16, len(selected_columns)) :]
        correlation_cost = sum(
            abs(popcount(column_mask ^ previous) - instance.k / 2.0)
            for previous in sample
        ) / len(sample)

    default_score = (
        config.deficit_weight * covered_deficit_sum
        + config.pressure_weight * covered_pressure_sum
        + config.critical_weight * covered_critical_count
        + 2.0 * future_space
        + config.row_need_weight * float(row_need_scores[column_mask])
        - config.impossible_critical_penalty * uncovered_critical_count
        - abs(column_weight - target_weight)
        - config.row_balance_weight * (max_row_after - min_row_after)
        - config.duplicate_penalty * duplicate_cost
        - config.correlation_penalty * correlation_cost
    )
    score = _safe_priority(
        priority_fn,
        column_mask,
        instance,
        step,
        column_weight,
        covered_deficit_sum,
        covered_critical_count,
        uncovered_critical_count,
        feasible_next_count,
        min_row_after,
        max_row_after,
        avg_pair_balance_after,
        default_score,
    )
    record = ColumnStepRecord(
        step=step,
        column=format_mask(column_mask, instance.k),
        column_weight=column_weight,
        covered_deficit_sum=covered_deficit_sum,
        covered_critical_count=covered_critical_count,
        uncovered_critical_count=uncovered_critical_count,
        feasible_next_count=feasible_next_count,
        remaining_unsatisfied=0,
        remaining_deficit_sum=0,
        min_row_weight_after=min_row_after,
        max_row_weight_after=max_row_after,
        score=score,
    )
    return score, record


def search_generator_columns(
    instance: GeneratorColumnInstance,
    config: ColumnSearchConfig | None = None,
    priority_fn: ColumnPriorityFn | None = None,
) -> ColumnSearchResult:
    """Greedily fill parity columns in G=[I_k|P] using deficit multicover scores."""
    config = config or ColumnSearchConfig()
    if instance.r <= 0:
        raise ValueError("Require positive redundancy r=n-k")
    if instance.k > config.max_k:
        raise ValueError(f"k={instance.k} exceeds configured max_k={config.max_k}")

    low_message_count = len(low_message_masks(instance.k, instance.d))
    if low_message_count > config.max_low_messages:
        raise ValueError(
            f"{low_message_count} low-weight messages exceeds max_low_messages="
            f"{config.max_low_messages}"
        )

    best_result: ColumnSearchResult | None = None
    for restart_index in range(config.restarts):
        candidate = _search_one_restart(instance, config, priority_fn, restart_index)
        if best_result is None:
            best_result = candidate
        candidate_key = (
            candidate.success,
            -candidate.remaining_deficit_sum,
            -candidate.unsatisfied_count,
        )
        best_key = (
            best_result.success,
            -best_result.remaining_deficit_sum,
            -best_result.unsatisfied_count,
        )
        if candidate_key > best_key:
            best_result = candidate
        if candidate.success:
            break

    assert best_result is not None
    return best_result


def _search_one_restart(
    instance: GeneratorColumnInstance,
    config: ColumnSearchConfig,
    priority_fn: ColumnPriorityFn | None,
    restart_index: int,
) -> ColumnSearchResult:
    start_time = time.perf_counter()
    rng = random.Random(config.seed + 1009 * restart_index)
    deficits = initial_deficits(instance.k, instance.d)
    all_masks = np.arange(_mask_count(instance.k), dtype=np.uint32)
    mask_weights = _weights_for_all_masks(instance.k)
    column_weights = mask_weights.astype(np.float64)
    target_weight = instance.k * config.target_column_weight_fraction
    selected_columns: list[int] = []
    selected_counts: dict[int, int] = {}
    row_weights = np.zeros(instance.k, dtype=np.int16)
    step_records: list[ColumnStepRecord] = []
    scoring_time = 0.0

    for step in range(instance.r):
        remaining_after = instance.r - step - 1
        positive = deficits > 0
        if not bool(positive.any()):
            break
        if int(deficits.max(initial=0)) > remaining_after + 1:
            break

        scoring_start = time.perf_counter()
        covered_deficit_sums = _odd_dot_sums(deficits)
        if config.pressure_weight:
            pressure = (deficits.astype(np.int32) * deficits.astype(np.int32)).astype(np.int32)
            covered_pressure_sums = _odd_dot_sums(pressure)
        else:
            covered_pressure_sums = np.zeros_like(covered_deficit_sums)
        critical = (deficits > remaining_after).astype(np.int16)
        current_critical_count = int(critical.sum())
        covered_critical_counts = _odd_dot_sums(critical)
        current_critical_sample = _sample_mask_array(all_masks[critical.astype(bool)])
        near_critical_sample = _sample_mask_array(all_masks[deficits == remaining_after])
        if config.row_need_weight:
            row_need = np.maximum(0, instance.d - 1 - row_weights).astype(np.int16)
            row_need_scores = _row_need_sums(all_masks, row_need)
        else:
            row_need_scores = np.zeros_like(all_masks, dtype=np.float64)
        base_scores = (
            config.deficit_weight * covered_deficit_sums.astype(np.float64)
            + config.pressure_weight * covered_pressure_sums.astype(np.float64)
            + config.critical_weight * covered_critical_counts.astype(np.float64)
            + config.row_need_weight * row_need_scores
            - config.impossible_critical_penalty
            * (current_critical_count - covered_critical_counts).astype(np.float64)
            - np.abs(column_weights - target_weight)
        )
        for previous, count in selected_counts.items():
            base_scores[previous] -= config.duplicate_penalty * count
        pool = _candidate_pool(base_scores, rng, config)
        pool = _hard_feasible_pool(
            pool,
            covered_critical_counts,
            current_critical_count,
            base_scores,
        )
        scoring_time += time.perf_counter() - scoring_start

        best_score = -math.inf
        best_tie_break = -1
        best_column = 0
        best_record: ColumnStepRecord | None = None
        for raw_mask in pool:
            column_mask = int(raw_mask)
            feasible_next_count = _estimate_feasible_next_count(
                column_mask,
                instance.k,
                remaining_after,
                deficits,
                current_critical_sample,
                near_critical_sample,
            )
            score, record = _score_shortlist_candidate(
                column_mask,
                instance,
                config,
                priority_fn,
                step,
                row_weights,
                selected_counts,
                selected_columns,
                covered_deficit_sums,
                covered_pressure_sums,
                covered_critical_counts,
                current_critical_count,
                feasible_next_count,
                row_need_scores,
                all_masks,
            )
            tie_break = ((column_mask * 1103515245) + restart_index * 2654435761) & 0xFFFFFFFF
            if score > best_score or (score == best_score and tie_break > best_tie_break):
                best_score = score
                best_tie_break = tie_break
                best_column = column_mask
                best_record = record

        if best_column == 0 or best_record is None:
            break

        odd = _dot_parity_mask(all_masks, best_column)
        covered_positive = odd & (deficits > 0)
        deficits[covered_positive] -= 1
        row_bits = _column_row_bits(best_column, instance.k)
        row_weights += row_bits
        selected_columns.append(best_column)
        selected_counts[best_column] = selected_counts.get(best_column, 0) + 1

        unsatisfied = int((deficits > 0).sum())
        remaining_deficit_sum = int(deficits[deficits > 0].sum())
        step_records.append(
            ColumnStepRecord(
                **{
                    **asdict(best_record),
                    "remaining_unsatisfied": unsatisfied,
                    "remaining_deficit_sum": remaining_deficit_sum,
                }
            )
        )

    unsatisfied_count = int((deficits > 0).sum())
    remaining_deficit_sum = int(deficits[deficits > 0].sum())
    min_margin = _minimum_margin_from_deficits(deficits)
    verify_start = time.perf_counter()
    d_actual = None
    success = unsatisfied_count == 0 and len(selected_columns) == instance.r
    if instance.k <= config.exact_verify_max_k:
        d_actual = exact_minimum_distance(instance.k, selected_columns)
        success = success and d_actual >= instance.d
    exact_time = time.perf_counter() - verify_start
    total_time = time.perf_counter() - start_time

    return ColumnSearchResult(
        instance=instance,
        config=config,
        success=success,
        columns=tuple(selected_columns),
        column_bits=tuple(format_mask(column, instance.k) for column in selected_columns),
        row_weights=tuple(int(value) for value in row_weights),
        unsatisfied_count=unsatisfied_count,
        remaining_deficit_sum=remaining_deficit_sum,
        min_margin=min_margin,
        d_actual=d_actual,
        step_records=tuple(step_records),
        restart_index=restart_index,
        candidate_scoring_time=scoring_time,
        exact_verification_time=exact_time,
        total_time=total_time,
    )


def _minimum_margin_from_deficits(deficits: np.ndarray) -> int:
    max_deficit = int(deficits.max(initial=0))
    return -max_deficit


def exact_minimum_distance(k: int, parity_columns: Sequence[int]) -> int:
    """Exhaustively verify min_{m != 0} wt(mG) for G=[I_k|P]."""
    all_messages = np.arange(_mask_count(k), dtype=np.uint32)
    weights = _weights_for_all_masks(k).astype(np.int16)
    parity_hits = np.zeros_like(weights)
    for column in parity_columns:
        parity_hits += _dot_parity_mask(all_messages, int(column)).astype(np.int16)
    codeword_weights = weights + parity_hits
    return int(codeword_weights[1:].min(initial=k + len(parity_columns) + 1))


def generator_matrix_rows(k: int, parity_columns: Sequence[int]) -> tuple[str, ...]:
    """Return G=[I_k|P] as k row strings."""
    rows = []
    for row_index in range(k):
        identity = ["1" if col_index == row_index else "0" for col_index in range(k)]
        parity = [
            "1" if (int(column) >> row_index) & 1 else "0"
            for column in parity_columns
        ]
        rows.append("".join(identity + parity))
    return tuple(rows)


def parity_check_matrix_rows(k: int, parity_columns: Sequence[int]) -> tuple[str, ...]:
    """Return H=[P^T|I_r] corresponding to G=[I_k|P]."""
    r = len(parity_columns)
    rows = []
    for parity_index, column in enumerate(parity_columns):
        left = [
            "1" if (int(column) >> row_index) & 1 else "0"
            for row_index in range(k)
        ]
        identity = ["1" if col_index == parity_index else "0" for col_index in range(r)]
        rows.append("".join(left + identity))
    return tuple(rows)


def result_to_json_dict(
    result: ColumnSearchResult,
    include_steps: bool = True,
) -> dict[str, object]:
    """Convert result dataclasses into JSON-friendly dictionaries."""
    data = asdict(result)
    if not include_steps:
        data["step_records"] = []
    data["generator_matrix"] = list(generator_matrix_rows(result.instance.k, result.columns))
    data["parity_check_matrix"] = list(parity_check_matrix_rows(result.instance.k, result.columns))
    return data


def _initial_deficit_sum(k: int, d: int) -> int:
    deficits = initial_deficits(k, d)
    return int(deficits[deficits > 0].sum())


def metrics_from_result(result: ColumnSearchResult) -> dict[str, float]:
    """Build OpenEvolve metrics from a generator-column search result."""
    initial_sum = max(1, _initial_deficit_sum(result.instance.k, result.instance.d))
    coverage_progress = 1.0 - (result.remaining_deficit_sum / initial_sum)
    coverage_progress = max(0.0, min(1.0, coverage_progress))
    column_progress = min(len(result.columns) / max(1, result.instance.r), 1.0)
    if result.success:
        combined_score = 1.0
    else:
        same_column_tiebreak = coverage_progress
        column_bonus = same_column_tiebreak / max(2.0, float(result.instance.r + 1))
        combined_score = min(0.999, column_progress + column_bonus)
    return {
        "combined_score": float(combined_score),
        "success_rate": 1.0 if result.success else 0.0,
        "constructed_columns": float(len(result.columns)),
        "target_columns": float(result.instance.r),
        "column_progress": float(column_progress),
        "coverage_progress": float(coverage_progress),
    }


def artifacts_from_result(result: ColumnSearchResult) -> dict[str, str]:
    """Build JSON artifacts for OpenEvolve and manual inspection."""
    generator_rows = generator_matrix_rows(result.instance.k, result.columns)
    parity_rows = parity_check_matrix_rows(result.instance.k, result.columns)
    matrix_summary = {
        "matrix_form": "G=[I_k|P]",
        "dual_parity_check_form": "H=[P^T|I_r]",
        "complete": result.success,
        "n": result.instance.n,
        "k": result.instance.k,
        "r": result.instance.r,
        "d_target": result.instance.d,
        "d_actual": result.d_actual,
        "selected_parity_columns": list(result.column_bits),
        "unsatisfied_messages": result.unsatisfied_count,
        "remaining_deficit_sum": result.remaining_deficit_sum,
        "min_margin": result.min_margin,
    }
    return {
        "search_result": json.dumps(result_to_json_dict(result), sort_keys=True),
        "matrix_summary": json.dumps(matrix_summary, sort_keys=True),
        "generator_matrix": json.dumps(list(generator_rows)),
        "parity_check_matrix": json.dumps(list(parity_rows)),
    }


def evaluate_priority_function(
    priority_fn: ColumnPriorityFn | None,
    instance: GeneratorColumnInstance | None = None,
    config: ColumnSearchConfig | None = None,
) -> EvaluationResult:
    """Evaluate a priority function with the fixed generator-column skeleton."""
    instance = instance or instance_from_env()
    config = config or config_from_env()
    try:
        result = search_generator_columns(instance, config, priority_fn)
    except Exception as exc:
        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "success_rate": 0.0,
                "constructed_columns": 0.0,
                "target_columns": float(instance.r),
                "column_progress": 0.0,
                "coverage_progress": 0.0,
            },
            artifacts={
                "search_result": json.dumps(
                    {
                        "success": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "instance": asdict(instance),
                        "config": asdict(config),
                    },
                    sort_keys=True,
                )
            },
        )
    return EvaluationResult(
        metrics=metrics_from_result(result),
        artifacts=artifacts_from_result(result),
    )


def _benchmark_instances() -> tuple[GeneratorColumnInstance, ...]:
    return (
        make_instance(20, 10, 5),
        make_instance(31, 21, 5),
        make_instance(50, 20, 13),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experimental G=[I_k|P] parity-column deficit multicover search."
    )
    parser.add_argument("--N", type=int, dest="n", help="Code length n.")
    parser.add_argument("--K", type=int, dest="k", help="Code dimension k.")
    parser.add_argument("--D", type=int, dest="d", help="Target minimum distance d.")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the three suggested targets.",
    )
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--shortlist-size", type=int, default=1024)
    parser.add_argument("--random-pool-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pressure-weight", type=float, default=0.0)
    parser.add_argument("--row-need-weight", type=float, default=0.0)
    parser.add_argument("--row-balance-weight", type=float, default=0.2)
    parser.add_argument("--duplicate-penalty", type=float, default=0.5)
    parser.add_argument("--json", action="store_true", help="Print full JSON output.")
    parser.add_argument("--no-steps", action="store_true", help="Omit per-step records from JSON.")
    args = parser.parse_args()

    if args.benchmark:
        instances = _benchmark_instances()
    else:
        if args.n is None or args.k is None or args.d is None:
            parser.error("provide --N, --K, --D or use --benchmark")
        instances = (make_instance(args.n, args.k, args.d),)

    config = ColumnSearchConfig(
        restarts=args.restarts,
        shortlist_size=args.shortlist_size,
        random_pool_size=args.random_pool_size,
        seed=args.seed,
        pressure_weight=args.pressure_weight,
        row_need_weight=args.row_need_weight,
        row_balance_weight=args.row_balance_weight,
        duplicate_penalty=args.duplicate_penalty,
    )
    results = [search_generator_columns(instance, config) for instance in instances]
    if args.json:
        print(
            json.dumps(
                [result_to_json_dict(r, include_steps=not args.no_steps) for r in results],
                indent=2,
            )
        )
        return

    for result in results:
        print(
            json.dumps(
                {
                    "instance": result.instance.name,
                    "success": result.success,
                    "columns": len(result.columns),
                    "target_columns": result.instance.r,
                    "d_actual": result.d_actual,
                    "unsatisfied_count": result.unsatisfied_count,
                    "remaining_deficit_sum": result.remaining_deficit_sum,
                    "min_margin": result.min_margin,
                    "candidate_scoring_time": round(result.candidate_scoring_time, 6),
                    "exact_verification_time": round(result.exact_verification_time, 6),
                    "total_time": round(result.total_time, 6),
                    "restart_index": result.restart_index,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
