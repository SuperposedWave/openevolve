#!/usr/bin/env python3
"""Incremental row-legality search for systematic G=[I_k|P].

This is an experimental counterpart to the H-side forbidden-xor construction.
Rows of P are added one at a time.  If an existing subset of selected rows has
parity xor ``a`` and size ``s``, a new row ``x`` is legal only when:

    (s + 1) + wt(a ^ x) >= d

for every relevant subset.  Equivalently, ``x`` must avoid Hamming balls around
all previous subset xors.  Maintaining these subset-xor layers gives the G-side
search the same kind of exact "red/green light" feedback that the H-side search
gets from forbidden xor sets.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import random
import time
from typing import Callable, Sequence

from fast_p_evaluator import evaluate_parity_rows_gray


RowPriorityFn = Callable[..., float]


@dataclass(frozen=True)
class RowSearchConfig:
    """Controls for one G-row incremental search run."""

    n: int
    k: int
    d: int
    restarts: int = 8
    max_attempts_per_step: int = 20000
    legal_pool_target: int = 1
    seed: int = 1
    min_row_weight: int | None = None
    prefer_weight: int | None = None
    near_margin_radius: int = 1
    repair_events: int = 4
    repair_drop_count: int = 2
    repair_strategy: str = "recent"
    repair_tabu_tenure: int = 16

    @property
    def r(self) -> int:
        return self.n - self.k


@dataclass(frozen=True)
class RowStepRecord:
    """Diagnostics for one selected row."""

    step: int
    row: str
    row_weight: int
    attempts: int
    legal_seen: int
    checked_constraints: int
    min_margin: int
    near_margin_count: int
    layer_counts: tuple[int, ...]
    score: float


@dataclass(frozen=True)
class RowSearchResult:
    """Best result from the row-legality search."""

    success: bool
    n: int
    k: int
    d: int
    r: int
    rows: tuple[int, ...]
    row_bits: tuple[str, ...]
    restart: int
    step_records: tuple[RowStepRecord, ...]
    exact_minimum_distance: int | None
    exact_violation_count: int | None
    exact_minimum_margin: int | None
    evaluated_messages: int | None
    total_attempts: int
    repair_events: int
    dropped_rows: int
    elapsed_seconds: float


def _format_mask(mask: int, width: int) -> str:
    return format(mask, f"0{width}b")


def _default_min_row_weight(config: RowSearchConfig) -> int:
    return config.min_row_weight if config.min_row_weight is not None else config.d - 1


def _default_prefer_weight(config: RowSearchConfig) -> int:
    return config.prefer_weight if config.prefer_weight is not None else config.r // 2


def _random_row_mask(rng: random.Random, r: int, min_weight: int) -> int:
    """Sample a random r-bit row, retrying until it meets the weight floor."""
    while True:
        mask = rng.getrandbits(r)
        if mask and mask.bit_count() >= min_weight:
            return mask


def _validate_config(config: RowSearchConfig) -> None:
    if config.n <= 0 or config.k <= 0 or config.d <= 0:
        raise ValueError("n, k, and d must be positive")
    if config.k >= config.n:
        raise ValueError("require n > k")
    if config.r >= 63:
        raise ValueError("this prototype stores rows in a Python int and expects r < 63")
    if config.d > config.n:
        raise ValueError("d must be at most n")
    if config.max_attempts_per_step <= 0:
        raise ValueError("max_attempts_per_step must be positive")
    if config.legal_pool_target <= 0:
        raise ValueError("legal_pool_target must be positive")
    if config.repair_events < 0:
        raise ValueError("repair_events must be non-negative")
    if config.repair_drop_count < 0:
        raise ValueError("repair_drop_count must be non-negative")
    if config.repair_tabu_tenure < 0:
        raise ValueError("repair_tabu_tenure must be non-negative")
    if config.repair_strategy not in {"recent", "random", "tight"}:
        raise ValueError("repair_strategy must be recent, random, or tight")


def _can_add_with_diagnostics(
    layers: list[set[int]],
    selected_count: int,
    row: int,
    d: int,
    near_margin_radius: int,
) -> tuple[bool, int, int, int]:
    """Return legality plus min margin, near-margin count, and checks performed."""
    min_margin = math.inf
    near_margin_count = 0
    checked_constraints = 0
    max_existing_subset = min(selected_count, d - 2, len(layers) - 1)
    for subset_size in range(max_existing_subset + 1):
        threshold = d - (subset_size + 1)
        if threshold <= 0:
            continue
        for xor_value in layers[subset_size]:
            checked_constraints += 1
            margin = (xor_value ^ row).bit_count() - threshold
            if margin < 0:
                return False, margin, near_margin_count, checked_constraints
            if margin < min_margin:
                min_margin = margin
            if margin <= near_margin_radius:
                near_margin_count += 1
    if min_margin == math.inf:
        min_margin = 0
    return True, int(min_margin), near_margin_count, checked_constraints


def _add_row_to_layers(layers: list[set[int]], selected_count: int, row: int, d: int) -> None:
    """Update subset-xor layers after accepting a legal row."""
    max_source_subset = min(selected_count, d - 2, len(layers) - 2)
    for subset_size in range(max_source_subset, -1, -1):
        target = layers[subset_size + 1]
        for xor_value in layers[subset_size]:
            target.add(xor_value ^ row)


def _new_layers(config: RowSearchConfig) -> list[set[int]]:
    max_layer = max(0, config.d - 1)
    layers: list[set[int]] = [set() for _ in range(max_layer + 1)]
    layers[0].add(0)
    return layers


def _rebuild_layers(config: RowSearchConfig, selected: Sequence[int]) -> list[set[int]]:
    layers = _new_layers(config)
    for index, row in enumerate(selected):
        _add_row_to_layers(layers, index, row, config.d)
    return layers


def _candidate_score(
    *,
    row: int,
    row_weight: int,
    prefer_weight: int,
    min_margin: int,
    near_margin_count: int,
    checked_constraints: int,
) -> float:
    """Score a legal row.  Wider margins dominate softer balance preferences."""
    near_rate = near_margin_count / max(1, checked_constraints)
    return (
        1000.0 * float(min_margin)
        - 100.0 * near_rate
        - abs(row_weight - prefer_weight)
        + (row * 0.000000001)
    )


def _safe_priority(
    priority_fn: RowPriorityFn | None,
    *,
    row: int,
    config: RowSearchConfig,
    step: int,
    restart: int,
    row_weight: int,
    attempts: int,
    legal_seen: int,
    selected_count: int,
    checked_constraints: int,
    min_margin: int,
    near_margin_count: int,
    layer_counts: tuple[int, ...],
    default_score: float,
) -> float:
    """Call an evolved row priority function, falling back to the fixed score."""
    if priority_fn is None:
        return default_score
    try:
        value = priority_fn(
            row,
            config.n,
            config.k,
            config.d,
            step,
            row_weight,
            min_margin,
            near_margin_count,
            checked_constraints,
            legal_seen,
            attempts,
            selected_count,
            sum(layer_counts),
            max(layer_counts) if layer_counts else 0,
            restart,
        )
    except TypeError:
        try:
            value = priority_fn(row, config.n, config.k, config.d)
        except Exception:
            return default_score
    except Exception:
        return default_score
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default_score
    if not math.isfinite(score):
        return default_score
    return score


def _repair_drop_indices(
    selected: Sequence[int],
    records: Sequence[RowStepRecord],
    rng: random.Random,
    config: RowSearchConfig,
) -> list[int]:
    """Choose selected-row indices to drop when a restart gets stuck."""
    drop_count = min(config.repair_drop_count, len(selected))
    if drop_count <= 0:
        return []
    if config.repair_strategy == "random":
        return sorted(rng.sample(range(len(selected)), drop_count), reverse=True)
    if config.repair_strategy == "tight":
        def key(index: int) -> tuple[float, int, int]:
            if index >= len(records):
                return (0.0, 0, index)
            record = records[index]
            near_rate = record.near_margin_count / max(1, record.checked_constraints)
            return (near_rate, -record.min_margin, index)

        ranked = sorted(range(len(selected)), key=key, reverse=True)
        return sorted(ranked[:drop_count], reverse=True)
    start = max(0, len(selected) - drop_count)
    return list(range(len(selected) - 1, start - 1, -1))


def _remember_tabu(tabu: list[int], dropped: Sequence[int], tenure: int) -> None:
    if tenure <= 0:
        return
    tabu.extend(dropped)
    if len(tabu) > tenure:
        del tabu[: len(tabu) - tenure]


def search_rows_once(
    config: RowSearchConfig,
    restart: int,
    priority_fn: RowPriorityFn | None = None,
) -> RowSearchResult:
    """Run one randomized incremental legality search."""
    _validate_config(config)
    started_at = time.perf_counter()
    rng = random.Random(config.seed + 1009 * restart)
    min_row_weight = _default_min_row_weight(config)
    prefer_weight = _default_prefer_weight(config)
    layers = _new_layers(config)
    selected: list[int] = []
    selected_set: set[int] = set()
    records: list[RowStepRecord] = []
    tabu_rows: list[int] = []
    total_attempts = 0
    repair_events = 0
    dropped_rows = 0

    while len(selected) < config.k:
        step = len(selected)
        best_row = 0
        best_score = -math.inf
        best_min_margin = 0
        best_near_count = 0
        best_checked = 0
        legal_seen = 0
        attempts = 0

        while attempts < config.max_attempts_per_step and legal_seen < config.legal_pool_target:
            attempts += 1
            row = _random_row_mask(rng, config.r, min_row_weight)
            if row in selected_set or row in tabu_rows:
                continue
            legal, min_margin, near_count, checked = _can_add_with_diagnostics(
                layers,
                len(selected),
                row,
                config.d,
                config.near_margin_radius,
            )
            if not legal:
                continue
            legal_seen += 1
            row_weight = row.bit_count()
            default_score = _candidate_score(
                row=row,
                row_weight=row_weight,
                prefer_weight=prefer_weight,
                min_margin=min_margin,
                near_margin_count=near_count,
                checked_constraints=checked,
            )
            layer_counts = tuple(len(layer) for layer in layers)
            score = _safe_priority(
                priority_fn,
                row=row,
                config=config,
                step=step,
                restart=restart,
                row_weight=row_weight,
                attempts=attempts,
                legal_seen=legal_seen,
                selected_count=len(selected),
                checked_constraints=checked,
                min_margin=min_margin,
                near_margin_count=near_count,
                layer_counts=layer_counts,
                default_score=default_score,
            )
            if score > best_score:
                best_row = row
                best_score = score
                best_min_margin = min_margin
                best_near_count = near_count
                best_checked = checked

        total_attempts += attempts
        if best_row == 0:
            if (
                repair_events >= config.repair_events
                or config.repair_drop_count <= 0
                or not selected
            ):
                break
            drop_indices = _repair_drop_indices(selected, records, rng, config)
            if not drop_indices:
                break
            dropped = []
            for index in drop_indices:
                if index < 0 or index >= len(selected):
                    continue
                dropped.append(selected.pop(index))
                if index < len(records):
                    records.pop(index)
            if not dropped:
                break
            selected_set = set(selected)
            _remember_tabu(tabu_rows, dropped, config.repair_tabu_tenure)
            layers = _rebuild_layers(config, selected)
            repair_events += 1
            dropped_rows += len(dropped)
            continue

        _add_row_to_layers(layers, len(selected), best_row, config.d)
        selected.append(best_row)
        selected_set.add(best_row)
        records.append(
            RowStepRecord(
                step=step,
                row=_format_mask(best_row, config.r),
                row_weight=best_row.bit_count(),
                attempts=attempts,
                legal_seen=legal_seen,
                checked_constraints=best_checked,
                min_margin=best_min_margin,
                near_margin_count=best_near_count,
                layer_counts=tuple(len(layer) for layer in layers),
                score=best_score,
            )
        )

    exact_distance = None
    exact_violations = None
    exact_margin = None
    evaluated_messages = None
    if selected:
        exact = evaluate_parity_rows_gray(
            selected,
            r=config.r,
            target_distance=config.d,
        )
        exact_distance = exact.minimum_distance
        exact_violations = exact.violation_count
        exact_margin = exact.minimum_margin
        evaluated_messages = exact.evaluated_messages

    success = len(selected) == config.k and exact_distance is not None and exact_distance >= config.d
    return RowSearchResult(
        success=success,
        n=config.n,
        k=config.k,
        d=config.d,
        r=config.r,
        rows=tuple(selected),
        row_bits=tuple(_format_mask(row, config.r) for row in selected),
        restart=restart,
        step_records=tuple(records),
        exact_minimum_distance=exact_distance,
        exact_violation_count=exact_violations,
        exact_minimum_margin=exact_margin,
        evaluated_messages=evaluated_messages,
        total_attempts=total_attempts,
        repair_events=repair_events,
        dropped_rows=dropped_rows,
        elapsed_seconds=time.perf_counter() - started_at,
    )


def _result_key(result: RowSearchResult) -> tuple[object, ...]:
    return (
        result.success,
        len(result.rows),
        result.exact_minimum_distance if result.exact_minimum_distance is not None else -1,
        -(result.exact_violation_count if result.exact_violation_count is not None else 10**18),
        result.exact_minimum_margin if result.exact_minimum_margin is not None else -10**18,
        -result.repair_events,
        -result.dropped_rows,
    )


def search_rows(config: RowSearchConfig) -> RowSearchResult:
    """Run multiple restarts and return the best row-legality result."""
    return search_rows_with_priority(config, priority_fn=None)


def search_rows_with_priority(
    config: RowSearchConfig,
    priority_fn: RowPriorityFn | None = None,
) -> RowSearchResult:
    """Run multiple restarts with an optional evolved legal-row ranker."""
    best: RowSearchResult | None = None
    for restart in range(config.restarts):
        result = search_rows_once(config, restart, priority_fn)
        if best is None or _result_key(result) > _result_key(best):
            best = result
        if result.success:
            break
    assert best is not None
    return best


def metrics_from_result(result: RowSearchResult) -> dict[str, float]:
    """Build OpenEvolve metrics for row-legality search."""
    row_progress = len(result.rows) / max(1, result.k)
    if result.success:
        combined_score = 1.0
    else:
        distance_bonus = 0.0
        if result.exact_minimum_distance is not None:
            distance_bonus = min(result.exact_minimum_distance / max(1, result.d), 1.0)
        combined_score = min(0.999, row_progress + 0.04 * distance_bonus / max(1, result.k))
    return {
        "combined_score": float(combined_score),
        "success_rate": 1.0 if result.success else 0.0,
        "constructed_rows": float(len(result.rows)),
        "target_rows": float(result.k),
        "row_progress": float(row_progress),
        "exact_minimum_distance": float(result.exact_minimum_distance or 0),
        "exact_violation_count": float(result.exact_violation_count or 0),
        "total_attempts": float(result.total_attempts),
        "repair_events": float(result.repair_events),
        "dropped_rows": float(result.dropped_rows),
        "evaluation_time_seconds": float(result.elapsed_seconds),
    }


def result_to_json_dict(result: RowSearchResult, include_steps: bool = True) -> dict[str, object]:
    """Convert a result to a compact JSON-friendly dict."""
    payload = asdict(result)
    payload["rows"] = list(result.row_bits)
    if not include_steps:
        payload["step_records"] = []
    payload["matrix_summary"] = {
        "form": "G=[I_k|P], H=[P^T|I_r]",
        "complete": len(result.rows) == result.k,
        "n": result.n,
        "k": result.k,
        "d": result.d,
        "r": result.r,
        "filled_rows": len(result.rows),
        "selected_free_columns": list(result.row_bits),
        "d_actual": result.exact_minimum_distance,
        "violation_count": result.exact_violation_count,
        "minimum_margin": result.exact_minimum_margin,
        "repair_events": result.repair_events,
        "dropped_rows": result.dropped_rows,
    }
    return payload


def evaluate_priority_function(
    priority_fn: RowPriorityFn | None,
    config: RowSearchConfig,
):
    """Evaluate an evolved priority function and return OpenEvolve-style data."""
    result = search_rows_with_priority(config, priority_fn=priority_fn)
    return metrics_from_result(result), {
        "search_result": json.dumps(result_to_json_dict(result), sort_keys=True),
        "matrix_summary": json.dumps(
            result_to_json_dict(result, include_steps=False)["matrix_summary"],
            sort_keys=True,
        ),
        "selected_free_columns": json.dumps(list(result.row_bits)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experimental G-row incremental legality search for G=[I_k|P]."
    )
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--d", type=int, default=12)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--max-attempts-per-step", type=int, default=20000)
    parser.add_argument("--legal-pool-target", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--min-row-weight", type=int)
    parser.add_argument("--prefer-weight", type=int)
    parser.add_argument("--near-margin-radius", type=int, default=1)
    parser.add_argument("--repair-events", type=int, default=4)
    parser.add_argument("--repair-drop-count", type=int, default=2)
    parser.add_argument(
        "--repair-strategy",
        choices=("recent", "random", "tight"),
        default="recent",
    )
    parser.add_argument("--repair-tabu-tenure", type=int, default=16)
    parser.add_argument("--no-steps", action="store_true", help="Omit per-step diagnostics.")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    config = RowSearchConfig(
        n=args.n,
        k=args.k,
        d=args.d,
        restarts=args.restarts,
        max_attempts_per_step=args.max_attempts_per_step,
        legal_pool_target=args.legal_pool_target,
        seed=args.seed,
        min_row_weight=args.min_row_weight,
        prefer_weight=args.prefer_weight,
        near_margin_radius=args.near_margin_radius,
        repair_events=args.repair_events,
        repair_drop_count=args.repair_drop_count,
        repair_strategy=args.repair_strategy,
        repair_tabu_tenure=args.repair_tabu_tenure,
    )
    result = search_rows(config)
    print(
        json.dumps(
            result_to_json_dict(result, include_steps=not args.no_steps),
            indent=2 if args.pretty else None,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
