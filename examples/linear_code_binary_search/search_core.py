"""Shared binary linear-code utilities for the C-kernel evaluator.

The production search path for this example lives in `c_search_skeleton.c`.
This module intentionally keeps only instance parsing, exact validation helpers,
and matrix formatting used by the C runner, verifier, and tests.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class BenchmarkInstance:
    """Single binary feasibility instance for systematic parity-check search."""

    name: str
    n: int
    k: int
    target_distance: int
    restarts: int = 3
    description: str = ""

    @property
    def r(self) -> int:
        return self.n - self.k


class IncrementalForbiddenState:
    """Reference exact low-order xor layers for binary legality checks."""

    def __init__(self, r: int, distance: int):
        self.r = r
        self.distance = distance
        self.max_subset_size = max(distance - 2, 0)
        self.reachable = _initialize_reachable_layers(r, distance)
        self.forbidden = forbidden_masks_from_layers(self.reachable)
        self.selected_free_columns: List[int] = []

    def can_add(self, column_mask: int) -> bool:
        return column_mask not in self.forbidden

    def add(self, column_mask: int) -> int:
        if not self.can_add(column_mask):
            raise ValueError(f"Illegal free column {column_mask}")
        before_count = len(self.forbidden)
        _add_column_to_reachable(self.reachable, column_mask, self.max_subset_size)
        self.forbidden = forbidden_masks_from_layers(self.reachable)
        self.selected_free_columns.append(column_mask)
        return len(self.forbidden) - before_count


DEFAULT_INSTANCE = BenchmarkInstance(
    name="default_[8,4,4]",
    n=8,
    k=4,
    target_distance=4,
    restarts=3,
    description="Default single-instance target used by the C-kernel example.",
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
    """Validate a systematic construction independently of the C search state."""
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


def make_instance(
    n: int,
    k: int,
    distance: int,
    restarts: int = 3,
    name: str | None = None,
) -> BenchmarkInstance:
    """Create and validate a single C-kernel search instance."""
    if n <= 0 or k <= 0 or distance <= 0:
        raise ValueError("n, k, and d must be positive")
    if k >= n:
        raise ValueError("Require 0 < k < n")
    if distance > n:
        raise ValueError("Require d <= n")
    r = n - k
    if distance - 1 > r and k > 1:
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
