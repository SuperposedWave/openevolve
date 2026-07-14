"""Stage-one greedy search for binary maximum-code instances."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable, Sequence

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


PriorityFn = Callable[..., float]


@dataclass(frozen=True)
class MaxCodeInstance:
    """Single binary maximum-code search target."""

    name: str
    n: int
    distance: int
    restarts: int = 4


@dataclass(frozen=True)
class AcceptedWordRecord:
    """Analysis details for one accepted codeword."""

    fill_index: int
    rank: int
    word: str
    weight: int
    score: float


@dataclass(frozen=True)
class SearchResult:
    """Result from one greedy maximum-code construction."""

    codewords: tuple[int, ...]
    accepted_records: tuple[AcceptedWordRecord, ...]
    candidate_count: int
    blocked_candidate_count: int
    restart_index: int
    valid: bool
    minimum_distance: int
    forbidden_count: int
    repair_moves: int = 0
    repair_gain: int = 0
    repair_rollout_evaluations: int = 0


@dataclass(frozen=True)
class RepairReward:
    """Rollout outcome used to compare MCTS repair choices."""

    constructed_count: int
    dropped_count: int
    steps: int
    forbidden_count: int


@dataclass
class RepairStats:
    """Aggregated rollout statistics for one root drop."""

    visits: int = 0
    success_count: int = 0
    total_constructed: int = 0
    total_dropped: int = 0
    total_steps: int = 0
    total_forbidden: int = 0
    best_reward: RepairReward | None = None


DEFAULT_INSTANCE = MaxCodeInstance(
    name="default_A(17,4)",
    n=17,
    distance=4,
    restarts=4,
)


def popcount(mask: int) -> int:
    """Return the Hamming weight of a binary word."""
    return mask.bit_count()


def format_word(mask: int, n: int) -> str:
    """Format a binary word with fixed width."""
    return format(mask, f"0{n}b")


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return max(int(raw_value), minimum)
    except ValueError:
        return default


def _env_flag_enabled(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def make_instance(
    n: int,
    distance: int,
    restarts: int = 4,
    name: str | None = None,
) -> MaxCodeInstance:
    """Create and validate a maximum-code search instance."""
    if n <= 0 or distance <= 0:
        raise ValueError("n and d must be positive")
    if distance > n:
        raise ValueError("Require d <= n")
    if n > 62:
        raise ValueError("Stage-one uint-mask search requires n <= 62")
    return MaxCodeInstance(
        name=name or f"A({n},{distance})",
        n=n,
        distance=distance,
        restarts=restarts,
    )


def instance_from_env(prefix: str = "MAX_CODE_") -> MaxCodeInstance:
    """Build one instance from environment variables."""
    n = int(os.environ.get(f"{prefix}N", DEFAULT_INSTANCE.n))
    distance = int(os.environ.get(f"{prefix}D", DEFAULT_INSTANCE.distance))
    restarts = int(os.environ.get(f"{prefix}RESTARTS", DEFAULT_INSTANCE.restarts))
    return make_instance(n=n, distance=distance, restarts=restarts)


@lru_cache(maxsize=None)
def ball_offsets(n: int, distance: int) -> tuple[int, ...]:
    """All xor offsets with Hamming weight strictly below the target distance."""
    offsets = [0]
    max_weight = min(distance - 1, n)
    for weight in range(1, max_weight + 1):
        for bit_indices in combinations(range(n), weight):
            mask = 0
            for bit_index in bit_indices:
                mask |= 1 << bit_index
            offsets.append(mask)
    return tuple(offsets)


@lru_cache(maxsize=None)
def exact_weight_offsets(n: int, weight: int) -> tuple[int, ...]:
    """All xor offsets with exactly one Hamming weight."""
    if weight < 0 or weight > n:
        return tuple()
    offsets = []
    for bit_indices in combinations(range(n), weight):
        mask = 0
        for bit_index in bit_indices:
            mask |= 1 << bit_index
        offsets.append(mask)
    return tuple(offsets)


@lru_cache(maxsize=None)
def sampled_exact_weight_offsets(n: int, weight: int, sample_size: int) -> tuple[int, ...]:
    """Deterministic compact sample of an exact-weight offset shell."""
    offsets = exact_weight_offsets(n, weight)
    if sample_size <= 0 or len(offsets) <= sample_size:
        return offsets
    stride = max(len(offsets) // sample_size, 1)
    sampled = offsets[::stride][:sample_size]
    return tuple(sampled)


def _max_candidate_count() -> int:
    return _env_int("MAX_CODE_MAX_CANDIDATES", 1 << 22)


def candidate_words(n: int, distance: int) -> tuple[int, ...]:
    """Enumerate candidates not already forbidden by the forced all-zero word."""
    total_candidates = (1 << n) - 1
    if total_candidates > _max_candidate_count():
        raise ValueError(
            "Stage-one full enumeration would visit "
            f"{total_candidates} candidates; raise MAX_CODE_MAX_CANDIDATES or use a smaller n"
        )
    initially_forbidden = set(ball_offsets(n, distance))
    return tuple(
        word for word in range(1, 1 << n) if word not in initially_forbidden
    )


def _safe_priority(
    priority_fn: PriorityFn,
    word_mask: int,
    n: int,
    distance: int,
    *features: float,
) -> float:
    """Protect the fixed search skeleton from invalid evolved priority values."""
    try:
        value = priority_fn(word_mask, n, distance, *features)
    except TypeError:
        try:
            value = priority_fn(word_mask, n, distance)
        except Exception:
            value = popcount(word_mask)
    except Exception:
        value = popcount(word_mask)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = float(popcount(word_mask))
    if not math.isfinite(value):
        return float(popcount(word_mask))
    return value


def _optional_priority_fn(priority_fn: PriorityFn, name: str) -> PriorityFn | None:
    fn = getattr(priority_fn, name, None)
    return fn if callable(fn) else None


def _finite_or_default(value: object, default: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def _safe_destroy_priority(
    destroy_fn: PriorityFn | None,
    word_mask: int,
    instance: MaxCodeInstance,
    blocker_count: int,
    pair_blocker_count: int,
    candidate_score: float,
    min_neighbor_distance: int,
    avg_neighbor_distance: float,
    code_size: int,
) -> float:
    """Score a selected codeword for local removal; larger means try destroying earlier."""
    default = (
        float(blocker_count)
        + 0.25 * float(pair_blocker_count)
        - 0.001 * float(candidate_score)
    )
    if destroy_fn is None:
        return default
    weight = popcount(word_mask)
    try:
        value = destroy_fn(
            word_mask,
            instance.n,
            instance.distance,
            blocker_count,
            pair_blocker_count,
            code_size,
            weight,
            candidate_score,
            min_neighbor_distance,
            avg_neighbor_distance,
        )
    except TypeError:
        try:
            value = destroy_fn(word_mask, instance.n, instance.distance)
        except Exception:
            return default
    except Exception:
        return default
    return _finite_or_default(value, default)


def _safe_repair_priority(
    repair_fn: PriorityFn | None,
    word_mask: int,
    instance: MaxCodeInstance,
    removed_count: int,
    blocker_count: int,
    base_size: int,
    candidate_score: float,
) -> float:
    """Score a local repair candidate; larger means refill earlier."""
    if repair_fn is None:
        return candidate_score
    weight = popcount(word_mask)
    try:
        value = repair_fn(
            word_mask,
            instance.n,
            instance.distance,
            removed_count,
            blocker_count,
            base_size,
            weight,
            candidate_score,
        )
    except TypeError:
        try:
            value = repair_fn(word_mask, instance.n, instance.distance)
        except Exception:
            return candidate_score
    except Exception:
        return candidate_score
    return _finite_or_default(value, candidate_score)


def _deterministic_tiebreak(word_mask: int, restart_index: int) -> int:
    """Restart-specific deterministic tie-break for equal priority scores."""
    return (
        word_mask * 1103515245
        + restart_index * 2654435761
        + 12345
    ) & 0xFFFFFFFF


def score_static_candidates(
    instance: MaxCodeInstance,
    priority_fn: PriorityFn,
) -> tuple[tuple[int, float], ...]:
    """Score each eligible word once."""
    return tuple(
        (
            word_mask,
            _safe_priority(priority_fn, word_mask, instance.n, instance.distance),
        )
        for word_mask in candidate_words(instance.n, instance.distance)
    )


def _ranked_scores(
    static_scores: Sequence[tuple[int, float]],
    restart_index: int,
) -> tuple[tuple[int, float], ...]:
    ranked = [
        (score, _deterministic_tiebreak(word_mask, restart_index), word_mask)
        for word_mask, score in static_scores
    ]
    ranked.sort(reverse=True)
    return tuple((word_mask, score) for score, _, word_mask in ranked)


def _score_lookup(static_scores: Sequence[tuple[int, float]]) -> dict[int, float]:
    return {word_mask: float(score) for word_mask, score in static_scores}


def _rank_repair_pool(
    pool: Iterable[int],
    score_by_word: dict[int, float],
    restart_index: int,
    instance: MaxCodeInstance | None = None,
    repair_fn: PriorityFn | None = None,
    removed_count: int = 0,
    base_size: int = 0,
    blocker_counts: dict[int, int] | None = None,
) -> list[int]:
    blocker_counts = blocker_counts or {}
    ranked = [
        (
            _safe_repair_priority(
                repair_fn,
                word_mask,
                instance,
                removed_count,
                blocker_counts.get(word_mask, removed_count),
                base_size,
                score_by_word.get(word_mask, float(popcount(word_mask))),
            )
            if instance is not None
            else score_by_word.get(word_mask, float(popcount(word_mask))),
            _deterministic_tiebreak(word_mask, restart_index),
            word_mask,
        )
        for word_mask in set(pool)
    ]
    ranked.sort(reverse=True)
    return [word_mask for _score, _tie_break, word_mask in ranked]


class ForbiddenBallState:
    """Exact forbidden-set state for a binary code under minimum distance d."""

    def __init__(self, n: int, distance: int):
        self.n = n
        self.distance = distance
        self.offsets = ball_offsets(n, distance)
        self.forbidden: set[int] = set()
        self.codewords: list[int] = []

    def can_add(self, word_mask: int) -> bool:
        return word_mask not in self.forbidden

    def add(self, word_mask: int) -> None:
        if not self.can_add(word_mask):
            raise ValueError(f"Illegal codeword {word_mask}")
        self.codewords.append(word_mask)
        for offset in self.offsets:
            self.forbidden.add(word_mask ^ offset)

    def damage_features(
        self,
        word_mask: int,
        local_offsets: Sequence[int] = tuple(),
    ) -> tuple[int, int, int]:
        """Return new forbidden count, overlap count, and local availability estimate."""
        new_forbidden_count = 0
        for offset in self.offsets:
            if (word_mask ^ offset) not in self.forbidden:
                new_forbidden_count += 1
        overlap_forbidden_count = len(self.offsets) - new_forbidden_count
        local_available_count = 0
        for offset in local_offsets:
            if (word_mask ^ offset) not in self.forbidden:
                local_available_count += 1
        return new_forbidden_count, overlap_forbidden_count, local_available_count


def _forbidden_count_for_codewords(n: int, distance: int, codewords: Sequence[int]) -> int:
    forbidden: set[int] = set()
    offsets = ball_offsets(n, distance)
    for word_mask in codewords:
        for offset in offsets:
            forbidden.add(word_mask ^ offset)
    return len(forbidden)


def _rebuild_forbidden_state(
    instance: MaxCodeInstance,
    codewords: Sequence[int],
) -> ForbiddenBallState:
    state = ForbiddenBallState(instance.n, instance.distance)
    for codeword in codewords:
        state.add(codeword)
    return state


def _repair_rounds() -> int:
    return _env_int("MAX_CODE_REPAIR_ROUNDS", 2, minimum=0)


def _repair_pair_destroy_candidates() -> int:
    return _env_int("MAX_CODE_REPAIR_PAIR_DESTROY_CANDIDATES", 48, minimum=0)


def _repair_max_pool_size() -> int:
    return _env_int("MAX_CODE_REPAIR_MAX_POOL_SIZE", 4096, minimum=1)


def _candidate_blockers(
    candidate: int,
    codewords: Sequence[int],
    distance: int,
    max_blockers: int = 3,
) -> tuple[int, ...]:
    blockers: list[int] = []
    for index, codeword in enumerate(codewords):
        if hamming_distance(candidate, codeword) < distance:
            blockers.append(index)
            if len(blockers) >= max_blockers:
                break
    return tuple(blockers)


def _greedy_repair_fill(
    base_codewords: Sequence[int],
    pool: Iterable[int],
    instance: MaxCodeInstance,
    score_by_word: dict[int, float],
    restart_index: int,
    repair_fn: PriorityFn | None = None,
    removed_count: int = 0,
    blocker_counts: dict[int, int] | None = None,
) -> tuple[int, ...]:
    additions: list[int] = []
    ranked_pool = _rank_repair_pool(
        pool,
        score_by_word,
        restart_index,
        instance=instance,
        repair_fn=repair_fn,
        removed_count=removed_count,
        base_size=len(base_codewords),
        blocker_counts=blocker_counts,
    )
    max_pool_size = _repair_max_pool_size()
    if len(ranked_pool) > max_pool_size:
        ranked_pool = ranked_pool[:max_pool_size]

    for candidate in ranked_pool:
        if all(
            hamming_distance(candidate, codeword) >= instance.distance
            for codeword in base_codewords
        ) and all(
            hamming_distance(candidate, codeword) >= instance.distance
            for codeword in additions
        ):
            additions.append(candidate)
    return tuple(additions)


def _rebuild_accepted_records(
    codewords: Sequence[int],
    instance: MaxCodeInstance,
    score_by_word: dict[int, float],
) -> tuple[AcceptedWordRecord, ...]:
    records: list[AcceptedWordRecord] = []
    for index, word_mask in enumerate(codewords[1:], start=1):
        records.append(
            AcceptedWordRecord(
                fill_index=index + 1,
                rank=index,
                word=format_word(word_mask, instance.n),
                weight=popcount(word_mask),
                score=score_by_word.get(word_mask, float(popcount(word_mask))),
            )
        )
    return tuple(records)


def local_repair_codewords(
    instance: MaxCodeInstance,
    static_scores: Sequence[tuple[int, float]],
    initial_codewords: Sequence[int],
    restart_index: int,
    priority_fn: PriorityFn | None = None,
) -> tuple[tuple[int, ...], int, int]:
    """Improve a maximal packing using deterministic 1-for-many and 2-for-many moves."""
    codewords = tuple(initial_codewords)
    score_by_word = _score_lookup(static_scores)
    destroy_fn = _optional_priority_fn(priority_fn, "destroy_priority") if priority_fn else None
    repair_fn = _optional_priority_fn(priority_fn, "repair_priority") if priority_fn else None
    repair_moves = 0
    repair_gain = 0
    rounds = _repair_rounds()
    if rounds <= 0 or len(codewords) <= 1:
        return codewords, repair_moves, repair_gain

    for round_index in range(rounds):
        selected = set(codewords)
        single_blocked: dict[int, list[int]] = {}
        pair_blocked: dict[tuple[int, int], list[int]] = {}
        blocker_counts: dict[int, int] = {}
        pair_blocker_count_by_index: dict[int, int] = {}

        for candidate, _score in static_scores:
            if candidate in selected:
                continue
            blockers = _candidate_blockers(candidate, codewords, instance.distance, 3)
            blocker_counts[candidate] = len(blockers)
            if len(blockers) == 0:
                single_blocked.setdefault(-1, []).append(candidate)
            elif len(blockers) == 1:
                single_blocked.setdefault(blockers[0], []).append(candidate)
            elif len(blockers) == 2:
                left, right = sorted(blockers)
                pair_blocked.setdefault((left, right), []).append(candidate)
                pair_blocker_count_by_index[left] = pair_blocker_count_by_index.get(left, 0) + 1
                pair_blocker_count_by_index[right] = pair_blocker_count_by_index.get(right, 0) + 1

        destroy_scores: dict[int, float] = {}
        for index, word_mask in enumerate(codewords):
            if index == 0:
                continue
            min_distance = instance.n + 1
            total_distance = 0
            distance_count = 0
            for other_index, other in enumerate(codewords):
                if other_index == index:
                    continue
                distance = hamming_distance(word_mask, other)
                min_distance = min(min_distance, distance)
                total_distance += distance
                distance_count += 1
            avg_distance = total_distance / max(1, distance_count)
            destroy_scores[index] = _safe_destroy_priority(
                destroy_fn,
                word_mask,
                instance,
                len(single_blocked.get(index, ())),
                pair_blocker_count_by_index.get(index, 0),
                score_by_word.get(word_mask, float(popcount(word_mask))),
                min_distance if min_distance <= instance.n else instance.n,
                avg_distance,
                len(codewords),
            )

        best_removed: tuple[int, ...] = tuple()
        best_additions: tuple[int, ...] = tuple()
        best_gain = 0
        best_score = -math.inf

        legal_pool = single_blocked.get(-1)
        if legal_pool:
            additions = _greedy_repair_fill(
                codewords,
                legal_pool,
                instance,
                score_by_word,
                restart_index + 1009 * (round_index + 1),
                repair_fn=repair_fn,
                removed_count=0,
                blocker_counts=blocker_counts,
            )
            if additions:
                best_removed = tuple()
                best_additions = additions
                best_gain = len(additions)
                best_score = sum(
                    _safe_repair_priority(
                        repair_fn,
                        word,
                        instance,
                        0,
                        blocker_counts.get(word, 0),
                        len(codewords),
                        score_by_word.get(word, float(popcount(word))),
                    )
                    for word in additions
                )

        for removed_index, pool in single_blocked.items():
            if removed_index <= 0:
                continue
            base = tuple(
                codeword for index, codeword in enumerate(codewords) if index != removed_index
            )
            additions = _greedy_repair_fill(
                base,
                pool,
                instance,
                score_by_word,
                restart_index + 2003 * (round_index + 1) + removed_index,
                repair_fn=repair_fn,
                removed_count=1,
                blocker_counts=blocker_counts,
            )
            gain = len(additions) - 1
            score = destroy_scores.get(removed_index, 0.0) + sum(
                _safe_repair_priority(
                    repair_fn,
                    word,
                    instance,
                    1,
                    blocker_counts.get(word, 1),
                    len(base),
                    score_by_word.get(word, float(popcount(word))),
                )
                for word in additions
            )
            if gain > best_gain or (gain == best_gain and gain > 0 and score > best_score):
                best_removed = (removed_index,)
                best_additions = additions
                best_gain = gain
                best_score = score

        pair_candidate_count = _repair_pair_destroy_candidates()
        if pair_candidate_count > 0:
            destroy_candidates = sorted(
                (
                    index
                    for index, pool in single_blocked.items()
                    if index > 0 and pool
                ),
                key=lambda index: (
                    destroy_scores.get(index, 0.0),
                    len(single_blocked.get(index, ())),
                ),
                reverse=True,
            )[:pair_candidate_count]
            pair_keys = set(pair_blocked)
            for left_pos, left in enumerate(destroy_candidates):
                for right in destroy_candidates[left_pos + 1 :]:
                    pair_keys.add((min(left, right), max(left, right)))

            for left, right in pair_keys:
                if left <= 0 or right <= 0 or left == right:
                    continue
                pool = []
                pool.extend(single_blocked.get(left, ()))
                pool.extend(single_blocked.get(right, ()))
                pool.extend(pair_blocked.get((min(left, right), max(left, right)), ()))
                if not pool:
                    continue
                removed = {left, right}
                base = tuple(
                    codeword for index, codeword in enumerate(codewords) if index not in removed
                )
                additions = _greedy_repair_fill(
                    base,
                    pool,
                    instance,
                    score_by_word,
                    restart_index + 4001 * (round_index + 1) + left * 131 + right,
                    repair_fn=repair_fn,
                    removed_count=2,
                    blocker_counts=blocker_counts,
                )
                gain = len(additions) - 2
                score = destroy_scores.get(left, 0.0) + destroy_scores.get(right, 0.0)
                score += sum(
                    _safe_repair_priority(
                        repair_fn,
                        word,
                        instance,
                        2,
                        blocker_counts.get(word, 2),
                        len(base),
                        score_by_word.get(word, float(popcount(word))),
                    )
                    for word in additions
                )
                if gain > best_gain or (gain == best_gain and gain > 0 and score > best_score):
                    best_removed = (left, right)
                    best_additions = additions
                    best_gain = gain
                    best_score = score

        if best_gain <= 0:
            break

        removed_set = set(best_removed)
        codewords = tuple(
            codeword for index, codeword in enumerate(codewords) if index not in removed_set
        ) + best_additions
        repair_moves += 1
        repair_gain += best_gain

    return codewords, repair_moves, repair_gain


class ParityTransformState:
    """State for searching A(n-1,3) before parity-extending to A(n,4)."""

    def __init__(self, search_n: int):
        self.search_n = search_n
        self.center_forbidden_offsets = ball_offsets(search_n, 3)
        local_sample_size = _env_int("MAX_CODE_LOCAL_SAMPLE_SIZE", 64, minimum=0)
        self.local_offsets = sampled_exact_weight_offsets(search_n, 3, local_sample_size)
        self.forbidden_centers: set[int] = set()
        self.codewords: list[int] = []

    @property
    def universe_size(self) -> int:
        return 1 << self.search_n

    def can_add(self, word_mask: int) -> bool:
        return word_mask not in self.forbidden_centers

    def add(self, word_mask: int) -> None:
        if not self.can_add(word_mask):
            raise ValueError(f"Illegal transformed center {word_mask}")
        self.codewords.append(word_mask)
        for offset in self.center_forbidden_offsets:
            self.forbidden_centers.add(word_mask ^ offset)

    def damage_features(self, word_mask: int) -> tuple[int, int, int]:
        """Return new forbidden count, overlap count, and local availability estimate."""
        new_forbidden_count = 0
        for offset in self.center_forbidden_offsets:
            if (word_mask ^ offset) not in self.forbidden_centers:
                new_forbidden_count += 1
        overlap_forbidden_count = len(self.center_forbidden_offsets) - new_forbidden_count
        local_available_count = 0
        for offset in self.local_offsets:
            if (word_mask ^ offset) not in self.forbidden_centers:
                local_available_count += 1
        return new_forbidden_count, overlap_forbidden_count, local_available_count


def _rebuild_parity_state(search_n: int, centers: Sequence[int]) -> ParityTransformState:
    state = ParityTransformState(search_n)
    for center in centers:
        state.add(center)
    return state


def _rebuild_parity_accepted_records(
    centers: Sequence[int],
    instance: MaxCodeInstance,
    score_by_center: dict[int, float],
) -> tuple[AcceptedWordRecord, ...]:
    search_n = instance.n - 1
    records: list[AcceptedWordRecord] = []
    for index, center in enumerate(centers[1:], start=1):
        extended_word = parity_extend_word(center, search_n)
        records.append(
            AcceptedWordRecord(
                fill_index=index + 1,
                rank=index,
                word=format_word(extended_word, instance.n),
                weight=popcount(extended_word),
                score=score_by_center.get(center, 0.0),
            )
        )
    return tuple(records)


def _parity_sample_dynamic_candidate(
    state: ParityTransformState,
    priority_fn: PriorityFn,
    restart_index: int,
    rng: random.Random,
    pool_size: int,
    attempts_per_refill: int,
    tabu: set[int],
) -> tuple[int | None, float, int, int]:
    """Sample legal centers and return the best dynamic-priority candidate."""
    pool: list[tuple[float, float, int]] = []
    local_seen: set[int] = set()
    blocked = 0
    attempts = 0
    while len(pool) < pool_size and attempts < attempts_per_refill:
        attempts += 1
        candidate = rng.randrange(1, state.universe_size)
        if candidate in local_seen or candidate in tabu:
            continue
        local_seen.add(candidate)
        if not state.can_add(candidate):
            blocked += 1
            continue
        new_forbidden, overlap_forbidden, local_available = state.damage_features(candidate)
        score = _safe_priority(
            priority_fn,
            candidate,
            state.search_n,
            3,
            len(state.codewords),
            popcount(candidate),
            new_forbidden,
            overlap_forbidden,
            local_available,
        )
        pool.append((score, rng.random(), candidate))
    if not pool:
        return None, 0.0, blocked, 0
    pool.sort(reverse=True)
    score, _tie, candidate = pool[0]
    return candidate, float(score), blocked, len(pool)


def _parity_count_sampled_legal(
    state: ParityTransformState,
    rng: random.Random,
    sample_count: int,
    tabu: set[int],
) -> int:
    legal = 0
    seen: set[int] = set()
    attempts = 0
    max_attempts = max(sample_count * 4, sample_count + 128)
    while len(seen) < sample_count and attempts < max_attempts:
        attempts += 1
        candidate = rng.randrange(1, state.universe_size)
        if candidate in seen or candidate in tabu:
            continue
        seen.add(candidate)
        if state.can_add(candidate):
            legal += 1
    return legal


def _parity_drop_choice_is_better(
    candidate_legal_count: int,
    candidate_release: int,
    candidate_tie: int,
    current_legal_count: int,
    current_release: int,
    current_tie: int,
) -> bool:
    return (
        candidate_legal_count > current_legal_count
        or (
            candidate_legal_count == current_legal_count
            and candidate_release > current_release
        )
        or (
            candidate_legal_count == current_legal_count
            and candidate_release == current_release
            and candidate_tie > current_tie
        )
    )


def _parity_choose_rollout_drop_index(
    centers: Sequence[int],
    search_n: int,
    restart_index: int,
    repair_event_index: int,
    step: int,
    before_forbidden_count: int,
    candidate_window: int,
    tabu: set[int],
    drop_topk: int,
    rng: random.Random,
) -> int | None:
    removable = [index for index in range(1, len(centers)) if centers[index] not in tabu]
    if not removable:
        return None
    if drop_topk == 0:
        return removable[rng.randrange(len(removable))]

    top: list[tuple[int, int, int, int]] = []
    for drop_index in removable:
        trial_centers = [center for index, center in enumerate(centers) if index != drop_index]
        trial_state = _rebuild_parity_state(search_n, trial_centers)
        release = max(0, before_forbidden_count - len(trial_state.forbidden_centers))
        legal_count = _parity_count_sampled_legal(
            trial_state,
            rng,
            candidate_window,
            tabu | {centers[drop_index]},
        )
        tie = _deterministic_tiebreak(
            centers[drop_index],
            restart_index + repair_event_index + step + 4099,
        )
        item = (legal_count, release, tie, drop_index)
        insert_at = len(top)
        while insert_at > 0 and _parity_drop_choice_is_better(
            legal_count,
            release,
            tie,
            top[insert_at - 1][0],
            top[insert_at - 1][1],
            top[insert_at - 1][2],
        ):
            insert_at -= 1
        top.insert(insert_at, item)
        if len(top) > drop_topk:
            top.pop()

    if not top:
        return removable[rng.randrange(len(removable))]
    return top[rng.randrange(len(top))][3]


def _repair_reward_is_better(
    candidate: RepairReward,
    current: RepairReward | None,
) -> bool:
    if current is None:
        return True
    if candidate.constructed_count != current.constructed_count:
        return candidate.constructed_count > current.constructed_count
    if candidate.dropped_count != current.dropped_count:
        return candidate.dropped_count < current.dropped_count
    if candidate.steps != current.steps:
        return candidate.steps < current.steps
    return candidate.forbidden_count < current.forbidden_count


def _repair_stats_is_better(
    candidate: RepairStats,
    current: RepairStats | None,
) -> bool:
    if candidate.visits <= 0 or candidate.best_reward is None:
        return False
    if current is None or current.visits <= 0 or current.best_reward is None:
        return True
    if candidate.success_count * current.visits != current.success_count * candidate.visits:
        return candidate.success_count * current.visits > current.success_count * candidate.visits
    if _repair_reward_is_better(candidate.best_reward, current.best_reward):
        return True
    if _repair_reward_is_better(current.best_reward, candidate.best_reward):
        return False
    if candidate.total_constructed * current.visits != current.total_constructed * candidate.visits:
        return candidate.total_constructed * current.visits > current.total_constructed * candidate.visits
    if candidate.total_dropped * current.visits != current.total_dropped * candidate.visits:
        return candidate.total_dropped * current.visits < current.total_dropped * candidate.visits
    if candidate.total_steps * current.visits != current.total_steps * candidate.visits:
        return candidate.total_steps * current.visits < current.total_steps * candidate.visits
    return candidate.total_forbidden * current.visits < current.total_forbidden * candidate.visits


def _parity_update_root_stats(
    stats: RepairStats,
    reward: RepairReward,
    original_count: int,
) -> None:
    stats.visits += 1
    stats.total_constructed += reward.constructed_count
    stats.total_dropped += reward.dropped_count
    stats.total_steps += reward.steps
    stats.total_forbidden += reward.forbidden_count
    if reward.constructed_count >= original_count:
        stats.success_count += 1
    if _repair_reward_is_better(reward, stats.best_reward):
        stats.best_reward = reward


def _parity_rollout_after_first_drop(
    centers: Sequence[int],
    first_drop_index: int,
    priority_fn: PriorityFn,
    search_n: int,
    restart_index: int,
    repair_event_index: int,
    seed: int,
    pool_size: int,
    attempts_per_refill: int,
    rollout_depth: int,
    drop_topk: int,
    candidate_window: int,
    tabu: set[int],
) -> tuple[RepairReward, int]:
    trial_centers = [center for index, center in enumerate(centers) if index != first_drop_index]
    trial_state = _rebuild_parity_state(search_n, trial_centers)
    trial_tabu = set(tabu)
    trial_tabu.add(centers[first_drop_index])
    rng = random.Random(
        seed
        ^ (restart_index << 32)
        ^ (repair_event_index << 16)
        ^ (first_drop_index + 1)
    )
    dropped_count = 1
    steps = 0
    evaluations = 0

    while len(trial_state.codewords) < trial_state.universe_size and steps < rollout_depth:
        candidate, _score, _blocked, evals = _parity_sample_dynamic_candidate(
            trial_state,
            priority_fn,
            restart_index + repair_event_index + steps + 1,
            rng,
            pool_size,
            attempts_per_refill,
            trial_tabu,
        )
        evaluations += evals
        if candidate is not None:
            trial_state.add(candidate)
            steps += 1
            continue

        if len(trial_state.codewords) <= 1:
            break
        drop_index = _parity_choose_rollout_drop_index(
            trial_state.codewords,
            search_n,
            restart_index,
            repair_event_index,
            steps,
            len(trial_state.forbidden_centers),
            candidate_window,
            trial_tabu,
            drop_topk,
            rng,
        )
        if drop_index is None:
            break
        dropped = trial_state.codewords[drop_index]
        trial_tabu.add(dropped)
        trial_centers = [
            center for index, center in enumerate(trial_state.codewords) if index != drop_index
        ]
        trial_state = _rebuild_parity_state(search_n, trial_centers)
        dropped_count += 1
        steps += 1

    return (
        RepairReward(
            constructed_count=len(trial_state.codewords),
            dropped_count=dropped_count,
            steps=steps,
            forbidden_count=len(trial_state.forbidden_centers),
        ),
        evaluations,
    )


def _parity_choose_mcts_drop_index_with_evaluations(
    state: ParityTransformState,
    priority_fn: PriorityFn,
    restart_index: int,
    repair_event_index: int,
    seed: int,
    pool_size: int,
    attempts_per_refill: int,
    tabu: set[int],
) -> tuple[int | None, int]:
    centers = tuple(state.codewords)
    if len(centers) <= 1:
        return None, 0

    removable = [index for index in range(1, len(centers)) if centers[index] not in tabu]
    if not removable:
        return None, 0

    rollout_depth = _parity_repair_mcts_depth()
    simulations = _parity_repair_mcts_simulations()
    drop_topk = _parity_repair_mcts_drop_topk()
    candidate_window = _parity_repair_candidate_window(state.search_n)
    root_stats = {index: RepairStats() for index in removable}
    total_evaluations = 0

    for simulation in range(simulations):
        drop_index = removable[simulation % len(removable)]
        reward, evaluations = _parity_rollout_after_first_drop(
            centers,
            drop_index,
            priority_fn,
            state.search_n,
            restart_index,
            repair_event_index,
            seed + simulation * 0x9E3779B97F4A7C15,
            pool_size,
            attempts_per_refill,
            rollout_depth,
            drop_topk,
            candidate_window,
            tabu,
        )
        total_evaluations += evaluations
        _parity_update_root_stats(root_stats[drop_index], reward, len(centers))

    best_index: int | None = None
    for drop_index, stats in root_stats.items():
        current = root_stats[best_index] if best_index is not None else None
        if _repair_stats_is_better(stats, current):
            best_index = drop_index

    if best_index is None:
        return None, total_evaluations
    best_reward = root_stats[best_index].best_reward
    if best_reward is None or best_reward.constructed_count < len(centers):
        return None, total_evaluations
    return best_index, total_evaluations


def _parity_choose_mcts_drop_index(
    state: ParityTransformState,
    priority_fn: PriorityFn,
    restart_index: int,
    repair_event_index: int,
    seed: int,
    pool_size: int,
    attempts_per_refill: int,
    tabu: set[int],
) -> int | None:
    drop_index, _evaluations = _parity_choose_mcts_drop_index_with_evaluations(
        state,
        priority_fn,
        restart_index,
        repair_event_index,
        seed,
        pool_size,
        attempts_per_refill,
        tabu,
    )
    return drop_index


def _parity_apply_drop(
    state: ParityTransformState,
    drop_index: int,
) -> tuple[ParityTransformState, int]:
    dropped = state.codewords[drop_index]
    centers = [center for index, center in enumerate(state.codewords) if index != drop_index]
    return _rebuild_parity_state(state.search_n, centers), dropped


def _parity_add_tabu(
    tabu_queue: list[int],
    tabu_set: set[int],
    word_mask: int,
    tenure: int,
) -> None:
    if tenure <= 0:
        return
    tabu_queue.append(word_mask)
    tabu_set.add(word_mask)
    while len(tabu_queue) > tenure:
        expired = tabu_queue.pop(0)
        if expired not in tabu_queue:
            tabu_set.discard(expired)


def _dynamic_repair_local_offsets(instance: MaxCodeInstance) -> tuple[int, ...]:
    return sampled_exact_weight_offsets(
        instance.n,
        instance.distance,
        _env_int("MAX_CODE_LOCAL_SAMPLE_SIZE", 64, minimum=0),
    )


def _choose_dynamic_candidate_from_ranked(
    ranked_scores: Sequence[tuple[int, float]],
    state: ForbiddenBallState,
    priority_fn: PriorityFn,
    instance: MaxCodeInstance,
    restart_index: int,
    window_size: int,
    tabu: set[int],
    local_offsets: Sequence[int],
) -> tuple[int | None, float, int, int]:
    """Choose the best dynamic candidate from the next legal ranked window."""
    best_word: int | None = None
    best_score = -math.inf
    best_tie = -1
    blocked = 0
    evaluations = 0
    legal_seen = 0
    step = len(state.codewords)
    effective_window = window_size if window_size > 0 else len(ranked_scores)

    for word_mask, _static_score in ranked_scores:
        if word_mask in tabu:
            continue
        if not state.can_add(word_mask):
            blocked += 1
            continue
        new_forbidden, overlap_forbidden, local_available = state.damage_features(
            word_mask,
            local_offsets,
        )
        score = _safe_priority(
            priority_fn,
            word_mask,
            instance.n,
            instance.distance,
            step,
            popcount(word_mask),
            new_forbidden,
            overlap_forbidden,
            local_available,
        )
        tie = _deterministic_tiebreak(word_mask, restart_index + step + 1)
        evaluations += 1
        if score > best_score or (score == best_score and tie > best_tie):
            best_word = word_mask
            best_score = score
            best_tie = tie
        legal_seen += 1
        if legal_seen >= effective_window:
            break

    if best_word is None:
        return None, 0.0, blocked, evaluations
    return best_word, float(best_score), blocked, evaluations


def _count_legal_candidates_in_ranked_prefix(
    ranked_scores: Sequence[tuple[int, float]],
    state: ForbiddenBallState,
    prefix_size: int,
    tabu: set[int],
) -> int:
    legal_count = 0
    limit = prefix_size if prefix_size > 0 else len(ranked_scores)
    for word_mask, _score in ranked_scores[:limit]:
        if word_mask in tabu:
            continue
        if state.can_add(word_mask):
            legal_count += 1
    return legal_count


def _choose_rollout_drop_index(
    codewords: Sequence[int],
    instance: MaxCodeInstance,
    ranked_scores: Sequence[tuple[int, float]],
    restart_index: int,
    repair_event_index: int,
    step: int,
    before_forbidden_count: int,
    candidate_window: int,
    tabu: set[int],
    drop_topk: int,
    rng: random.Random,
) -> int | None:
    removable = [index for index in range(1, len(codewords)) if codewords[index] not in tabu]
    if not removable:
        return None
    if drop_topk == 0:
        return removable[rng.randrange(len(removable))]

    top: list[tuple[int, int, int, int]] = []
    for drop_index in removable:
        trial_codewords = [
            codeword for index, codeword in enumerate(codewords) if index != drop_index
        ]
        trial_state = _rebuild_forbidden_state(instance, trial_codewords)
        release = max(0, before_forbidden_count - len(trial_state.forbidden))
        legal_count = _count_legal_candidates_in_ranked_prefix(
            ranked_scores,
            trial_state,
            candidate_window,
            tabu | {codewords[drop_index]},
        )
        tie = _deterministic_tiebreak(
            codewords[drop_index],
            restart_index + repair_event_index + step + 4099,
        )
        item = (legal_count, release, tie, drop_index)
        insert_at = len(top)
        while insert_at > 0 and _parity_drop_choice_is_better(
            legal_count,
            release,
            tie,
            top[insert_at - 1][0],
            top[insert_at - 1][1],
            top[insert_at - 1][2],
        ):
            insert_at -= 1
        top.insert(insert_at, item)
        if len(top) > drop_topk:
            top.pop()

    if not top:
        return removable[rng.randrange(len(removable))]
    return top[rng.randrange(len(top))][3]


def _rollout_after_first_drop(
    codewords: Sequence[int],
    first_drop_index: int,
    instance: MaxCodeInstance,
    ranked_scores: Sequence[tuple[int, float]],
    priority_fn: PriorityFn,
    restart_index: int,
    repair_event_index: int,
    seed: int,
    dynamic_window: int,
    rollout_depth: int,
    drop_topk: int,
    candidate_window: int,
    tabu: set[int],
    local_offsets: Sequence[int],
) -> tuple[RepairReward, int]:
    trial_codewords = [
        codeword for index, codeword in enumerate(codewords) if index != first_drop_index
    ]
    trial_state = _rebuild_forbidden_state(instance, trial_codewords)
    trial_tabu = set(tabu)
    trial_tabu.add(codewords[first_drop_index])
    rng = random.Random(
        seed
        ^ (restart_index << 32)
        ^ (repair_event_index << 16)
        ^ (first_drop_index + 1)
    )
    dropped_count = 1
    steps = 0
    evaluations = 0

    while steps < rollout_depth:
        candidate, _score, _blocked, evals = _choose_dynamic_candidate_from_ranked(
            ranked_scores,
            trial_state,
            priority_fn,
            instance,
            restart_index + repair_event_index + steps + 1,
            dynamic_window,
            trial_tabu,
            local_offsets,
        )
        evaluations += evals
        if candidate is not None:
            trial_state.add(candidate)
            trial_codewords.append(candidate)
            steps += 1
            continue

        if len(trial_codewords) <= 1:
            break
        drop_index = _choose_rollout_drop_index(
            trial_codewords,
            instance,
            ranked_scores,
            restart_index,
            repair_event_index,
            steps,
            len(trial_state.forbidden),
            candidate_window,
            trial_tabu,
            drop_topk,
            rng,
        )
        if drop_index is None:
            break
        dropped = trial_codewords[drop_index]
        trial_tabu.add(dropped)
        trial_codewords = [
            codeword for index, codeword in enumerate(trial_codewords) if index != drop_index
        ]
        trial_state = _rebuild_forbidden_state(instance, trial_codewords)
        dropped_count += 1
        steps += 1

    reward = RepairReward(
        constructed_count=len(trial_codewords),
        dropped_count=dropped_count,
        steps=steps,
        forbidden_count=len(trial_state.forbidden),
    )
    return reward, evaluations


def _choose_mcts_drop_index(
    codewords: Sequence[int],
    instance: MaxCodeInstance,
    ranked_scores: Sequence[tuple[int, float]],
    priority_fn: PriorityFn,
    restart_index: int,
    repair_event_index: int,
    seed: int,
    dynamic_window: int,
    tabu: set[int],
    local_offsets: Sequence[int],
) -> tuple[int | None, int]:
    removable = [index for index in range(1, len(codewords)) if codewords[index] not in tabu]
    if not removable:
        return None, 0

    rollout_depth = _parity_repair_mcts_depth()
    simulations = _parity_repair_mcts_simulations()
    drop_topk = _parity_repair_mcts_drop_topk()
    candidate_window = _parity_repair_candidate_window(instance.n)
    root_stats = {index: RepairStats() for index in removable}
    total_evaluations = 0

    for simulation in range(simulations):
        drop_index = removable[simulation % len(removable)]
        reward, evaluations = _rollout_after_first_drop(
            codewords,
            drop_index,
            instance,
            ranked_scores,
            priority_fn,
            restart_index,
            repair_event_index,
            seed + simulation * 0x9E3779B97F4A7C15,
            dynamic_window,
            rollout_depth,
            drop_topk,
            candidate_window,
            tabu,
            local_offsets,
        )
        total_evaluations += evaluations
        _parity_update_root_stats(root_stats[drop_index], reward, len(codewords))

    best_index: int | None = None
    for drop_index, stats in root_stats.items():
        current = root_stats[best_index] if best_index is not None else None
        if _repair_stats_is_better(stats, current):
            best_index = drop_index

    if best_index is None:
        return None, total_evaluations
    best_reward = root_stats[best_index].best_reward
    if best_reward is None or best_reward.constructed_count < len(codewords):
        return None, total_evaluations
    return best_index, total_evaluations


def greedy_construct(
    instance: MaxCodeInstance,
    static_scores: Sequence[tuple[int, float]],
    restart_index: int,
    priority_fn: PriorityFn | None = None,
) -> SearchResult:
    """Run one sorted greedy pass for a maximum-code instance."""
    search_state = ForbiddenBallState(instance.n, instance.distance)
    search_state.add(0)
    accepted_records: list[AcceptedWordRecord] = []
    blocked_candidate_count = 0
    ranked_scores = _ranked_scores(static_scores, restart_index)

    for rank, (word_mask, score) in enumerate(ranked_scores, start=1):
        if search_state.can_add(word_mask):
            search_state.add(word_mask)
            accepted_records.append(
                AcceptedWordRecord(
                    fill_index=len(search_state.codewords),
                    rank=rank,
                    word=format_word(word_mask, instance.n),
                    weight=popcount(word_mask),
                    score=float(score),
                )
            )
        else:
            blocked_candidate_count += 1

    codewords = tuple(search_state.codewords)
    repair_moves = 0
    repair_gain = 0
    if _env_flag_enabled("MAX_CODE_LOCAL_REPAIR", default=True):
        codewords, repair_moves, repair_gain = local_repair_codewords(
            instance,
            static_scores,
            codewords,
            restart_index,
            priority_fn=priority_fn,
        )
        accepted_records = list(
            _rebuild_accepted_records(
                codewords,
                instance,
                _score_lookup(static_scores),
            )
        )

    minimum_distance = actual_minimum_distance(codewords)
    valid = minimum_distance >= instance.distance
    return SearchResult(
        codewords=codewords,
        accepted_records=tuple(accepted_records),
        candidate_count=(1 << instance.n) - 1,
        blocked_candidate_count=blocked_candidate_count,
        restart_index=restart_index,
        valid=valid,
        minimum_distance=minimum_distance,
        forbidden_count=_forbidden_count_for_codewords(instance.n, instance.distance, codewords),
        repair_moves=repair_moves,
        repair_gain=repair_gain,
    )


def dynamic_mcts_construct(
    instance: MaxCodeInstance,
    static_scores: Sequence[tuple[int, float]],
    restart_index: int,
    priority_fn: PriorityFn,
) -> SearchResult:
    """Run dynamic-window greedy fill with bounded MCTS repair for stuck states."""
    search_state = ForbiddenBallState(instance.n, instance.distance)
    search_state.add(0)
    ranked_scores = _ranked_scores(static_scores, restart_index)
    local_offsets = _dynamic_repair_local_offsets(instance)
    score_by_word = _score_lookup(static_scores)
    accepted_records: list[AcceptedWordRecord] = []
    blocked_candidate_count = 0
    repair_moves = 0
    repair_gain_start_size: int | None = None
    repair_rollout_evaluations = 0
    repair_events = _parity_repair_events()
    repair_drop_count = _parity_repair_drop_count()
    repair_tabu_tenure = _parity_repair_tabu_tenure()
    repair_tabu_queue: list[int] = []
    repair_tabu_set: set[int] = set()
    dynamic_window = _parity_repair_candidate_window(instance.n)

    while len(search_state.forbidden) < (1 << instance.n):
        candidate, score, blocked, _evaluations = _choose_dynamic_candidate_from_ranked(
            ranked_scores,
            search_state,
            priority_fn,
            instance,
            restart_index,
            dynamic_window,
            repair_tabu_set,
            local_offsets,
        )
        blocked_candidate_count += blocked
        if candidate is not None:
            search_state.add(candidate)
            score_by_word[candidate] = float(score)
            accepted_records.append(
                AcceptedWordRecord(
                    fill_index=len(search_state.codewords),
                    rank=len(accepted_records) + 1,
                    word=format_word(candidate, instance.n),
                    weight=popcount(candidate),
                    score=float(score),
                )
            )
            continue

        repaired = False
        if repair_moves < repair_events and len(search_state.codewords) > 1:
            for _drop_event in range(repair_drop_count):
                if repair_moves >= repair_events or len(search_state.codewords) <= 1:
                    break
                drop_index, rollout_evaluations = _choose_mcts_drop_index(
                    tuple(search_state.codewords),
                    instance,
                    ranked_scores,
                    priority_fn,
                    restart_index,
                    repair_moves,
                    _sampled_seed(restart_index),
                    dynamic_window,
                    repair_tabu_set,
                    local_offsets,
                )
                repair_rollout_evaluations += rollout_evaluations
                if drop_index is None:
                    break
                if repair_gain_start_size is None:
                    repair_gain_start_size = len(search_state.codewords)
                dropped = search_state.codewords[drop_index]
                retained = [
                    codeword
                    for index, codeword in enumerate(search_state.codewords)
                    if index != drop_index
                ]
                search_state = _rebuild_forbidden_state(instance, retained)
                _parity_add_tabu(
                    repair_tabu_queue,
                    repair_tabu_set,
                    dropped,
                    repair_tabu_tenure,
                )
                repair_moves += 1
                repaired = True
        if repaired:
            continue
        break

    codewords = tuple(search_state.codewords)
    accepted_records = list(_rebuild_accepted_records(codewords, instance, score_by_word))
    repair_gain = (
        max(0, len(codewords) - repair_gain_start_size)
        if repair_gain_start_size is not None
        else 0
    )
    minimum_distance = actual_minimum_distance(codewords)
    valid = minimum_distance >= instance.distance
    return SearchResult(
        codewords=codewords,
        accepted_records=tuple(accepted_records),
        candidate_count=(1 << instance.n) - 1,
        blocked_candidate_count=blocked_candidate_count,
        restart_index=restart_index,
        valid=valid,
        minimum_distance=minimum_distance,
        forbidden_count=len(search_state.forbidden),
        repair_moves=repair_moves,
        repair_gain=repair_gain,
        repair_rollout_evaluations=repair_rollout_evaluations,
    )


def _sampled_seed(restart_index: int) -> int:
    base_seed = _env_int("MAX_CODE_RANDOM_SEED", 0, minimum=0)
    return (base_seed + restart_index * 1000003) & 0xFFFFFFFF


def _parity_pool_size(search_n: int) -> int:
    default_size = max(1024, min(8192, search_n * 512))
    return _env_int("MAX_CODE_PARITY_POOL_SIZE", default_size)


def _parity_attempts_per_refill(pool_size: int) -> int:
    return _env_int("MAX_CODE_PARITY_ATTEMPTS_PER_REFILL", pool_size * 20)


def _parity_max_refills(search_n: int) -> int:
    default_refills = max(128, min(2048, (1 << max(search_n - 8, 0)) * 4))
    return _env_int("MAX_CODE_PARITY_MAX_REFILLS", default_refills)


def _parity_max_stale_refills() -> int:
    return _env_int("MAX_CODE_PARITY_MAX_STALE_REFILLS", 24)


def _parity_full_scan_n_limit() -> int:
    return _env_int("MAX_CODE_PARITY_FULL_SCAN_N", 18)


def _parity_full_batch_size() -> int:
    return _env_int("MAX_CODE_PARITY_FULL_BATCH_SIZE", 64)


def _parity_repair_mode() -> str:
    return os.environ.get("MAX_CODE_REPAIR_MODE", "greedy").strip().lower()


def _parity_repair_events() -> int:
    return _env_int("MAX_CODE_REPAIR_EVENTS", 4, minimum=0)


def _parity_repair_drop_count() -> int:
    return _env_int("MAX_CODE_REPAIR_DROP_COUNT", 1, minimum=1)


def _parity_repair_tabu_tenure() -> int:
    default_tenure = _parity_repair_events() * _parity_repair_drop_count()
    return _env_int("MAX_CODE_REPAIR_TABU_TENURE", default_tenure, minimum=0)


def _parity_repair_candidate_window(search_n: int) -> int:
    default_window = max(4096, min(65536, search_n * 4096))
    return _env_int("MAX_CODE_REPAIR_CANDIDATE_WINDOW", default_window)


def _parity_repair_mcts_simulations() -> int:
    return _env_int("MAX_CODE_REPAIR_MCTS_SIMULATIONS", 64, minimum=1)


def _parity_repair_mcts_depth() -> int:
    return _env_int("MAX_CODE_REPAIR_MCTS_DEPTH", 4, minimum=1)


def _parity_repair_mcts_drop_topk() -> int:
    return _env_int("MAX_CODE_REPAIR_MCTS_DROP_TOPK", 2, minimum=0)


def parity_extend_word(word_mask: int, search_n: int) -> int:
    """Append an overall parity bit, mapping distance 3 to distance 4."""
    parity_bit = popcount(word_mask) & 1
    return word_mask | (parity_bit << search_n)


def parity_transform_full_scan_construct(
    instance: MaxCodeInstance,
    priority_fn: PriorityFn,
    restart_index: int,
) -> SearchResult:
    """Dynamic full-scan construction for moderate d=4 instances."""
    search_n = instance.n - 1
    search_state = ParityTransformState(search_n)
    search_state.add(0)
    accepted_records: list[AcceptedWordRecord] = []
    blocked_candidate_count = 0
    batch_size = _parity_full_batch_size()
    scan_index = 0
    repair_moves = 0
    repair_gain_start_size: int | None = None
    repair_rollout_evaluations = 0
    repair_mode = _parity_repair_mode()
    repair_events = _parity_repair_events()
    repair_drop_count = _parity_repair_drop_count()
    repair_tabu_tenure = _parity_repair_tabu_tenure()
    repair_tabu_queue: list[int] = []
    repair_tabu_set: set[int] = set()
    score_by_center: dict[int, float] = {0: 0.0}

    while len(search_state.forbidden_centers) < search_state.universe_size:
        scan_index += 1
        pool: list[tuple[float, int, int]] = []
        for candidate in range(1, search_state.universe_size):
            if candidate in repair_tabu_set:
                continue
            if not search_state.can_add(candidate):
                blocked_candidate_count += 1
                continue
            new_forbidden, overlap_forbidden, local_available = search_state.damage_features(
                candidate
            )
            score = _safe_priority(
                priority_fn,
                candidate,
                search_n,
                3,
                len(search_state.codewords),
                popcount(candidate),
                new_forbidden,
                overlap_forbidden,
                local_available,
            )
            tie_break = _deterministic_tiebreak(candidate, restart_index + scan_index)
            pool.append((score, tie_break, candidate))

        if not pool:
            repaired = False
            if repair_mode == "mcts" and repair_moves < repair_events:
                for _drop_event in range(repair_drop_count):
                    if repair_moves >= repair_events:
                        break
                    drop_index, rollout_evaluations = _parity_choose_mcts_drop_index_with_evaluations(
                        search_state,
                        priority_fn,
                        restart_index,
                        repair_moves,
                        _sampled_seed(restart_index),
                        batch_size,
                        max(batch_size * 8, 128),
                        repair_tabu_set,
                    )
                    repair_rollout_evaluations += rollout_evaluations
                    if drop_index is None:
                        break
                    if repair_gain_start_size is None:
                        repair_gain_start_size = len(search_state.codewords)
                    search_state, dropped = _parity_apply_drop(search_state, drop_index)
                    _parity_add_tabu(
                        repair_tabu_queue,
                        repair_tabu_set,
                        dropped,
                        repair_tabu_tenure,
                    )
                    repair_moves += 1
                    repaired = True
            if repaired:
                continue
            break

        pool.sort(reverse=True)
        accepted_this_scan = 0
        for pool_rank, (score, _tie_break, candidate) in enumerate(pool, start=1):
            if accepted_this_scan >= batch_size:
                break
            if not search_state.can_add(candidate):
                continue
            search_state.add(candidate)
            score_by_center[candidate] = float(score)
            accepted_this_scan += 1
            extended_word = parity_extend_word(candidate, search_n)
            accepted_records.append(
                AcceptedWordRecord(
                    fill_index=len(search_state.codewords),
                    rank=pool_rank,
                    word=format_word(extended_word, instance.n),
                    weight=popcount(extended_word),
                    score=float(score),
                )
            )

        if accepted_this_scan == 0:
            repaired = False
            if repair_mode == "mcts" and repair_moves < repair_events:
                for _drop_event in range(repair_drop_count):
                    if repair_moves >= repair_events:
                        break
                    drop_index, rollout_evaluations = _parity_choose_mcts_drop_index_with_evaluations(
                        search_state,
                        priority_fn,
                        restart_index,
                        repair_moves,
                        _sampled_seed(restart_index),
                        batch_size,
                        max(batch_size * 8, 128),
                        repair_tabu_set,
                    )
                    repair_rollout_evaluations += rollout_evaluations
                    if drop_index is None:
                        break
                    if repair_gain_start_size is None:
                        repair_gain_start_size = len(search_state.codewords)
                    search_state, dropped = _parity_apply_drop(search_state, drop_index)
                    _parity_add_tabu(
                        repair_tabu_queue,
                        repair_tabu_set,
                        dropped,
                        repair_tabu_tenure,
                    )
                    repair_moves += 1
                    repaired = True
            if not repaired:
                break

    transformed_codewords = tuple(
        parity_extend_word(word_mask, search_n) for word_mask in search_state.codewords
    )
    accepted_records = list(
        _rebuild_parity_accepted_records(search_state.codewords, instance, score_by_center)
    )
    repair_gain = (
        max(0, len(search_state.codewords) - repair_gain_start_size)
        if repair_gain_start_size is not None
        else 0
    )
    return SearchResult(
        codewords=transformed_codewords,
        accepted_records=tuple(accepted_records),
        candidate_count=(1 << search_n) - 1,
        blocked_candidate_count=blocked_candidate_count,
        restart_index=restart_index,
        valid=True,
        minimum_distance=instance.distance,
        forbidden_count=len(search_state.forbidden_centers),
        repair_moves=repair_moves,
        repair_gain=repair_gain,
        repair_rollout_evaluations=repair_rollout_evaluations,
    )


def parity_transform_construct(
    instance: MaxCodeInstance,
    priority_fn: PriorityFn,
    restart_index: int,
) -> SearchResult:
    """Search A(n-1,3), then parity-extend the resulting centers to A(n,4)."""
    if instance.distance != 4:
        raise ValueError("Parity transform construction is only valid for target distance 4")

    search_n = instance.n - 1
    if search_n <= _parity_full_scan_n_limit():
        return parity_transform_full_scan_construct(instance, priority_fn, restart_index)

    search_state = ParityTransformState(search_n)
    search_state.add(0)
    rng = random.Random(_sampled_seed(restart_index))
    pool_size = _parity_pool_size(search_n)
    attempts_per_refill = _parity_attempts_per_refill(pool_size)
    max_refills = _parity_max_refills(search_n)
    max_stale_refills = _parity_max_stale_refills()
    accepted_records: list[AcceptedWordRecord] = []
    blocked_candidate_count = 0
    stale_refills = 0
    repair_moves = 0
    repair_gain_start_size: int | None = None
    repair_rollout_evaluations = 0
    repair_mode = _parity_repair_mode()
    repair_events = _parity_repair_events()
    repair_drop_count = _parity_repair_drop_count()
    repair_tabu_tenure = _parity_repair_tabu_tenure()
    repair_tabu_queue: list[int] = []
    repair_tabu_set: set[int] = set()
    score_by_center: dict[int, float] = {0: 0.0}

    def try_mcts_repair(repair_event_seed: int) -> bool:
        nonlocal search_state, repair_moves, repair_gain_start_size, repair_rollout_evaluations
        if repair_mode != "mcts" or repair_moves >= repair_events:
            return False
        repaired = False
        for _drop_event in range(repair_drop_count):
            if repair_moves >= repair_events:
                break
            drop_index, rollout_evaluations = _parity_choose_mcts_drop_index_with_evaluations(
                search_state,
                priority_fn,
                restart_index,
                repair_moves,
                repair_event_seed,
                pool_size,
                attempts_per_refill,
                repair_tabu_set,
            )
            repair_rollout_evaluations += rollout_evaluations
            if drop_index is None:
                break
            if repair_gain_start_size is None:
                repair_gain_start_size = len(search_state.codewords)
            search_state, dropped = _parity_apply_drop(search_state, drop_index)
            _parity_add_tabu(
                repair_tabu_queue,
                repair_tabu_set,
                dropped,
                repair_tabu_tenure,
            )
            repair_moves += 1
            repaired = True
        return repaired

    for refill_index in range(1, max_refills + 1):
        if len(search_state.forbidden_centers) >= search_state.universe_size:
            break

        pool: list[tuple[float, float, int, int, int, int]] = []
        local_seen: set[int] = set()
        attempts = 0
        while len(pool) < pool_size and attempts < attempts_per_refill:
            attempts += 1
            candidate = rng.randrange(1, search_state.universe_size)
            if candidate in local_seen or candidate in repair_tabu_set:
                continue
            local_seen.add(candidate)
            if not search_state.can_add(candidate):
                blocked_candidate_count += 1
                continue
            new_forbidden, overlap_forbidden, local_available = search_state.damage_features(
                candidate
            )
            score = _safe_priority(
                priority_fn,
                candidate,
                search_n,
                3,
                len(search_state.codewords),
                popcount(candidate),
                new_forbidden,
                overlap_forbidden,
                local_available,
            )
            pool.append(
                (
                    score,
                    rng.random(),
                    candidate,
                    new_forbidden,
                    overlap_forbidden,
                    local_available,
                )
            )

        if not pool:
            if try_mcts_repair(_sampled_seed(restart_index) + refill_index * 7919):
                stale_refills = 0
                continue
            stale_refills += 1
            if stale_refills >= max_stale_refills:
                break
            continue

        pool.sort(reverse=True)
        accepted_this_refill = 0
        for pool_rank, (
            score,
            _,
            candidate,
            _new_forbidden,
            _overlap_forbidden,
            _local_available,
        ) in enumerate(pool, start=1):
            if search_state.can_add(candidate):
                search_state.add(candidate)
                score_by_center[candidate] = float(score)
                accepted_this_refill += 1
                extended_word = parity_extend_word(candidate, search_n)
                accepted_records.append(
                    AcceptedWordRecord(
                        fill_index=len(search_state.codewords),
                        rank=pool_rank,
                        word=format_word(extended_word, instance.n),
                        weight=popcount(extended_word),
                        score=float(score),
                    )
                )

        if accepted_this_refill:
            stale_refills = 0
        else:
            if try_mcts_repair(_sampled_seed(restart_index) + refill_index * 104729):
                stale_refills = 0
                continue
            stale_refills += 1
            if stale_refills >= max_stale_refills:
                break

    if _env_flag_enabled("MAX_CODE_PARITY_FINAL_SWEEP", default=True):
        sweep_rank = 0
        while len(search_state.forbidden_centers) < search_state.universe_size:
            progress = 0
            for candidate in range(1, search_state.universe_size):
                if candidate in repair_tabu_set:
                    continue
                if not search_state.can_add(candidate):
                    continue
                sweep_rank += 1
                new_forbidden, overlap_forbidden, local_available = search_state.damage_features(
                    candidate
                )
                score = _safe_priority(
                    priority_fn,
                    candidate,
                    search_n,
                    3,
                    len(search_state.codewords),
                    popcount(candidate),
                    new_forbidden,
                    overlap_forbidden,
                    local_available,
                )
                search_state.add(candidate)
                score_by_center[candidate] = float(score)
                progress += 1
                extended_word = parity_extend_word(candidate, search_n)
                accepted_records.append(
                    AcceptedWordRecord(
                        fill_index=len(search_state.codewords),
                        rank=sweep_rank,
                        word=format_word(extended_word, instance.n),
                        weight=popcount(extended_word),
                        score=float(score),
                    )
                )
            if progress == 0:
                if try_mcts_repair(_sampled_seed(restart_index) + sweep_rank * 65537):
                    continue
                break

    transformed_codewords = tuple(
        parity_extend_word(word_mask, search_n) for word_mask in search_state.codewords
    )
    accepted_records = list(
        _rebuild_parity_accepted_records(search_state.codewords, instance, score_by_center)
    )
    repair_gain = (
        max(0, len(search_state.codewords) - repair_gain_start_size)
        if repair_gain_start_size is not None
        else 0
    )
    return SearchResult(
        codewords=transformed_codewords,
        accepted_records=tuple(accepted_records),
        candidate_count=(1 << search_n) - 1,
        blocked_candidate_count=blocked_candidate_count,
        restart_index=restart_index,
        valid=True,
        minimum_distance=instance.distance,
        forbidden_count=len(search_state.forbidden_centers),
        repair_moves=repair_moves,
        repair_gain=repair_gain,
        repair_rollout_evaluations=repair_rollout_evaluations,
    )


def hamming_distance(left: int, right: int) -> int:
    """Return Hamming distance between two binary words."""
    return popcount(left ^ right)


def actual_minimum_distance(codewords: Sequence[int]) -> int:
    """Return the exact pairwise minimum distance of a code."""
    if len(codewords) < 2:
        return math.inf
    best = math.inf
    for left_index, left in enumerate(codewords):
        for right in codewords[left_index + 1 :]:
            distance = hamming_distance(left, right)
            if distance < best:
                best = distance
    return int(best)


def validate_code(codewords: Sequence[int], distance: int) -> bool:
    """Validate a candidate binary code independently of the greedy state."""
    return actual_minimum_distance(codewords) >= distance


def load_priority_function(program_path: str) -> PriorityFn:
    """Load `priority(word_mask, n, d)` from an evolved Python file."""
    program_path_obj = Path(program_path).resolve()
    module_name = f"max_code_candidate_{program_path_obj.stem}_{id(program_path_obj)}"
    inserted_path = str(program_path_obj.parent)
    sys.path.insert(0, inserted_path)
    try:
        spec = importlib.util.spec_from_file_location(module_name, program_path_obj)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {program_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(inserted_path)
        except ValueError:
            pass
    priority_fn = getattr(module, "priority", None)
    if priority_fn is None or not callable(priority_fn):
        raise AttributeError("Program must define callable priority(word_mask, n, d)")
    for optional_name in ("destroy_priority", "repair_priority"):
        optional_fn = getattr(module, optional_name, None)
        if callable(optional_fn):
            setattr(priority_fn, optional_name, optional_fn)
    return priority_fn


def _codeword_strings(codewords: Iterable[int], n: int) -> list[str]:
    return [format_word(word, n) for word in codewords]


def evaluate_priority_function(
    priority_fn: PriorityFn,
    instance: MaxCodeInstance,
) -> EvaluationResult:
    """Evaluate a priority function on one maximum-code instance."""
    if instance.distance == 4:
        results = [
            parity_transform_construct(instance, priority_fn, restart_index)
            for restart_index in range(instance.restarts)
        ]
        search_mode = (
            "parity_transform_dynamic_mcts_repair"
            if _parity_repair_mode() == "mcts"
            else "parity_transform_dynamic"
        )
    else:
        static_scores = score_static_candidates(instance, priority_fn)
        if _parity_repair_mode() == "mcts":
            results = [
                dynamic_mcts_construct(
                    instance,
                    static_scores,
                    restart_index,
                    priority_fn,
                )
                for restart_index in range(instance.restarts)
            ]
            search_mode = "dynamic_mcts_repair"
        else:
            results = [
                greedy_construct(instance, static_scores, restart_index, priority_fn=priority_fn)
                for restart_index in range(instance.restarts)
            ]
            search_mode = "static_full_greedy"
    best = max(
        results,
        key=lambda result: (
            int(result.valid),
            len(result.codewords),
            result.minimum_distance,
            -result.blocked_candidate_count,
        ),
    )
    code_size = len(best.codewords) if best.valid else 0
    combined_score = float(code_size)

    metrics = {
        "combined_score": combined_score,
        "code_size": float(code_size),
        "valid": 1.0 if best.valid else 0.0,
        "minimum_distance": float(best.minimum_distance),
        "target_distance": float(instance.distance),
        "n": float(instance.n),
    }
    artifacts = {
        "codewords": json.dumps(_codeword_strings(best.codewords, instance.n)),
        "search_result": json.dumps(
            {
                "instance": {
                    "name": instance.name,
                    "n": instance.n,
                    "d": instance.distance,
                    "restarts": instance.restarts,
                },
                "search_mode": search_mode,
                "code_size": code_size,
                "valid": best.valid,
                "minimum_distance": best.minimum_distance,
                "restart_index": best.restart_index,
                "candidate_count": best.candidate_count,
                "forbidden_count": best.forbidden_count,
                "repair_mode": _parity_repair_mode() if instance.distance == 4 else (
                    "mcts" if search_mode == "dynamic_mcts_repair" else "local"
                ),
                "repair_mcts_simulations": (
                    _parity_repair_mcts_simulations()
                    if instance.distance == 4 or search_mode == "dynamic_mcts_repair"
                    else 0
                ),
                "repair_mcts_depth": (
                    _parity_repair_mcts_depth()
                    if instance.distance == 4 or search_mode == "dynamic_mcts_repair"
                    else 0
                ),
                "repair_mcts_drop_topk": (
                    _parity_repair_mcts_drop_topk()
                    if instance.distance == 4 or search_mode == "dynamic_mcts_repair"
                    else 0
                ),
                "repair_moves": best.repair_moves,
                "repair_gain": best.repair_gain,
                "repair_rollout_evaluations": best.repair_rollout_evaluations,
                "accepted_words": [record.__dict__ for record in best.accepted_records],
            },
            indent=2,
        ),
    }
    return EvaluationResult(metrics=metrics, artifacts=artifacts)


def _error_result(error: Exception) -> EvaluationResult:
    """Return a structured zero-score result for evaluator failures."""
    return EvaluationResult(
        metrics={
            "combined_score": 0.0,
            "code_size": 0.0,
            "valid": 0.0,
            "minimum_distance": 0.0,
            "target_distance": 0.0,
            "n": 0.0,
            "error": 1.0,
        },
        artifacts={
            "error_type": type(error).__name__,
            "error_message": str(error),
        },
    )


def evaluate_program_path(program_path: str) -> EvaluationResult:
    """OpenEvolve adapter for evaluating an evolved priority program."""
    try:
        priority_fn = load_priority_function(program_path)
        return evaluate_priority_function(priority_fn, instance_from_env())
    except Exception as error:
        return _error_result(error)
