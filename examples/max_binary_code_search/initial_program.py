"""Baseline static priority heuristic for binary maximum-code construction."""


# EVOLVE-BLOCK-START
def _target_weight(n, d):
    if d <= 2:
        return n // 2
    return max(d, n // 2)


def _run_count(word_mask, n):
    bits = format(word_mask, f"0{n}b")
    runs = 1
    for i in range(1, n):
        if bits[i] != bits[i - 1]:
            runs += 1
    return runs


def _balance_score(word_mask, n):
    half = n // 2
    left = 0
    right = 0
    for bit_index in range(half):
        if word_mask & (1 << bit_index):
            left += 1
        if word_mask & (1 << (n - 1 - bit_index)):
            right += 1
    return -abs(left - right)


def _deterministic_mix(word_mask):
    value = word_mask & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 33
    value = (value * 0xff51afd7ed558ccd) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 33
    value = (value * 0xc4ceb9fe1a85ec53) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 33
    return value / float(1 << 64)


def priority(
    word_mask,
    n,
    d,
    step=0,
    word_weight=None,
    new_forbidden_count=0,
    overlap_forbidden_count=0,
    local_available_count=0,
):
    """
    Score one candidate binary word for a greedy maximum-code construction.

    The evaluator enforces minimum distance exactly. This function only defines
    a deterministic ranking over candidate words. In the d=4 parity-transform
    search, the evaluator also passes dynamic damage features for the current
    partial code.
    """
    weight = word_mask.bit_count() if word_weight is None else word_weight
    target = _target_weight(n, d)

    weight_score = -abs(weight - target)
    run_score = _run_count(word_mask, n)
    balance_score = _balance_score(word_mask, n)
    endpoint_bonus = 0.25 * ((word_mask & 1) != 0) + 0.25 * ((word_mask >> (n - 1)) & 1)
    fine_tie_break = _deterministic_mix(word_mask)
    damage_total = max(1, new_forbidden_count + overlap_forbidden_count)
    damage_score = -float(new_forbidden_count) / damage_total
    overlap_score = float(overlap_forbidden_count) / damage_total
    local_score = float(local_available_count) / max(1, n * n)
    phase = step / max(1, 1 << min(n, 12))

    return (
        7.0 * damage_score
        + 1.5 * overlap_score
        + 0.8 * local_score
        + 2.0 * weight_score
        + 0.25 * run_score
        + 0.5 * balance_score
        + endpoint_bonus
        - 0.05 * phase * abs(weight - n // 2)
        + 0.01 * fine_tie_break
    )


def destroy_priority(
    word_mask,
    n,
    d,
    blocker_count=0,
    pair_blocker_count=0,
    code_size=0,
    word_weight=None,
    candidate_score=0.0,
    min_neighbor_distance=0,
    avg_neighbor_distance=0.0,
):
    """Score selected codewords for local removal during repair."""
    weight = word_mask.bit_count() if word_weight is None else word_weight
    target = _target_weight(n, d)
    crowding = max(0, d + 2 - min_neighbor_distance)
    low_weight_penalty = -0.1 * abs(weight - target)
    return (
        4.0 * float(blocker_count)
        + 1.0 * float(pair_blocker_count)
        + 0.5 * float(crowding)
        + low_weight_penalty
        - 0.05 * float(candidate_score)
        - 0.01 * float(avg_neighbor_distance)
    )


def repair_priority(
    word_mask,
    n,
    d,
    removed_count=0,
    blocker_count=0,
    base_size=0,
    word_weight=None,
    candidate_score=0.0,
):
    """Score local refill candidates after one or two codewords are removed."""
    weight = word_mask.bit_count() if word_weight is None else word_weight
    target = _target_weight(n, d)
    balance = -abs(weight - target)
    novelty = _deterministic_mix(word_mask)
    return (
        float(candidate_score)
        + 0.6 * float(blocker_count)
        + 0.25 * float(removed_count)
        + 0.3 * balance
        + 0.01 * novelty
    )


priority.destroy_priority = destroy_priority
priority.repair_priority = repair_priority


# EVOLVE-BLOCK-END

from search_core import DEFAULT_INSTANCE, evaluate_priority_function, instance_from_env


def get_priority_function():
    """Stable entry point for tests and manual inspection."""
    return priority


def run_baseline_suite():
    """Evaluate the current priority function on one configured instance."""
    return evaluate_priority_function(priority, instance_from_env())


if __name__ == "__main__":
    configured_instance = instance_from_env()
    result = run_baseline_suite()
    print("Default instance:", DEFAULT_INSTANCE)
    print("Configured instance:", configured_instance)
    print("Metrics:", result.metrics)
    print("Artifacts:", result.artifacts)
