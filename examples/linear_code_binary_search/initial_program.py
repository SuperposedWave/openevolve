"""Baseline static priority heuristic for FunSearch-style matrix construction."""


# EVOLVE-BLOCK-START
def _row_count(n, k):
    return n - k


def _target_weight(n, k, d):
    r = _row_count(n, k)
    return max(d - 1, r // 2 + 1)


def _center_symmetry_score(column_mask, r):
    left = 0
    right = 0
    for i in range(r // 2):
        if column_mask & (1 << i):
            left += 1
        if column_mask & (1 << (r - 1 - i)):
            right += 1
    return -abs(left - right)


def _run_count_score(column_mask, r):
    bits = format(column_mask, f"0{r}b")
    runs = 1
    for i in range(1, len(bits)):
        if bits[i] != bits[i - 1]:
            runs += 1
    return runs


def priority(column_mask, n, k, d):
    """
    Score a candidate free column using only static instance information.

    The evaluator keeps legality exact; this function only defines a global ranking.
    """
    r = _row_count(n, k)
    weight = column_mask.bit_count()
    target_weight = _target_weight(n, k, d)

    weight_score = -abs(weight - target_weight)
    symmetry_score = _center_symmetry_score(column_mask, r)
    run_score = _run_count_score(column_mask, r)
    endpoint_bonus = 0.5 * ((column_mask & 1) != 0) + 0.5 * ((column_mask >> (r - 1)) & 1)

    return 2.5 * weight_score + 0.8 * symmetry_score + 0.35 * run_score + endpoint_bonus


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
