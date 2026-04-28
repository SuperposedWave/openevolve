"""Baseline static priority heuristic for FunSearch-style ternary matrix construction."""


# EVOLVE-BLOCK-START
def _row_count(n, k):
    return n - k


def _target_weight(n, k, d):
    r = _row_count(n, k)
    return max(d - 1, r // 2 + 1)


def _decode_column(column_code, r):
    digits = []
    value = column_code
    for _ in range(r):
        digits.append(value % 3)
        value //= 3
    return digits


def _support_weight(digits):
    return sum(1 for digit in digits if digit)


def _center_symmetry_score(digits):
    r = len(digits)
    left = 0
    right = 0
    for i in range(r // 2):
        if digits[i]:
            left += 1
        if digits[r - 1 - i]:
            right += 1
    return -abs(left - right)


def _run_count_score(digits):
    bits = "".join(str(digit) for digit in digits)
    runs = 1
    for i in range(1, len(bits)):
        if bits[i] != bits[i - 1]:
            runs += 1
    return runs


def _symbol_balance_score(digits):
    ones = digits.count(1)
    twos = digits.count(2)
    return -abs(ones - twos)


def _nonzero_span_score(digits):
    first = -1
    last = -1
    for index, digit in enumerate(digits):
        if digit:
            if first < 0:
                first = index
            last = index
    if first < 0:
        return 0
    return last - first


def priority(column_code, n, k, d):
    """
    Score a candidate free column using only static instance information.

    The evaluator keeps legality exact; this function only defines a global ranking.
    """
    r = _row_count(n, k)
    digits = _decode_column(column_code, r)
    weight = _support_weight(digits)
    target_weight = _target_weight(n, k, d)

    weight_score = -abs(weight - target_weight)
    symmetry_score = _center_symmetry_score(digits)
    run_score = _run_count_score(digits)
    balance_score = _symbol_balance_score(digits)
    span_score = _nonzero_span_score(digits)
    endpoint_bonus = 0.5 * (digits[0] != 0) + 0.5 * (digits[-1] != 0)

    return (
        2.5 * weight_score
        + 0.8 * symmetry_score
        + 0.4 * balance_score
        + 0.35 * run_score
        + 0.15 * span_score
        + endpoint_bonus
    )


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
