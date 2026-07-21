"""Baseline priority for generator-matrix binary linear-code construction."""


# EVOLVE-BLOCK-START
def _deterministic_mix(column_mask):
    value = column_mask & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 33
    value = (value * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 33
    value = (value * 0xC4CEB9FE1A85EC53) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 33
    return value / float(1 << 64)


def priority(
    column_mask,
    n,
    k,
    d,
    step=0,
    column_weight=0,
    covered_deficit_sum=0,
    covered_critical_count=0,
    uncovered_critical_count=0,
    feasible_next_count=0,
    min_row_weight_after=0,
    max_row_weight_after=0,
    avg_pair_balance_after=0.0,
):
    """
    Score one k-bit parity column p_j for G=[I_k|P].

    The fixed evaluator computes exact coverage features from the current
    message deficits. This function only ranks candidates.
    """
    weight = int(column_weight) if column_weight else int(column_mask).bit_count()
    target_weight = max(1.0, 0.5 * float(k))
    weight_balance = -abs(float(weight) - target_weight)
    row_spread = float(max_row_weight_after - min_row_weight_after)
    fill_phase = float(step) / max(1.0, float(n - k))
    future_space = float(feasible_next_count) / (float(feasible_next_count) + 1024.0)
    novelty = _deterministic_mix(int(column_mask))

    return (
        1.0 * float(covered_deficit_sum)
        + 4.0 * float(covered_critical_count)
        - 1000.0 * float(uncovered_critical_count)
        + 2.0 * future_space
        + 0.6 * weight_balance
        - 0.4 * row_spread
        - 0.2 * float(avg_pair_balance_after)
        - 0.05 * fill_phase * abs(float(weight) - target_weight)
        + 0.001 * novelty
    )


# EVOLVE-BLOCK-END


from search_core import DEFAULT_INSTANCE, evaluate_priority_function, instance_from_env


def get_priority_function():
    """Stable entry point for evaluator and tests."""
    return priority


def run_baseline_suite():
    """Evaluate this priority on the configured instance."""
    return evaluate_priority_function(priority, instance_from_env())


if __name__ == "__main__":
    configured_instance = instance_from_env()
    result = run_baseline_suite()
    print("Default instance:", DEFAULT_INSTANCE)
    print("Configured instance:", configured_instance)
    print("Metrics:", result.metrics)
    print("Artifacts:", result.artifacts)
