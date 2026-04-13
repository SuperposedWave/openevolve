"""Baseline priority heuristic for binary linear-code feasibility search."""


# EVOLVE-BLOCK-START
def _target_weight(state):
    """Prefer medium-to-high weight columns rather than extreme supports."""
    return max(state["D"] - 1, state["r"] // 2 + 1)


def _coordinate_balance_score(candidate_mask, coordinate_usage):
    """Reward columns that activate coordinates that have appeared less often."""
    score = 0.0
    for bit_index, usage in enumerate(coordinate_usage):
        if candidate_mask & (1 << bit_index):
            score += 1.0 / (1.0 + usage)
    return score


def _pairwise_separation_score(candidate_mask, selected_columns):
    """Prefer candidates that stay far from already chosen free columns."""
    if not selected_columns:
        return 0.0
    xor_weights = [(candidate_mask ^ other_mask).bit_count() for other_mask in selected_columns]
    overlaps = [(candidate_mask & other_mask).bit_count() for other_mask in selected_columns]
    return min(xor_weights) - 0.35 * max(overlaps)


def priority(candidate_mask, state):
    """
    Score a legal free column for the fixed greedy skeleton.

    The evaluator keeps legality exact; this function only ranks legal choices.
    """
    weight = candidate_mask.bit_count()
    selected_columns = state["selected_free_columns"]
    coordinate_usage = state["coordinate_usage"]
    target_weight = _target_weight(state)

    weight_score = -abs(weight - target_weight)
    balance_score = _coordinate_balance_score(candidate_mask, coordinate_usage)
    separation_score = _pairwise_separation_score(candidate_mask, selected_columns)
    remaining_slots_bonus = 0.15 * state["remaining_slots"]

    return 2.0 * weight_score + 1.75 * balance_score + separation_score + remaining_slots_bonus


# EVOLVE-BLOCK-END

from search_core import BENCHMARKS, evaluate_priority_function


def get_priority_function():
    """Stable entry point for tests and manual inspection."""
    return priority


def run_baseline_suite():
    """Evaluate the current priority function on the fixed benchmark set."""
    return evaluate_priority_function(priority, BENCHMARKS)


if __name__ == "__main__":
    result = run_baseline_suite()
    print("Metrics:", result.metrics)
    print("Artifacts:", result.artifacts)
