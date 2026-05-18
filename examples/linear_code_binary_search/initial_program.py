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

import json
import os
from pathlib import Path

from search_core import (
    DEFAULT_INSTANCE,
    actual_minimum_distance,
    evaluate_priority_function,
    generator_matrix_rows,
    instance_from_env,
    parity_check_matrix_rows,
)


def get_priority_function():
    """Stable entry point for tests and manual inspection."""
    return priority


def run_baseline_suite():
    """Evaluate the current priority function on one configured instance."""
    return evaluate_priority_function(priority, instance_from_env())


def _matrix_output_path():
    raw_path = os.environ.get("LINEAR_CODE_MATRIX_OUTPUT")
    if raw_path:
        return Path(raw_path)
    return Path("matrix_verification.txt")


def _matrix_exact_distance_limit():
    raw_limit = os.environ.get("LINEAR_CODE_MATRIX_MAX_EXHAUSTIVE_K")
    if raw_limit is None:
        return 24
    try:
        return max(int(raw_limit), 0)
    except ValueError:
        return 24


def save_matrix_report(result, instance):
    """Save the constructed parity-check and generator matrices for a single run."""
    search_result = json.loads(result.artifacts["search_result"])
    selected_free_columns = tuple(
        int(bits, 2) for bits in search_result["selected_free_columns"]
    )
    is_complete = search_result["added_free_columns"] == instance.k
    matrix_rows = parity_check_matrix_rows(instance.r, selected_free_columns)
    generator_rows = generator_matrix_rows(instance.r, selected_free_columns)
    exact_distance_limit = _matrix_exact_distance_limit()
    output_path = _matrix_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"program_path: {Path(__file__).resolve()}",
        "instance: "
        + json.dumps(
            {
                "n": instance.n,
                "k": instance.k,
                "d_target": instance.target_distance,
                "r": instance.r,
                "restarts": instance.restarts,
            },
            sort_keys=True,
        ),
        "construction: "
        + json.dumps(
            {
                "success": search_result["success"],
                "added_free_columns": search_result["added_free_columns"],
                "remaining_free_columns": instance.k - search_result["added_free_columns"],
                "selected_free_columns": search_result["selected_free_columns"],
            },
            sort_keys=True,
        ),
    ]

    if search_result["added_free_columns"] <= exact_distance_limit:
        d_actual = actual_minimum_distance(instance.r, selected_free_columns)
        lines.append(f"d_actual: {d_actual}" if is_complete else f"d_partial: {d_actual}")
    else:
        lines.append(
            "d_actual: skipped"
            if is_complete
            else "d_partial: skipped"
        )
        lines.append(
            "warning: exhaustive distance check skipped because "
            f"added_free_columns={search_result['added_free_columns']} exceeds "
            f"LINEAR_CODE_MATRIX_MAX_EXHAUSTIVE_K={exact_distance_limit}"
        )
    if not is_complete:
        lines.append(
            "warning: construction is incomplete, so this distance only applies to the partial column set"
        )

    lines.append(f"H shape: {instance.r} x {search_result['added_free_columns'] + instance.r}")
    lines.append("H rows:" if is_complete else "Partial H rows:")
    lines.extend(matrix_rows)

    if is_complete:
        lines.append(f"G shape: {instance.k} x {instance.n}")
        lines.append("G rows:")
    else:
        lines.append(
            f"Partial G shape: {search_result['added_free_columns']} x "
            f"{search_result['added_free_columns'] + instance.r}"
        )
        lines.append("Partial G rows:")
    lines.extend(generator_rows)

    output_path.write_text("\n".join(lines) + "\n")
    return output_path


if __name__ == "__main__":
    configured_instance = instance_from_env()
    result = run_baseline_suite()
    print("Default instance:", DEFAULT_INSTANCE)
    print("Configured instance:", configured_instance)
    print("Metrics:", result.metrics)
    print("Artifacts:", result.artifacts)
    matrix_output = save_matrix_report(result, configured_instance)
    print("Matrix report:", matrix_output)
