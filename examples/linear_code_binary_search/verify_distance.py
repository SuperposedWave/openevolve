"""Verify the actual minimum distance reached by a priority program."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from search_core import (
    actual_minimum_distance,
    best_restart_for_instance,
    format_mask,
    generator_matrix_rows,
    instance_from_env,
    make_instance,
    load_priority_function,
    parity_check_matrix_rows,
)


def _configure_console_logging() -> None:
    """Ensure INFO-level logs are visible when this script runs standalone."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(logging.INFO)
        return
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )


def main() -> None:
    _configure_console_logging()
    parser = argparse.ArgumentParser(
        description="Construct a GF(3) H from a priority program and report the actual minimum distance."
    )
    parser.add_argument(
        "program_path",
        nargs="?",
        default="initial_program.py",
        help="Path to a Python file that defines priority(column_code, n, k, d).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )
    parser.add_argument("--N", type=int, dest="n", help="Code length n.")
    parser.add_argument("--K", type=int, dest="k", help="Code dimension k.")
    parser.add_argument("--D", type=int, dest="d", help="Target minimum distance d.")
    parser.add_argument("--restarts", type=int, help="Number of randomized restarts.")
    args = parser.parse_args()

    program_path = Path(args.program_path).resolve()
    priority_fn = load_priority_function(str(program_path))
    env_instance = instance_from_env()
    instance = make_instance(
        n=args.n if args.n is not None else env_instance.n,
        k=args.k if args.k is not None else env_instance.k,
        distance=args.d if args.d is not None else env_instance.target_distance,
        restarts=args.restarts if args.restarts is not None else env_instance.restarts,
    )
    attempt = best_restart_for_instance(
        instance,
        priority_fn,
        show_progress=not args.no_progress,
    )
    d_actual = actual_minimum_distance(instance.r, attempt.selected_free_columns)
    matrix_rows = parity_check_matrix_rows(instance.r, attempt.selected_free_columns)
    generator_rows = generator_matrix_rows(instance.r, attempt.selected_free_columns)
    is_complete = attempt.added_free_columns == instance.k

    print(f"program_path: {program_path}")
    print(
        "instance:",
        json.dumps(
            {
                "n": instance.n,
                "k": instance.k,
                "d_target": instance.target_distance,
                "r": instance.r,
                "restarts": instance.restarts,
            },
            sort_keys=True,
        ),
    )
    print(
        "construction:",
        json.dumps(
            {
                "success": attempt.success,
                "added_free_columns": attempt.added_free_columns,
                "remaining_free_columns": instance.k - attempt.added_free_columns,
                "selected_free_columns": [
                    format_mask(column_mask, instance.r)
                    for column_mask in attempt.selected_free_columns
                ],
            },
            sort_keys=True,
        ),
    )
    if is_complete:
        print(f"d_actual: {d_actual}")
    else:
        print(f"d_partial: {d_actual}")
        print("warning: construction is incomplete, so this distance only applies to the partial column set")

    print(f"H shape: {instance.r} x {attempt.added_free_columns + instance.r} over GF(3)")
    if is_complete:
        print("H rows:")
    else:
        print("Partial H rows:")
    for row in matrix_rows:
        print(row)

    if is_complete:
        print(f"G shape: {instance.k} x {instance.n} over GF(3)")
        print("G rows:")
    else:
        print(
            "Partial G shape: "
            f"{attempt.added_free_columns} x {attempt.added_free_columns + instance.r} over GF(3)"
        )
        print("Partial G rows:")
    for row in generator_rows:
        print(row)


if __name__ == "__main__":
    main()
