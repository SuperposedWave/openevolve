"""Verify the actual minimum distance reached by a C priority program."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path

from evaluator import evaluate
from search_core import actual_minimum_distance, instance_from_env, make_instance


@contextmanager
def patched_env(updates: dict[str, str]):
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the C kernel and report the actual minimum distance."
    )
    parser.add_argument(
        "program_path",
        nargs="?",
        default="initial_program.c",
        help="Path to a C file that defines oe_linear_code_priority().",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Accepted for compatibility; the C verifier does not show progress bars.",
    )
    parser.add_argument("--N", type=int, dest="n", help="Code length n.")
    parser.add_argument("--K", type=int, dest="k", help="Code dimension k.")
    parser.add_argument("--D", type=int, dest="d", help="Target minimum distance d.")
    parser.add_argument("--restarts", type=int, help="Number of C-kernel restarts.")
    args = parser.parse_args()

    program_path = Path(args.program_path).resolve()
    env_instance = instance_from_env()
    instance = make_instance(
        n=args.n if args.n is not None else env_instance.n,
        k=args.k if args.k is not None else env_instance.k,
        distance=args.d if args.d is not None else env_instance.target_distance,
        restarts=args.restarts if args.restarts is not None else env_instance.restarts,
    )
    env = {
        "LINEAR_CODE_N": str(instance.n),
        "LINEAR_CODE_K": str(instance.k),
        "LINEAR_CODE_D": str(instance.target_distance),
        "LINEAR_CODE_RESTARTS": str(instance.restarts),
    }

    with patched_env(env):
        result = evaluate(str(program_path))

    search_result = json.loads(result.artifacts.get("search_result", "{}"))
    selected_bits = search_result.get("selected_free_columns", [])
    selected = tuple(int(bits, 2) for bits in selected_bits)
    added_free_columns = int(search_result.get("added_free_columns", len(selected)))
    d_actual = actual_minimum_distance(instance.r, selected)
    matrix_rows = json.loads(result.artifacts.get("parity_check_matrix", "[]"))
    generator_rows = json.loads(result.artifacts.get("generator_matrix", "[]"))
    is_complete = added_free_columns == instance.k

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
                "success": bool(search_result.get("success", False)),
                "search_mode": search_result.get("search_mode", "c_kernel"),
                "added_free_columns": added_free_columns,
                "remaining_free_columns": instance.k - added_free_columns,
                "selected_free_columns": selected_bits,
            },
            sort_keys=True,
        ),
    )
    if is_complete:
        print(f"d_actual: {d_actual}")
    else:
        print(f"d_partial: {d_actual}")
        print("warning: construction is incomplete, so this distance only applies to the partial column set")

    print(f"H shape: {instance.r} x {added_free_columns + instance.r}")
    print("H rows:" if is_complete else "Partial H rows:")
    for row in matrix_rows:
        print(row)

    if is_complete:
        print(f"G shape: {instance.k} x {instance.n}")
        print("G rows:")
    else:
        print(f"Partial G shape: {added_free_columns} x {added_free_columns + instance.r}")
        print("Partial G rows:")
    for row in generator_rows:
        print(row)


if __name__ == "__main__":
    main()
