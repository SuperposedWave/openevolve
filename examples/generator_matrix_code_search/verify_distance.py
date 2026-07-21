"""Run one generator-matrix priority program and print matrix verification."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path

from evaluator import evaluate
from search_core import instance_from_env, make_instance


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
        description="Run generator-matrix search and report actual minimum distance."
    )
    parser.add_argument(
        "program_path",
        nargs="?",
        default="initial_program.py",
        help="Path to a Python file defining priority().",
    )
    parser.add_argument("--N", type=int, dest="n", help="Code length n.")
    parser.add_argument("--K", type=int, dest="k", help="Code dimension k.")
    parser.add_argument("--D", type=int, dest="d", help="Target minimum distance d.")
    parser.add_argument("--restarts", type=int, help="Number of deterministic restarts.")
    args = parser.parse_args()

    program_path = Path(args.program_path).resolve()
    env_instance = instance_from_env()
    instance = make_instance(
        n=args.n if args.n is not None else env_instance.n,
        k=args.k if args.k is not None else env_instance.k,
        d=args.d if args.d is not None else env_instance.d,
    )
    env = {
        "GEN_MATRIX_CODE_N": str(instance.n),
        "GEN_MATRIX_CODE_K": str(instance.k),
        "GEN_MATRIX_CODE_D": str(instance.d),
    }
    if args.restarts is not None:
        env["GEN_MATRIX_CODE_RESTARTS"] = str(args.restarts)

    with patched_env(env):
        result = evaluate(str(program_path))

    search_result = json.loads(result.artifacts.get("search_result", "{}"))
    matrix_summary = json.loads(result.artifacts.get("matrix_summary", "{}"))
    generator_rows = json.loads(result.artifacts.get("generator_matrix", "[]"))
    parity_rows = json.loads(result.artifacts.get("parity_check_matrix", "[]"))

    print(f"program_path: {program_path}")
    print(
        "instance:",
        json.dumps(
            {
                "n": instance.n,
                "k": instance.k,
                "d_target": instance.d,
                "r": instance.r,
            },
            sort_keys=True,
        ),
    )
    print("metrics:", json.dumps(result.metrics, sort_keys=True))
    print(
        "construction:",
        json.dumps(
            {
                "success": bool(search_result.get("success", False)),
                "columns": len(search_result.get("columns", [])),
                "target_columns": instance.r,
                "unsatisfied_messages": matrix_summary.get("unsatisfied_messages"),
                "remaining_deficit_sum": matrix_summary.get("remaining_deficit_sum"),
            },
            sort_keys=True,
        ),
    )
    d_actual = matrix_summary.get("d_actual")
    if search_result.get("success"):
        print(f"d_actual: {d_actual}")
    else:
        print(f"d_partial: {d_actual}")
        print("warning: construction is incomplete or below target distance")

    print(f"G shape: {instance.k} x {instance.n}")
    print("G rows:")
    for row in generator_rows:
        print(row)

    print(f"H shape: {instance.r} x {instance.n}")
    print("H rows:")
    for row in parity_rows:
        print(row)


if __name__ == "__main__":
    main()
