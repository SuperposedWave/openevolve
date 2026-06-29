#!/usr/bin/env python3
"""Benchmark initial C priority evaluation speed.

This script calls the same evaluator entry point OpenEvolve uses, but without
running any evolution iterations. The C path includes per-evaluation compilation
because evolved C variants are compiled by the evaluator before execution.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluator import evaluate  # noqa: E402


DEFAULT_INSTANCES = ((8, 4, 4), (20, 10, 5), (38, 23, 7))
PROGRAMS = {"c": SCRIPT_DIR / "initial_program.c"}


@contextmanager
def patched_env(updates: dict[str, str]):
    previous: dict[str, str | None] = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def parse_instance(raw: str) -> tuple[int, int, int]:
    parts = raw.replace(",", " ").split()
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("instance must be formatted as n,k,d")
    try:
        n, k, d = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("instance values must be integers") from exc
    if n <= 0 or k <= 0 or d <= 0 or k >= n:
        raise argparse.ArgumentTypeError("instance requires n > k > 0 and d > 0")
    return n, k, d


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def median(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.median(values) if values else 0.0


def run_one(
    *,
    mode: str,
    n: int,
    k: int,
    d: int,
    restarts: int,
    seed: int,
) -> dict[str, object]:
    env = {
        "LINEAR_CODE_N": str(n),
        "LINEAR_CODE_K": str(k),
        "LINEAR_CODE_D": str(d),
        "LINEAR_CODE_RESTARTS": str(restarts),
        "LINEAR_CODE_RANDOM_SEED": str(seed),
        "LINEAR_CODE_PROGRESS": "0",
    }
    program_path = PROGRAMS[mode]
    started_at = time.perf_counter()
    with patched_env(env):
        result = evaluate(str(program_path))
    wall_seconds = time.perf_counter() - started_at
    search_result = json.loads(result.artifacts.get("search_result", "{}"))
    metrics = result.metrics
    return {
        "mode": mode,
        "n": n,
        "k": k,
        "d": d,
        "r": n - k,
        "restarts": restarts,
        "seed": seed,
        "success": metrics.get("success_rate", 0.0),
        "constructed_columns": metrics.get("constructed_columns", 0.0),
        "candidate_count": search_result.get("candidate_count", 0),
        "scored_candidates": metrics.get("scored_candidates", 0.0),
        "evaluator_seconds": metrics.get("evaluation_time_seconds", 0.0),
        "wall_seconds": wall_seconds,
    }


def print_summary(rows: list[dict[str, object]]) -> None:
    groups: dict[tuple[int, int, int, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (int(row["n"]), int(row["k"]), int(row["d"]), str(row["mode"]))
        groups.setdefault(key, []).append(row)

    print("\nsummary")
    print(
        "n,k,d,mode,runs,success_rate,constructed,median_eval_s,mean_eval_s,"
        "median_wall_s,mean_wall_s,candidate_count"
    )
    for key in sorted(groups):
        n, k, d, mode = key
        group = groups[key]
        eval_times = [float(row["evaluator_seconds"]) for row in group]
        wall_times = [float(row["wall_seconds"]) for row in group]
        print(
            f"{n},{k},{d},{mode},{len(group)},"
            f"{mean(float(row['success']) for row in group):.3f},"
            f"{mean(float(row['constructed_columns']) for row in group):.2f},"
            f"{median(eval_times):.6f},{mean(eval_times):.6f},"
            f"{median(wall_times):.6f},{mean(wall_times):.6f},"
            f"{int(group[-1]['candidate_count'])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark initial_program.c evaluation speed."
    )
    parser.add_argument(
        "--instance",
        action="append",
        type=parse_instance,
        help="Instance formatted as n,k,d. May be repeated. Defaults to a small sweep.",
    )
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=("c",),
        action="append",
        help="Mode to run. May be repeated. Defaults to c.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional path to write per-run CSV rows.",
    )
    args = parser.parse_args()

    instances = args.instance or list(DEFAULT_INSTANCES)
    modes = args.mode or ["c"]
    rows: list[dict[str, object]] = []

    for n, k, d in instances:
        for mode in modes:
            for warmup_index in range(args.warmups):
                run_one(
                    mode=mode,
                    n=n,
                    k=k,
                    d=d,
                    restarts=args.restarts,
                    seed=args.seed + warmup_index,
                )
            for repeat_index in range(args.repeats):
                row = run_one(
                    mode=mode,
                    n=n,
                    k=k,
                    d=d,
                    restarts=args.restarts,
                    seed=args.seed + repeat_index,
                )
                rows.append(row)
                print(
                    f"run mode={mode} n={n} k={k} d={d} repeat={repeat_index + 1} "
                    f"eval_s={float(row['evaluator_seconds']):.6f} "
                    f"wall_s={float(row['wall_seconds']):.6f} "
                    f"constructed={row['constructed_columns']} "
                    f"success={row['success']}"
                )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")

    print_summary(rows)


if __name__ == "__main__":
    main()
