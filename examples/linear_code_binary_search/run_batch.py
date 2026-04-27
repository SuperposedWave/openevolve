#!/usr/bin/env python3
"""Batch-run OpenEvolve for binary linear-code instances from ECCRecord.json."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_RECORD = SCRIPT_DIR / "Misc" / "ECCRecord.json"
DEFAULT_CONFIG = SCRIPT_DIR / "config.yaml"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "batch_runs"
TARGET_BLOCK_RE = re.compile(
    r"(?m)^    Current target:\n(?:^    - .*\n){4}"
)


@dataclass(frozen=True)
class SweepTask:
    """One `(n, k, d)` instance to run."""

    n: int
    k: int
    d: int
    lower: int
    upper: int

    @property
    def r(self) -> int:
        return self.n - self.k

    @property
    def instance_name(self) -> str:
        return f"n{self.n}_k{self.k}_d{self.d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-run OpenEvolve for all binary matrix instances with n <= n_max."
    )
    parser.add_argument(
        "--record",
        default=str(DEFAULT_RECORD),
        help="Path to ECCRecord.json.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Base config template used to generate per-instance configs.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory where per-instance outputs are written.",
    )
    parser.add_argument(
        "--n-min",
        type=int,
        default=11,
        help="Only run tasks with n >= n_min.",
    )
    parser.add_argument(
        "--n-max",
        type=int,
        default=40,
        help="Only run tasks with n <= n_max.",
    )
    parser.add_argument(
        "--d-field",
        choices=("lower", "upper"),
        default="lower",
        help="Which field from ECCRecord.json to use as the target distance.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=40,
        help="Max OpenEvolve iterations per instance.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete an existing instance directory and rerun it.",
    )
    return parser.parse_args()


def load_record(record_path: Path) -> dict:
    return json.loads(record_path.read_text())


def load_tasks_from_record(
    record_path: Path,
    n_min: int = 11,
    n_max: int = 40,
    d_field: str = "lower",
) -> tuple[list[SweepTask], list[dict]]:
    """Load valid tasks and a list of skipped entries from ECCRecord.json."""
    raw_record = load_record(record_path)
    tasks: list[SweepTask] = []
    skipped: list[dict] = []

    for n_key in sorted(raw_record, key=lambda value: int(value)):
        n = int(n_key)
        if n < n_min or n > n_max:
            continue

        for k_key in sorted(raw_record[n_key], key=lambda value: int(value)):
            k = int(k_key)
            bounds = raw_record[n_key][k_key]
            lower = int(bounds["lower"])
            upper = int(bounds["upper"])
            d = int(bounds[d_field])

            if n <= 0 or k <= 0 or k >= n:
                skipped.append(
                    {
                        "n": n,
                        "k": k,
                        "d": d,
                        "lower": lower,
                        "upper": upper,
                        "reason": "requires 1 <= k < n for the systematic parity-check search",
                    }
                )
                continue

            if d <= 2:
                skipped.append(
                    {
                        "n": n,
                        "k": k,
                        "d": d,
                        "lower": lower,
                        "upper": upper,
                        "reason": "target distance d <= 2 does not require search",
                    }
                )
                continue

            tasks.append(SweepTask(n=n, k=k, d=d, lower=lower, upper=upper))

    return tasks, skipped


def render_resolved_config(base_config_text: str, task: SweepTask) -> str:
    """Inject the current target block into the base config prompt."""
    replacement = (
        f"    Current target:\n"
        f"    - n = {task.n}\n"
        f"    - k = {task.k}\n"
        f"    - d = {task.d}\n"
        f"    - r = n - k = {task.r}\n"
    )
    if not TARGET_BLOCK_RE.search(base_config_text):
        raise ValueError("Failed to find the 'Current target' block in the base config.")
    return TARGET_BLOCK_RE.sub(replacement, base_config_text, count=1)


def write_resolved_config(base_config_path: Path, output_dir: Path, task: SweepTask) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = output_dir / "resolved_config.yaml"
    resolved_config.write_text(render_resolved_config(base_config_path.read_text(), task))
    return resolved_config


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def extract_best_metrics(instance_dir: Path) -> dict:
    info_path = instance_dir / "best" / "best_program_info.json"
    if not info_path.exists():
        return {}
    info = json.loads(info_path.read_text())
    metrics = info.get("metrics", {})
    return {key: metrics[key] for key in metrics}


def parse_verification_output(stdout: str) -> dict:
    parsed: dict[str, object] = {"verification_status": "missing"}
    match = re.search(r"^d_actual:\s+(\d+)$", stdout, flags=re.MULTILINE)
    if match:
        parsed["verification_status"] = "complete"
        parsed["distance"] = int(match.group(1))
        return parsed

    match = re.search(r"^d_partial:\s+(\d+)$", stdout, flags=re.MULTILINE)
    if match:
        parsed["verification_status"] = "partial"
        parsed["distance"] = int(match.group(1))
        return parsed

    match = re.search(r"^d_at_least:\s+(\d+)$", stdout, flags=re.MULTILINE)
    if match:
        parsed["verification_status"] = "target_only"
        parsed["distance_lower_bound"] = int(match.group(1))
        return parsed

    if stdout.strip():
        parsed["verification_status"] = "unknown"
    return parsed


def run_instance(
    task: SweepTask,
    base_config_path: Path,
    output_root: Path,
    iterations: int,
    force: bool = False,
) -> dict:
    """Run OpenEvolve for one task and save verification artifacts."""
    instance_dir = output_root / task.instance_name
    if instance_dir.exists():
        if not force:
            return {
                **asdict(task),
                "status": "skipped_existing",
                "output_dir": str(instance_dir),
                "config_path": str(instance_dir / "resolved_config.yaml"),
                "verification_status": "skipped",
            }
        shutil.rmtree(instance_dir)

    instance_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = write_resolved_config(base_config_path, instance_dir, task)

    env = os.environ.copy()
    env.update(
        {
            "LINEAR_CODE_N": str(task.n),
            "LINEAR_CODE_K": str(task.k),
            "LINEAR_CODE_D": str(task.d),
        }
    )

    openevolve_cmd = [
        sys.executable,
        str(REPO_ROOT / "openevolve-run.py"),
        "initial_program.py",
        "evaluator.py",
        "--config",
        str(resolved_config_path),
        "--iterations",
        str(iterations),
        "--output",
        str(instance_dir),
    ]

    write_json(
        instance_dir / "run_metadata.json",
        {
            "task": asdict(task),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command": openevolve_cmd,
            "env": {
                "LINEAR_CODE_N": env["LINEAR_CODE_N"],
                "LINEAR_CODE_K": env["LINEAR_CODE_K"],
                "LINEAR_CODE_D": env["LINEAR_CODE_D"],
            },
        },
    )

    run_result = subprocess.run(
        openevolve_cmd,
        cwd=SCRIPT_DIR,
        env=env,
        capture_output=True,
        text=True,
    )
    (instance_dir / "runner_stdout.log").write_text(run_result.stdout)
    (instance_dir / "runner_stderr.log").write_text(run_result.stderr)

    summary_row = {
        **asdict(task),
        "status": "completed" if run_result.returncode == 0 else "failed",
        "returncode": run_result.returncode,
        "output_dir": str(instance_dir),
        "config_path": str(resolved_config_path),
    }
    summary_row.update(extract_best_metrics(instance_dir))

    best_program_path = instance_dir / "best" / "best_program.py"
    if best_program_path.exists():
        verify_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "verify_distance.py"),
            "--no-progress",
            str(best_program_path),
        ]
        verify_result = subprocess.run(
            verify_cmd,
            cwd=SCRIPT_DIR,
            env=env,
            capture_output=True,
            text=True,
        )
        (instance_dir / "matrix_verification.txt").write_text(verify_result.stdout)
        (instance_dir / "matrix_verification.stderr.log").write_text(verify_result.stderr)
        summary_row.update(parse_verification_output(verify_result.stdout))
        if verify_result.returncode != 0:
            summary_row["verification_status"] = "failed"
            summary_row["verification_returncode"] = verify_result.returncode
    else:
        summary_row["verification_status"] = "missing_best_program"

    return summary_row


def append_summary_jsonl(summary_path: Path, row: dict) -> None:
    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_summary_csv(summary_path: Path, rows: Sequence[dict]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_batch(
    record_path: Path,
    base_config_path: Path,
    output_root: Path,
    n_min: int = 11,
    n_max: int = 40,
    d_field: str = "lower",
    iterations: int = 40,
    force: bool = False,
) -> list[dict]:
    tasks, skipped = load_tasks_from_record(
        record_path,
        n_min=n_min,
        n_max=n_max,
        d_field=d_field,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    summary_jsonl = output_root / "summary.jsonl"
    summary_csv = output_root / "summary.csv"
    if force and summary_jsonl.exists():
        summary_jsonl.unlink()

    rows: list[dict] = []
    for skipped_row in skipped:
        row = {
            **skipped_row,
            "d": skipped_row["d"],
            "status": (
                "skipped_low_distance"
                if skipped_row["reason"] == "target distance d <= 2 does not require search"
                else "skipped_invalid"
            ),
            "verification_status": "skipped",
            "output_dir": "",
            "config_path": "",
        }
        rows.append(row)
        append_summary_jsonl(summary_jsonl, row)

    for task in tasks:
        row = run_instance(
            task,
            base_config_path=base_config_path,
            output_root=output_root,
            iterations=iterations,
            force=force,
        )
        rows.append(row)
        append_summary_jsonl(summary_jsonl, row)
        print(
            f"[{row['status']}] {task.instance_name} "
            f"(score={row.get('combined_score', 'n/a')}, verification={row.get('verification_status', 'n/a')})"
        )

    write_summary_csv(summary_csv, rows)
    return rows


def main() -> None:
    args = parse_args()
    run_batch(
        record_path=Path(args.record).resolve(),
        base_config_path=Path(args.config).resolve(),
        output_root=Path(args.output_root).resolve(),
        n_min=args.n_min,
        n_max=args.n_max,
        d_field=args.d_field,
        iterations=args.iterations,
        force=args.force,
    )


if __name__ == "__main__":
    main()
