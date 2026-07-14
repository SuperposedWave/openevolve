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

import code_table_db


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_RECORD = SCRIPT_DIR / "Misc" / "ECCRecord.json"
DEFAULT_CONFIG = SCRIPT_DIR / "Configs" / "config_c_kernel.yaml"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs"
DEFAULT_SQLITE_DB = SCRIPT_DIR / "code_table_records.sqlite"
DEFAULT_VIEWER_JSON = SCRIPT_DIR / "code_table_viewer" / "code_table_data.json"
TARGET_BLOCK_RE = re.compile(
    r"(?m)^    Current target:\n(?:^    - .*\n){4}"
)
LLM_CONFIG_PATH_RE = re.compile(r"(?m)^(llm_config_path:\s*)[\"']?([^\"'\n]+)[\"']?\s*$")
EARLY_STOPPING_SHUTDOWN_MODE_RE = re.compile(
    r'(?m)^early_stopping_shutdown_mode:\s*["\']?(wait|terminate)["\']?\s*$'
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
        help="Root directory where outputs are written as n{n}_k{k}_d{d}/{run_name}.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run-record folder name under each n/k/d directory. Defaults to batch_<UTC timestamp>.",
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
        "--early-stopping-shutdown-mode",
        choices=("wait", "terminate"),
        default="terminate",
        help="How OpenEvolve stops worker evaluations after early stopping.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete an existing instance directory and rerun it.",
    )
    parser.add_argument(
        "--sqlite-db",
        default=str(DEFAULT_SQLITE_DB),
        help="SQLite record store updated after each completed instance.",
    )
    parser.add_argument(
        "--viewer-output",
        default=str(DEFAULT_VIEWER_JSON),
        help="Viewer JSON refreshed after SQLite is updated.",
    )
    parser.add_argument(
        "--no-sqlite-update",
        action="store_true",
        help="Do not update the SQLite record store or viewer JSON.",
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


def render_resolved_config(
    base_config_text: str,
    task: SweepTask,
    base_config_path: Path,
    early_stopping_shutdown_mode: str = "terminate",
) -> str:
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
    resolved_text = TARGET_BLOCK_RE.sub(replacement, base_config_text, count=1)

    def resolve_llm_path(match: re.Match[str]) -> str:
        llm_path = Path(match.group(2)).expanduser()
        if not llm_path.is_absolute():
            llm_path = (base_config_path.parent / llm_path).resolve()
        return f'{match.group(1)}"{llm_path}"'

    resolved_text = LLM_CONFIG_PATH_RE.sub(resolve_llm_path, resolved_text, count=1)
    mode_line = f'early_stopping_shutdown_mode: "{early_stopping_shutdown_mode}"'
    if EARLY_STOPPING_SHUTDOWN_MODE_RE.search(resolved_text):
        return EARLY_STOPPING_SHUTDOWN_MODE_RE.sub(mode_line, resolved_text, count=1)
    if resolved_text.endswith("\n"):
        return f"{resolved_text}{mode_line}\n"
    return f"{resolved_text}\n{mode_line}\n"


def write_resolved_config(
    base_config_path: Path,
    output_dir: Path,
    task: SweepTask,
    early_stopping_shutdown_mode: str = "terminate",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = output_dir / "resolved_config.yaml"
    resolved_config.write_text(
        render_resolved_config(
            base_config_path.read_text(),
            task,
            base_config_path,
            early_stopping_shutdown_mode=early_stopping_shutdown_mode,
        )
    )
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

    if stdout.strip():
        parsed["verification_status"] = "unknown"
    return parsed


def run_instance(
    task: SweepTask,
    base_config_path: Path,
    output_root: Path,
    run_name: str,
    iterations: int,
    early_stopping_shutdown_mode: str = "terminate",
    force: bool = False,
) -> dict:
    """Run OpenEvolve for one task and save verification artifacts."""
    instance_dir = output_root / task.instance_name / run_name
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
    resolved_config_path = write_resolved_config(
        base_config_path,
        instance_dir,
        task,
        early_stopping_shutdown_mode=early_stopping_shutdown_mode,
    )

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
        "initial_program.c",
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
                "EARLY_STOPPING_SHUTDOWN_MODE": early_stopping_shutdown_mode,
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

    best_program_path = instance_dir / "best" / "best_program.c"
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


def update_sqlite_records(
    db_path: Path,
    record_path: Path,
    output_root: Path,
    viewer_output: Path,
) -> None:
    """Refresh the persistent code-table record store from local outputs."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with code_table_db.connect(db_path) as conn:
        code_table_db.init_db(conn)
        code_table_db.import_bounds(conn, record_path)
        code_table_db.import_runs(conn, [output_root])
        code_table_db.export_viewer_json(conn, viewer_output)


def run_batch(
    record_path: Path,
    base_config_path: Path,
    output_root: Path,
    run_name: str | None = None,
    n_min: int = 11,
    n_max: int = 40,
    d_field: str = "lower",
    iterations: int = 40,
    early_stopping_shutdown_mode: str = "terminate",
    force: bool = False,
    sqlite_db: Path | None = DEFAULT_SQLITE_DB,
    viewer_output: Path = DEFAULT_VIEWER_JSON,
) -> list[dict]:
    tasks, skipped = load_tasks_from_record(
        record_path,
        n_min=n_min,
        n_max=n_max,
        d_field=d_field,
    )
    if run_name is None:
        run_name = f"batch_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output_root.mkdir(parents=True, exist_ok=True)
    summary_dir = output_root / "_summaries" / run_name
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_jsonl = summary_dir / "summary.jsonl"
    summary_csv = summary_dir / "summary.csv"
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
            run_name=run_name,
            iterations=iterations,
            early_stopping_shutdown_mode=early_stopping_shutdown_mode,
            force=force,
        )
        rows.append(row)
        append_summary_jsonl(summary_jsonl, row)
        if sqlite_db is not None and row["status"] != "skipped_existing":
            update_sqlite_records(sqlite_db, record_path, output_root, viewer_output)
            print(f"[sqlite] updated {sqlite_db} and {viewer_output}")
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
        run_name=args.run_name,
        n_min=args.n_min,
        n_max=args.n_max,
        d_field=args.d_field,
        iterations=args.iterations,
        early_stopping_shutdown_mode=args.early_stopping_shutdown_mode,
        force=args.force,
        sqlite_db=None if args.no_sqlite_update else Path(args.sqlite_db).resolve(),
        viewer_output=Path(args.viewer_output).resolve(),
    )


if __name__ == "__main__":
    main()
