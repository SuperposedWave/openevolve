#!/usr/bin/env python3
"""Generate static JSON data for the binary-code search matrix viewer."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RECORD = SCRIPT_DIR / "Misc" / "ECCRecord.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "code_table_viewer" / "code_table_data.json"
TARGET_DIR_RE = re.compile(r"^n(?P<n>\d+)_k(?P<k>\d+)_d(?P<d>\d+)(?:[_-].*)?$")
STATUS_RANK = {
    "missing": 0,
    "unknown": 1,
    "partial": 2,
    "complete": 3,
}


def default_search_roots(base_dir: Path = SCRIPT_DIR) -> list[Path]:
    """Return local result roots that may contain n{n}_k{k}_d{d} run directories."""
    roots: list[Path] = []
    for child in sorted(base_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name == "outputs":
            roots.append(child)
    return roots


def parse_target_from_path(path: Path) -> tuple[int, int, int] | None:
    """Parse the nearest n/k/d target marker from a path."""
    for part in reversed(path.parts):
        match = TARGET_DIR_RE.match(part)
        if match:
            return (
                int(match.group("n")),
                int(match.group("k")),
                int(match.group("d")),
            )
    return None


def load_record(record_path: Path) -> dict[str, dict[str, dict[str, int]]]:
    """Load the CodeTables-derived bound record."""
    raw = json.loads(record_path.read_text())
    record: dict[str, dict[str, dict[str, int]]] = {}
    for n_key, row in raw.items():
        record[n_key] = {}
        for k_key, bounds in row.items():
            record[n_key][k_key] = {
                "lower": int(bounds["lower"]),
                "upper": int(bounds["upper"]),
            }
    return record


def parse_matrix_verification(path: Path) -> dict[str, Any]:
    """Parse a saved verify_distance.py text artifact without rerunning search."""
    if not path.exists():
        return {
            "status": "missing",
            "distance": None,
            "selectedFreeColumns": [],
            "hRows": [],
            "gRows": [],
        }

    text = path.read_text(errors="replace")
    status = "unknown"
    distance: int | None = None
    actual_match = re.search(r"^d_actual:\s+(\d+)$", text, flags=re.MULTILINE)
    partial_match = re.search(r"^d_partial:\s+(\d+)$", text, flags=re.MULTILINE)
    if actual_match:
        status = "complete"
        distance = int(actual_match.group(1))
    elif partial_match:
        status = "partial"
        distance = int(partial_match.group(1))

    selected_free_columns: list[str] = []
    construction_match = re.search(r"^construction:\s+(.+)$", text, flags=re.MULTILINE)
    if construction_match:
        try:
            construction = json.loads(construction_match.group(1))
            selected_free_columns = list(construction.get("selected_free_columns", []))
        except json.JSONDecodeError:
            selected_free_columns = []

    h_rows: list[str] = []
    g_rows: list[str] = []
    section: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line in {"H rows:", "Partial H rows:"}:
            section = "H"
            continue
        if line in {"G rows:", "Partial G rows:"}:
            section = "G"
            continue
        if re.fullmatch(r"[01]+", line):
            if section == "H":
                h_rows.append(line)
            elif section == "G":
                g_rows.append(line)

    return {
        "status": status,
        "distance": distance,
        "selectedFreeColumns": selected_free_columns,
        "hRows": h_rows,
        "gRows": g_rows,
    }


def read_best_info(path: Path) -> dict[str, Any]:
    """Read best_program_info.json when present."""
    if not path.exists():
        return {"iteration": None, "metrics": {}, "timestamp": None}
    try:
        info = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"iteration": None, "metrics": {}, "timestamp": None}
    best_program_time = info.get("saved_at")
    if best_program_time is None:
        best_program_time = info.get("timestamp")
    return {
        "iteration": info.get("iteration"),
        "generation": info.get("generation"),
        "metrics": info.get("metrics", {}),
        "timestamp": best_program_time,
    }


def is_checkpoint_path(path: Path) -> bool:
    """Return whether a run-like directory is inside an OpenEvolve checkpoint tree."""
    return any(part == "checkpoints" or part.startswith("checkpoint_") for part in path.parts)


def best_program_path_for(run_dir: Path) -> Path | None:
    """Return the best program path, preferring the current C-kernel output."""
    candidates = [
        run_dir / "best" / "best_program.c",
        run_dir / "best" / "best_program.py",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def scan_attempts(search_roots: list[Path]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    """Scan local run directories and group attempts by (n, k)."""
    attempts_by_cell: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    seen_dirs: set[Path] = set()
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for run_dir in search_root.rglob("*"):
            if not run_dir.is_dir() or run_dir in seen_dirs:
                continue
            if is_checkpoint_path(run_dir.relative_to(search_root)):
                continue
            best_program_path = best_program_path_for(run_dir)
            best_info_path = run_dir / "best" / "best_program_info.json"
            verification_path = run_dir / "matrix_verification.txt"
            if not (best_program_path or best_info_path.exists() or verification_path.exists()):
                continue
            relative_run_dir = run_dir.relative_to(search_root)
            target = parse_target_from_path(relative_run_dir)
            if target is None:
                continue
            seen_dirs.add(run_dir)
            n, k, target_distance = target
            verification = parse_matrix_verification(verification_path)
            best_info = read_best_info(best_info_path)
            priority_source = (
                best_program_path.read_text(errors="replace") if best_program_path else ""
            )
            attempts_by_cell[(n, k)].append(
                {
                    "n": n,
                    "k": k,
                    "targetDistance": target_distance,
                    "status": verification["status"],
                    "actualDistance": verification["distance"],
                    "selectedFreeColumns": verification["selectedFreeColumns"],
                    "hRows": verification["hRows"],
                    "gRows": verification["gRows"],
                    "prioritySource": priority_source,
                    "metrics": best_info["metrics"],
                    "iteration": best_info.get("iteration"),
                    "generation": best_info.get("generation"),
                    "timestamp": best_info.get("timestamp"),
                    "sourceRoot": search_root.name,
                    "sourceRun": str(relative_run_dir),
                }
            )
    return attempts_by_cell


def attempt_sort_score(attempt: dict[str, Any]) -> tuple[int, int, float, float, int, str]:
    """Score duplicate attempts so one run contributes only its strongest row."""
    status_rank = STATUS_RANK.get(str(attempt.get("status")), 1)
    actual_distance = int(attempt.get("actualDistance") or -1)
    metrics = attempt.get("metrics", {})
    combined_score = metrics.get("combined_score", -1)
    if not isinstance(combined_score, (int, float)):
        combined_score = -1
    constructed_columns = metrics.get("constructed_columns", -1)
    if not isinstance(constructed_columns, (int, float)):
        constructed_columns = -1
    iteration = attempt.get("iteration")
    if not isinstance(iteration, int):
        iteration = -1
    timestamp = attempt.get("timestamp") or ""
    return (
        status_rank,
        actual_distance,
        float(combined_score),
        float(constructed_columns),
        iteration,
        str(timestamp),
    )


def deduplicate_attempts(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove checkpoint rows and collapse duplicate rows from the same run."""
    best_by_run: dict[tuple[int, str, str], dict[str, Any]] = {}
    for attempt in attempts:
        source_run = str(attempt.get("sourceRun", ""))
        if is_checkpoint_path(Path(source_run)):
            continue
        key = (
            int(attempt["targetDistance"]),
            str(attempt.get("sourceRoot", "")),
            source_run,
        )
        existing = best_by_run.get(key)
        if existing is None or attempt_sort_score(attempt) > attempt_sort_score(existing):
            best_by_run[key] = attempt
    return sorted(
        best_by_run.values(),
        key=lambda item: (item["targetDistance"], item["sourceRoot"], item["sourceRun"]),
    )


def rank_attempt(attempt: dict[str, Any]) -> tuple[int, int, float, int, int]:
    """Rank attempts for choosing the best detail payload."""
    complete = 1 if attempt["status"] == "complete" else 0
    distance = int(attempt["actualDistance"] or -1)
    score = attempt["metrics"].get("combined_score", -1)
    if not isinstance(score, (int, float)):
        score = -1
    target_distance = int(attempt["targetDistance"])
    iteration = attempt.get("iteration")
    if not isinstance(iteration, int):
        iteration = -1
    return (complete, distance, float(score), target_distance, iteration)


def cell_label(lower: int, upper: int, best_distance: int | None) -> str:
    """Render lower, lower-upper, or lower-best-upper."""
    if lower == upper:
        return str(lower)
    if best_distance is not None and lower < best_distance < upper:
        return f"{lower}-{best_distance}-{upper}"
    return f"{lower}-{upper}"


def classify_cell(
    lower: int,
    upper: int,
    attempts: list[dict[str, Any]],
    best_distance: int | None,
) -> str:
    """Classify a matrix cell based on local attempts and known bounds."""
    if lower <= 2:
        attempted_targets = {int(attempt["targetDistance"]) for attempt in attempts}
        if lower < upper and best_distance is not None and best_distance < upper and upper in attempted_targets:
            return "upper_failed_after_found"
        return "found"
    if not attempts:
        return "unsearched"
    if best_distance is None or best_distance < lower:
        return "failed"
    attempted_targets = {int(attempt["targetDistance"]) for attempt in attempts}
    if lower < upper and best_distance < upper and upper in attempted_targets:
        return "upper_failed_after_found"
    return "found"


def build_detail(
    n: int,
    k: int,
    lower: int,
    upper: int,
    attempts: list[dict[str, Any]],
    best_attempt: dict[str, Any] | None,
    trivial_distance: bool = False,
) -> dict[str, Any]:
    """Build detail data for a searched cell."""
    complete_attempt = best_attempt if best_attempt and best_attempt["status"] == "complete" else None
    serialized_attempts = [
        {
            "targetDistance": attempt["targetDistance"],
            "status": attempt["status"],
            "actualDistance": attempt["actualDistance"],
            "metrics": attempt["metrics"],
            "iteration": attempt.get("iteration"),
            "generation": attempt.get("generation"),
            "timestamp": attempt.get("timestamp"),
            "sourceRoot": attempt["sourceRoot"],
            "sourceRun": attempt["sourceRun"],
        }
        for attempt in sorted(attempts, key=lambda item: (item["targetDistance"], item["sourceRoot"], item["sourceRun"]))
    ]
    return {
        "n": n,
        "k": k,
        "lower": lower,
        "upper": upper,
        "completeConstruction": complete_attempt is not None,
        "trivialDistance": trivial_distance,
        "bestDistance": complete_attempt["actualDistance"] if complete_attempt else (lower if trivial_distance else None),
        "targetDistance": complete_attempt["targetDistance"] if complete_attempt else None,
        "prioritySource": complete_attempt["prioritySource"] if complete_attempt else "",
        "hRows": complete_attempt["hRows"] if complete_attempt else [],
        "gRows": complete_attempt["gRows"] if complete_attempt else [],
        "selectedFreeColumns": complete_attempt["selectedFreeColumns"] if complete_attempt else [],
        "metrics": complete_attempt["metrics"] if complete_attempt else {},
        "sourceRoot": complete_attempt["sourceRoot"] if complete_attempt else "",
        "sourceRun": complete_attempt["sourceRun"] if complete_attempt else "",
        "attempts": serialized_attempts,
    }


def build_dataset(record_path: Path, search_roots: list[Path]) -> dict[str, Any]:
    """Build the full static JSON payload."""
    generated_at = datetime.now(timezone.utc).isoformat()
    record = load_record(record_path)
    attempts_by_cell = scan_attempts(search_roots)
    cells: list[dict[str, Any]] = []
    details: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()

    for n_key in sorted(record, key=lambda value: int(value)):
        n = int(n_key)
        for k_key in sorted(record[n_key], key=lambda value: int(value)):
            k = int(k_key)
            lower = record[n_key][k_key]["lower"]
            upper = record[n_key][k_key]["upper"]
            attempts = deduplicate_attempts(attempts_by_cell.get((n, k), []))
            complete_attempts = [
                attempt
                for attempt in attempts
                if attempt["status"] == "complete" and attempt["actualDistance"] is not None
            ]
            best_attempt = max(complete_attempts, key=rank_attempt) if complete_attempts else None
            trivial_distance = lower <= 2
            best_distance = int(best_attempt["actualDistance"]) if best_attempt else (lower if trivial_distance else None)
            status = classify_cell(lower, upper, attempts, best_distance)
            detail_id = f"n{n}_k{k}" if attempts or trivial_distance else None
            if detail_id:
                details[detail_id] = build_detail(
                    n,
                    k,
                    lower,
                    upper,
                    attempts,
                    best_attempt,
                    trivial_distance=trivial_distance and best_attempt is None,
                )
            cell = {
                "n": n,
                "k": k,
                "lower": lower,
                "upper": upper,
                "label": cell_label(lower, upper, best_distance),
                "status": status,
                "bestDistance": best_distance,
                "attemptedTargets": sorted({int(attempt["targetDistance"]) for attempt in attempts}),
                "detailId": detail_id,
            }
            cells.append(cell)
            counts[status] += 1

    scanned_roots = [root.name for root in search_roots if root.exists()]
    return {
        "meta": {
            "generatedAt": generated_at,
            "recordName": record_path.name,
            "scannedRootNames": scanned_roots,
            "totalCells": len(cells),
            "countsByStatus": dict(sorted(counts.items())),
        },
        "cells": cells,
        "details": details,
    }


def validate_dataset(dataset: dict[str, Any], require_full_range: bool = False) -> None:
    """Validate core invariants before writing viewer data."""
    cells = dataset["cells"]
    keys = {(cell["n"], cell["k"]) for cell in cells}
    if len(keys) != len(cells):
        raise ValueError("Duplicate (n, k) cells found")
    if require_full_range:
        expected = {(n, k) for n in range(1, 257) for k in range(1, n + 1)}
        if keys != expected:
            missing = sorted(expected - keys)[:10]
            extra = sorted(keys - expected)[:10]
            raise ValueError(f"Expected full 1..256 lower triangle; missing={missing} extra={extra}")
    for cell in cells:
        detail_id = cell.get("detailId")
        if cell["status"] != "unsearched" and not detail_id:
            raise ValueError(f"Searched cell lacks detailId: {cell}")
        if detail_id and detail_id not in dataset["details"]:
            raise ValueError(f"Missing detail for {detail_id}")
        best_distance = cell["bestDistance"]
        if best_distance is not None and cell["lower"] < best_distance < cell["upper"]:
            expected_label = f"{cell['lower']}-{best_distance}-{cell['upper']}"
            if cell["label"] != expected_label:
                raise ValueError(f"Intermediate label mismatch: {cell}")
    for detail_id, detail in dataset["details"].items():
        if detail["hRows"] and not detail["completeConstruction"]:
            raise ValueError(f"Detail {detail_id} has H rows without a complete construction")


def write_dataset(dataset: dict[str, Any], output_path: Path) -> None:
    """Write JSON with stable formatting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dataset, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate static data for the code table viewer.")
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD, help="Path to ECCRecord.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path")
    parser.add_argument(
        "--search-root",
        type=Path,
        action="append",
        dest="search_roots",
        help="Result root to scan. May be passed more than once.",
    )
    parser.add_argument(
        "--no-full-range-check",
        action="store_true",
        help="Skip the 1..256 lower-triangle validation for synthetic records.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record_path = args.record.resolve()
    search_roots = [root.resolve() for root in args.search_roots] if args.search_roots else default_search_roots()
    dataset = build_dataset(record_path, search_roots)
    validate_dataset(dataset, require_full_range=not args.no_full_range_check)
    write_dataset(dataset, args.output.resolve())
    print(
        f"Wrote {args.output.resolve()} with {dataset['meta']['totalCells']} cells "
        f"and {len(dataset['details'])} searched details"
    )


if __name__ == "__main__":
    main()
