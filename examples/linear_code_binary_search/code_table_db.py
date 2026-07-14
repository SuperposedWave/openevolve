#!/usr/bin/env python3
"""Maintain a SQLite record store for the binary-code table viewer."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import generate_viewer_data as viewer_data


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DB = SCRIPT_DIR / "code_table_records.sqlite"
DEFAULT_RECORD = SCRIPT_DIR / "Misc" / "ECCRecord.json"
DEFAULT_VIEWER_JSON = SCRIPT_DIR / "code_table_viewer" / "code_table_data.json"


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS code_bounds (
    n INTEGER NOT NULL,
    k INTEGER NOT NULL,
    lower_bound INTEGER NOT NULL,
    upper_bound INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (n, k)
);

CREATE TABLE IF NOT EXISTS search_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    n INTEGER NOT NULL,
    k INTEGER NOT NULL,
    target_distance INTEGER NOT NULL,
    status TEXT NOT NULL,
    actual_distance INTEGER,
    source_root TEXT NOT NULL,
    source_run TEXT NOT NULL,
    iteration INTEGER,
    generation INTEGER,
    timestamp TEXT,
    metrics_json TEXT NOT NULL,
    selected_free_columns_json TEXT NOT NULL,
    h_rows_json TEXT NOT NULL,
    g_rows_json TEXT NOT NULL,
    priority_source TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    imported_at TEXT NOT NULL,
    UNIQUE (n, k, target_distance, source_root, source_run)
);

CREATE INDEX IF NOT EXISTS idx_search_attempts_cell
    ON search_attempts (n, k);

CREATE INDEX IF NOT EXISTS idx_search_attempts_status
    ON search_attempts (status);

CREATE INDEX IF NOT EXISTS idx_search_attempts_source
    ON search_attempts (source_root, source_run);

CREATE TABLE IF NOT EXISTS record_meta (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def json_loads(value: str, default: Any) -> Any:
    if value == "":
        return default
    return json.loads(value)


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def set_meta(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute(
        """
        INSERT INTO record_meta (key, value_json, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value_json = excluded.value_json,
            updated_at = excluded.updated_at
        """,
        (key, json_dumps(value), utc_now()),
    )


def get_meta(conn: sqlite3.Connection, key: str, default: Any = None) -> Any:
    row = conn.execute("SELECT value_json FROM record_meta WHERE key = ?", (key,)).fetchone()
    if row is None:
        return default
    return json.loads(row["value_json"])


def import_bounds(conn: sqlite3.Connection, record_path: Path) -> int:
    record = viewer_data.load_record(record_path)
    now = utc_now()
    rows = [
        (int(n_key), int(k_key), bounds["lower"], bounds["upper"], now)
        for n_key, row in record.items()
        for k_key, bounds in row.items()
    ]
    conn.executemany(
        """
        INSERT INTO code_bounds (n, k, lower_bound, upper_bound, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(n, k) DO UPDATE SET
            lower_bound = excluded.lower_bound,
            upper_bound = excluded.upper_bound,
            updated_at = excluded.updated_at
        """,
        rows,
    )
    set_meta(conn, "recordName", record_path.name)
    set_meta(conn, "recordPath", str(record_path))
    conn.commit()
    return len(rows)


def normalize_attempt_payload(attempt: dict[str, Any]) -> dict[str, Any]:
    payload = dict(attempt)
    payload["n"] = int(payload["n"])
    payload["k"] = int(payload["k"])
    payload["targetDistance"] = int(payload["targetDistance"])
    payload["sourceRoot"] = str(payload.get("sourceRoot") or "unknown")
    payload["sourceRun"] = str(payload.get("sourceRun") or "unknown")
    payload["status"] = str(payload.get("status") or "unknown")
    payload.setdefault("actualDistance", None)
    payload.setdefault("selectedFreeColumns", [])
    payload.setdefault("hRows", [])
    payload.setdefault("gRows", [])
    payload.setdefault("prioritySource", "")
    payload.setdefault("metrics", {})
    payload.setdefault("iteration", None)
    payload.setdefault("generation", None)
    payload.setdefault("timestamp", None)
    return payload


def upsert_attempt(conn: sqlite3.Connection, attempt: dict[str, Any]) -> None:
    payload = normalize_attempt_payload(attempt)
    existing = conn.execute(
        """
        SELECT payload_json
        FROM search_attempts
        WHERE n = ? AND k = ? AND target_distance = ? AND source_root = ? AND source_run = ?
        """,
        (
            payload["n"],
            payload["k"],
            payload["targetDistance"],
            payload["sourceRoot"],
            payload["sourceRun"],
        ),
    ).fetchone()
    if existing is not None:
        existing_payload = normalize_attempt_payload(json.loads(existing["payload_json"]))
        if existing_payload.get("hRows") and not payload.get("hRows"):
            return
        if viewer_data.attempt_sort_score(existing_payload) > viewer_data.attempt_sort_score(payload):
            return
    conn.execute(
        """
        INSERT INTO search_attempts (
            n, k, target_distance, status, actual_distance, source_root, source_run,
            iteration, generation, timestamp, metrics_json, selected_free_columns_json,
            h_rows_json, g_rows_json, priority_source, payload_json, imported_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(n, k, target_distance, source_root, source_run) DO UPDATE SET
            status = excluded.status,
            actual_distance = excluded.actual_distance,
            iteration = excluded.iteration,
            generation = excluded.generation,
            timestamp = excluded.timestamp,
            metrics_json = excluded.metrics_json,
            selected_free_columns_json = excluded.selected_free_columns_json,
            h_rows_json = excluded.h_rows_json,
            g_rows_json = excluded.g_rows_json,
            priority_source = excluded.priority_source,
            payload_json = excluded.payload_json,
            imported_at = excluded.imported_at
        """,
        (
            payload["n"],
            payload["k"],
            payload["targetDistance"],
            payload["status"],
            payload["actualDistance"],
            payload["sourceRoot"],
            payload["sourceRun"],
            payload["iteration"],
            payload["generation"],
            payload["timestamp"],
            json_dumps(payload["metrics"]),
            json_dumps(payload["selectedFreeColumns"]),
            json_dumps(payload["hRows"]),
            json_dumps(payload["gRows"]),
            payload["prioritySource"],
            json_dumps(payload),
            utc_now(),
        ),
    )


def import_runs(conn: sqlite3.Connection, search_roots: list[Path]) -> int:
    attempts_by_cell = viewer_data.scan_attempts(search_roots)
    imported = 0
    for attempts in attempts_by_cell.values():
        for attempt in viewer_data.deduplicate_attempts(attempts):
            upsert_attempt(conn, attempt)
            imported += 1
    set_meta(conn, "lastRunImportRoots", [str(root) for root in search_roots])
    set_meta(conn, "lastRunImportAt", utc_now())
    conn.commit()
    return imported


def detail_attempt_payload(detail_id: str, detail: dict[str, Any], attempt: dict[str, Any]) -> dict[str, Any]:
    source_root = attempt.get("sourceRoot") or detail.get("sourceRoot") or "viewer_json"
    source_run = attempt.get("sourceRun") or detail.get("sourceRun") or detail_id
    payload = {
        "n": int(detail["n"]),
        "k": int(detail["k"]),
        "targetDistance": int(attempt.get("targetDistance") or detail.get("targetDistance") or 0),
        "status": attempt.get("status") or ("complete" if detail.get("completeConstruction") else "unknown"),
        "actualDistance": attempt.get("actualDistance"),
        "selectedFreeColumns": [],
        "hRows": [],
        "gRows": [],
        "prioritySource": "",
        "metrics": attempt.get("metrics") or {},
        "iteration": attempt.get("iteration"),
        "generation": attempt.get("generation"),
        "timestamp": attempt.get("timestamp"),
        "sourceRoot": source_root,
        "sourceRun": source_run,
    }
    is_best_detail = (
        detail.get("completeConstruction")
        and payload["status"] == "complete"
        and payload["targetDistance"] == detail.get("targetDistance")
        and payload["actualDistance"] == detail.get("bestDistance")
    )
    if is_best_detail:
        payload["selectedFreeColumns"] = detail.get("selectedFreeColumns") or []
        payload["hRows"] = detail.get("hRows") or []
        payload["gRows"] = detail.get("gRows") or []
        payload["prioritySource"] = detail.get("prioritySource") or ""
        payload["metrics"] = detail.get("metrics") or payload["metrics"]
    for optional_key in ("method", "derivedFrom"):
        if optional_key in attempt:
            payload[optional_key] = attempt[optional_key]
    return payload


def synthetic_detail_attempt(detail_id: str, detail: dict[str, Any]) -> dict[str, Any] | None:
    if not detail.get("completeConstruction"):
        return None
    target_distance = detail.get("targetDistance") or detail.get("bestDistance")
    if target_distance is None:
        return None
    return {
        "n": int(detail["n"]),
        "k": int(detail["k"]),
        "targetDistance": int(target_distance),
        "status": "complete",
        "actualDistance": detail.get("bestDistance"),
        "selectedFreeColumns": detail.get("selectedFreeColumns") or [],
        "hRows": detail.get("hRows") or [],
        "gRows": detail.get("gRows") or [],
        "prioritySource": detail.get("prioritySource") or "",
        "metrics": detail.get("metrics") or {},
        "iteration": None,
        "generation": None,
        "timestamp": None,
        "sourceRoot": detail.get("sourceRoot") or "viewer_json",
        "sourceRun": detail.get("sourceRun") or detail_id,
    }


def import_viewer_json(conn: sqlite3.Connection, dataset_path: Path) -> tuple[int, int]:
    dataset = json.loads(dataset_path.read_text())
    now = utc_now()
    bounds_rows = [
        (int(cell["n"]), int(cell["k"]), int(cell["lower"]), int(cell["upper"]), now)
        for cell in dataset["cells"]
    ]
    conn.executemany(
        """
        INSERT INTO code_bounds (n, k, lower_bound, upper_bound, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(n, k) DO UPDATE SET
            lower_bound = excluded.lower_bound,
            upper_bound = excluded.upper_bound,
            updated_at = excluded.updated_at
        """,
        bounds_rows,
    )
    imported_attempts = 0
    for detail_id, detail in dataset.get("details", {}).items():
        attempts = detail.get("attempts") or []
        for attempt in attempts:
            upsert_attempt(conn, detail_attempt_payload(detail_id, detail, attempt))
            imported_attempts += 1
        if not attempts:
            synthetic = synthetic_detail_attempt(detail_id, detail)
            if synthetic is not None:
                upsert_attempt(conn, synthetic)
                imported_attempts += 1
    set_meta(conn, "importedViewerJson", str(dataset_path))
    set_meta(conn, "importedViewerMeta", dataset.get("meta", {}))
    set_meta(conn, "lastViewerImportAt", utc_now())
    conn.commit()
    return len(bounds_rows), imported_attempts


def attempts_from_db(conn: sqlite3.Connection) -> dict[tuple[int, int], list[dict[str, Any]]]:
    attempts_by_cell: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in conn.execute("SELECT payload_json FROM search_attempts ORDER BY n, k, target_distance"):
        attempt = normalize_attempt_payload(json.loads(row["payload_json"]))
        attempts_by_cell[(attempt["n"], attempt["k"])].append(attempt)
    return attempts_by_cell


def build_dataset_from_db(conn: sqlite3.Connection) -> dict[str, Any]:
    attempts_by_cell = attempts_from_db(conn)
    cells: list[dict[str, Any]] = []
    details: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()

    rows = conn.execute(
        "SELECT n, k, lower_bound, upper_bound FROM code_bounds ORDER BY n, k"
    ).fetchall()
    for row in rows:
        n = int(row["n"])
        k = int(row["k"])
        lower = int(row["lower_bound"])
        upper = int(row["upper_bound"])
        attempts = viewer_data.deduplicate_attempts(attempts_by_cell.get((n, k), []))
        complete_attempts = [
            attempt
            for attempt in attempts
            if attempt["status"] == "complete" and attempt["actualDistance"] is not None
        ]
        best_attempt = max(complete_attempts, key=viewer_data.rank_attempt) if complete_attempts else None
        trivial_distance = lower <= 2
        best_distance = int(best_attempt["actualDistance"]) if best_attempt else (
            lower if trivial_distance else None
        )
        status = viewer_data.classify_cell(lower, upper, attempts, best_distance)
        detail_id = f"n{n}_k{k}" if attempts or trivial_distance else None
        if detail_id:
            details[detail_id] = viewer_data.build_detail(
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
            "label": viewer_data.cell_label(lower, upper, best_distance),
            "status": status,
            "bestDistance": best_distance,
            "attemptedTargets": sorted({int(attempt["targetDistance"]) for attempt in attempts}),
            "detailId": detail_id,
        }
        cells.append(cell)
        counts[status] += 1

    scanned_roots = [
        row["source_root"]
        for row in conn.execute(
            "SELECT DISTINCT source_root FROM search_attempts ORDER BY source_root"
        )
    ]
    return {
        "meta": {
            "generatedAt": utc_now(),
            "recordName": get_meta(conn, "recordName", "sqlite"),
            "scannedRootNames": scanned_roots,
            "totalCells": len(cells),
            "countsByStatus": dict(sorted(counts.items())),
            "sqlite": {
                "bounds": len(cells),
                "attempts": sum(len(value) for value in attempts_by_cell.values()),
            },
        },
        "cells": cells,
        "details": details,
    }


def export_viewer_json(conn: sqlite3.Connection, output_path: Path) -> dict[str, Any]:
    dataset = build_dataset_from_db(conn)
    viewer_data.validate_dataset(dataset, require_full_range=False)
    viewer_data.write_dataset(dataset, output_path)
    set_meta(conn, "lastViewerExport", str(output_path))
    set_meta(conn, "lastViewerExportAt", utc_now())
    conn.commit()
    return dataset


def add_common_db_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite database path.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create or migrate the SQLite schema.")
    add_common_db_arg(init_parser)

    record_parser = subparsers.add_parser("import-record", help="Import ECCRecord bounds.")
    add_common_db_arg(record_parser)
    record_parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)

    viewer_parser = subparsers.add_parser("import-viewer", help="Import an existing viewer JSON.")
    add_common_db_arg(viewer_parser)
    viewer_parser.add_argument("--input", type=Path, default=DEFAULT_VIEWER_JSON)

    runs_parser = subparsers.add_parser("import-runs", help="Scan run directories into SQLite.")
    add_common_db_arg(runs_parser)
    runs_parser.add_argument(
        "--search-root",
        type=Path,
        action="append",
        dest="search_roots",
        help="Result root to scan. May be passed more than once.",
    )

    export_parser = subparsers.add_parser("export-viewer", help="Export code_table_data.json.")
    add_common_db_arg(export_parser)
    export_parser.add_argument("--output", type=Path, default=DEFAULT_VIEWER_JSON)

    rebuild_parser = subparsers.add_parser(
        "rebuild-viewer",
        help="Import bounds and runs, then export the static viewer JSON.",
    )
    add_common_db_arg(rebuild_parser)
    rebuild_parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    rebuild_parser.add_argument("--output", type=Path, default=DEFAULT_VIEWER_JSON)
    rebuild_parser.add_argument(
        "--search-root",
        type=Path,
        action="append",
        dest="search_roots",
        help="Result root to scan. May be passed more than once.",
    )
    rebuild_parser.add_argument(
        "--reset-attempts",
        action="store_true",
        help="Clear existing attempts before scanning run directories.",
    )
    return parser.parse_args()


def command_search_roots(args: argparse.Namespace) -> list[Path]:
    if args.search_roots:
        return [root.resolve() for root in args.search_roots]
    return viewer_data.default_search_roots()


def main() -> None:
    args = parse_args()
    args.db.parent.mkdir(parents=True, exist_ok=True)
    with connect(args.db) as conn:
        init_db(conn)
        if args.command == "init":
            print(f"Initialized {args.db}")
        elif args.command == "import-record":
            count = import_bounds(conn, args.record.resolve())
            print(f"Imported {count} bounds into {args.db}")
        elif args.command == "import-viewer":
            bounds, attempts = import_viewer_json(conn, args.input.resolve())
            print(f"Imported {bounds} bounds and {attempts} attempts into {args.db}")
        elif args.command == "import-runs":
            count = import_runs(conn, command_search_roots(args))
            print(f"Imported {count} attempts into {args.db}")
        elif args.command == "export-viewer":
            dataset = export_viewer_json(conn, args.output.resolve())
            print(
                f"Wrote {args.output.resolve()} with {dataset['meta']['totalCells']} cells "
                f"and {len(dataset['details'])} details"
            )
        elif args.command == "rebuild-viewer":
            bounds = import_bounds(conn, args.record.resolve())
            if args.reset_attempts:
                conn.execute("DELETE FROM search_attempts")
                conn.commit()
            attempts = import_runs(conn, command_search_roots(args))
            dataset = export_viewer_json(conn, args.output.resolve())
            print(
                f"Imported {bounds} bounds and {attempts} attempts; wrote "
                f"{args.output.resolve()} with {dataset['meta']['totalCells']} cells "
                f"and {len(dataset['details'])} details"
            )


if __name__ == "__main__":
    main()
