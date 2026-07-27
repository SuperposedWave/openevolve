#!/usr/bin/env python3
"""Fast exact evaluation for systematic G=[I_k|P] rows.

The C-kernel search in this example returns ``selected_free_columns`` as the
free columns of ``H=[P^T|I_r]``.  The same masks are the rows of ``P`` in
``G=[I_k|P]``.  This module evaluates those rows directly:

    wt(uG) = wt(u) + wt(uP)

for every non-zero message ``u``.  A Gray-code walk updates ``uP`` by xoring
only the row that changed between consecutive messages.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import time
from typing import Sequence


@dataclass(frozen=True)
class ParityMapEvaluation:
    """Exact distance and diagnostics for rows of P in G=[I_k|P]."""

    row_count: int
    r: int
    target_distance: int | None
    minimum_distance: int
    minimum_message: int
    minimum_parity: int
    minimum_message_weight: int
    minimum_parity_weight: int
    violation_count: int
    minimum_margin: int | None
    evaluated_messages: int
    elapsed_seconds: float

    @property
    def success(self) -> bool:
        return (
            self.target_distance is not None
            and self.minimum_distance >= self.target_distance
            and self.violation_count == 0
        )


def parse_mask(value: int | str, width: int | None = None) -> int:
    """Parse an integer or binary mask string."""
    if isinstance(value, int):
        mask = value
    else:
        raw = value.strip()
        if not raw:
            raise ValueError("empty mask")
        if re.fullmatch(r"[01]+", raw):
            mask = int(raw, 2)
            if width is not None and len(raw) > width:
                raise ValueError(f"mask {raw!r} is wider than r={width}")
        else:
            mask = int(raw, 0)
    if mask < 0:
        raise ValueError("mask must be non-negative")
    if width is not None and mask >= (1 << width):
        raise ValueError(f"mask {mask} does not fit in r={width}")
    return mask


def evaluate_parity_rows_gray(
    parity_rows: Sequence[int | str],
    *,
    r: int,
    target_distance: int | None = None,
) -> ParityMapEvaluation:
    """Evaluate min_{u != 0} wt(u) + wt(uP) by a Gray-code walk."""
    if r <= 0:
        raise ValueError("r must be positive")
    if target_distance is not None and target_distance <= 0:
        raise ValueError("target_distance must be positive")

    rows = tuple(parse_mask(row, r) for row in parity_rows)
    row_count = len(rows)
    if row_count <= 0:
        raise ValueError("at least one parity row is required")
    if row_count >= 63:
        raise ValueError("Gray-code evaluator supports row_count < 63")

    started_at = time.perf_counter()
    message_count = (1 << row_count) - 1
    message = 0
    parity = 0
    best_distance = row_count + r + 1
    best_message = 0
    best_parity = 0
    violation_count = 0

    previous_gray = 0
    for index in range(1, message_count + 1):
        gray = index ^ (index >> 1)
        changed = gray ^ previous_gray
        changed_bit = changed.bit_length() - 1
        message = gray
        parity ^= rows[changed_bit]
        distance = message.bit_count() + parity.bit_count()
        if distance < best_distance:
            best_distance = distance
            best_message = message
            best_parity = parity
        if target_distance is not None and distance < target_distance:
            violation_count += 1
        previous_gray = gray

    elapsed_seconds = time.perf_counter() - started_at
    return ParityMapEvaluation(
        row_count=row_count,
        r=r,
        target_distance=target_distance,
        minimum_distance=best_distance,
        minimum_message=best_message,
        minimum_parity=best_parity,
        minimum_message_weight=best_message.bit_count(),
        minimum_parity_weight=best_parity.bit_count(),
        violation_count=violation_count,
        minimum_margin=(
            None if target_distance is None else best_distance - target_distance
        ),
        evaluated_messages=message_count,
        elapsed_seconds=elapsed_seconds,
    )


def load_selected_free_columns(path: Path) -> tuple[list[str], int | None, int | None]:
    """Load selected_free_columns plus optional r/d metadata from an artifact file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if "artifacts" in data:
        artifacts = data["artifacts"]
        search_result = json.loads(artifacts.get("search_result", "{}"))
        matrix_summary = json.loads(artifacts.get("matrix_summary", "{}"))
        selected = search_result.get("selected_free_columns") or matrix_summary.get(
            "selected_free_columns",
            [],
        )
        r = matrix_summary.get("r")
        d = matrix_summary.get("d") or matrix_summary.get("d_target")
        return list(selected), int(r) if r is not None else None, int(d) if d is not None else None
    if "selected_free_columns" in data:
        selected = data["selected_free_columns"]
        r = data.get("r")
        d = data.get("d") or data.get("d_target") or data.get("target_distance")
        return list(selected), int(r) if r is not None else None, int(d) if d is not None else None
    if isinstance(data, list):
        return list(data), None, None
    raise ValueError(
        "expected a best_program_info artifact, an object with selected_free_columns, or a list"
    )


def _format_mask(mask: int, width: int) -> str:
    return format(mask, f"0{width}b")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast exact Gray-code evaluation of selected_free_columns as P rows."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="JSON artifact/list containing selected_free_columns.",
    )
    parser.add_argument("--r", type=int, help="Redundancy / parity width.")
    parser.add_argument("--d", type=int, help="Target minimum distance.")
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    args = parser.parse_args()

    selected, inferred_r, inferred_d = load_selected_free_columns(args.input)
    r = args.r if args.r is not None else inferred_r
    d = args.d if args.d is not None else inferred_d
    if r is None:
        if not selected:
            raise SystemExit("cannot infer r from empty selected_free_columns")
        r = max(len(str(mask)) if re.fullmatch(r"[01]+", str(mask)) else int(mask).bit_length() for mask in selected)

    result = evaluate_parity_rows_gray(selected, r=r, target_distance=d)
    payload = asdict(result)
    payload["success"] = result.success
    payload["minimum_message_bits"] = _format_mask(
        result.minimum_message,
        result.row_count,
    )
    payload["minimum_parity_bits"] = _format_mask(result.minimum_parity, r)
    print(json.dumps(payload, indent=2 if args.pretty else None, sort_keys=True))


if __name__ == "__main__":
    main()
