"""Manual smoke benchmark for the optional native legality engine.

This script is intentionally not part of the default pytest suite.  The large
target case, for example ``--r 30 --d 13``, can allocate substantial memory.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _random_mask(r: int, min_weight: int, rng: random.Random) -> int:
    weight = rng.randint(min_weight, r)
    mask = 0
    for bit in rng.sample(range(r), weight):
        mask |= 1 << bit
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke benchmark for _linear_code_native.NativeForbiddenState."
    )
    parser.add_argument("--r", type=int, default=10)
    parser.add_argument("--d", type=int, default=5)
    parser.add_argument("--adds", type=int, default=4)
    parser.add_argument("--attempts-per-add", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    try:
        import _linear_code_native as native
    except ImportError as exc:
        raise SystemExit(
            "Native extension is not built. Run `python setup.py build_ext --inplace` first."
        ) from exc

    rng = random.Random(args.seed)
    min_weight = max(args.d - 1, 0)

    start = time.perf_counter()
    state = native.NativeForbiddenState(args.r, args.d)
    init_seconds = time.perf_counter() - start
    print(
        "init "
        f"r={args.r} d={args.d} seconds={init_seconds:.6f} "
        f"forbidden_count={state.forbidden_count()} "
        f"layer_counts={state.layer_counts()}"
    )

    add_seconds = 0.0
    clone_seconds = 0.0
    selected = []
    for add_index in range(1, args.adds + 1):
        column = None
        for _ in range(args.attempts_per_add):
            candidate = _random_mask(args.r, min_weight, rng)
            if state.can_add(candidate):
                column = candidate
                break
        if column is None:
            print(f"add {add_index}: no legal sampled candidate")
            break

        start = time.perf_counter()
        growth = state.add(column)
        add_seconds += time.perf_counter() - start
        selected.append(column)

        start = time.perf_counter()
        clone = state.clone()
        clone_seconds += time.perf_counter() - start
        del clone

        print(
            f"add {add_index}: column={column} growth={growth} "
            f"forbidden_count={state.forbidden_count()} "
            f"layer_counts={state.layer_counts()}"
        )

    print(
        "summary "
        f"selected={len(selected)} add_seconds={add_seconds:.6f} "
        f"clone_seconds={clone_seconds:.6f}"
    )


if __name__ == "__main__":
    main()
