"""Render an OpenEvolve config with the active generator-matrix target in prompt."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def _env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _target_header(n: int, k: int, d: int) -> str:
    return "\n".join(
        [
            "Current target:",
            f"- n = {n}",
            f"- k = {k}",
            f"- d = {d}",
            f"- r = n - k = {n - k}",
        ]
    )


def inject_target(system_message: str, n: int, k: int, d: int) -> str:
    """Replace or prepend the concrete target block in a prompt."""
    header = _target_header(n, k, d)
    lines = system_message.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != "Current target:":
            continue
        end = index + 1
        while end < len(lines) and lines[end].strip().startswith("- "):
            end += 1
        return "\n".join(lines[:index] + header.splitlines() + lines[end:])
    return f"{header}\n\n{system_message}"


def render_config(input_path: Path, output_path: Path, n: int, k: int, d: int) -> None:
    config = yaml.safe_load(input_path.read_text())
    prompt = config.setdefault("prompt", {})
    prompt["system_message"] = inject_target(
        str(prompt.get("system_message", "")),
        n,
        k,
        d,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(config, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--N", type=int, default=_env_int("GEN_MATRIX_CODE_N", 20))
    parser.add_argument("--K", type=int, default=_env_int("GEN_MATRIX_CODE_K", 10))
    parser.add_argument("--D", type=int, default=_env_int("GEN_MATRIX_CODE_D", 5))
    args = parser.parse_args()
    render_config(args.input, args.output, args.N, args.K, args.D)


if __name__ == "__main__":
    main()
