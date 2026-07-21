"""OpenEvolve evaluator for generator-matrix binary linear-code search."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from search_core import EvaluationResult, evaluate_priority_function


def _load_priority_function(program_path: Path):
    if program_path.suffix.lower() != ".py":
        raise ValueError("generator-matrix evaluator accepts Python priority files only")
    if not program_path.exists():
        raise FileNotFoundError(str(program_path))

    example_dir = Path(__file__).resolve().parent
    for path in (example_dir, program_path.parent):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)

    module_name = f"generator_matrix_candidate_{abs(hash(program_path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load {program_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if hasattr(module, "get_priority_function"):
        priority_fn = module.get_priority_function()
    else:
        priority_fn = getattr(module, "priority", None)
    if not callable(priority_fn):
        raise AttributeError("program must define priority() or get_priority_function()")
    return priority_fn


def evaluate(program_path: str) -> EvaluationResult:
    """Evaluate an evolved priority function."""
    try:
        priority_fn = _load_priority_function(Path(program_path).resolve())
    except Exception as exc:
        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "success_rate": 0.0,
                "constructed_columns": 0.0,
                "target_columns": 0.0,
                "column_progress": 0.0,
                "coverage_progress": 0.0,
            },
            artifacts={
                "search_result": json.dumps(
                    {
                        "success": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    sort_keys=True,
                )
            },
        )
    return evaluate_priority_function(priority_fn)
