"""OpenEvolve evaluator for G-row incremental legality search."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

try:
    from openevolve.evaluation_result import EvaluationResult
except Exception:
    _EVAL_RESULT_PATH = (
        Path(__file__).resolve().parents[2] / "openevolve" / "evaluation_result.py"
    )
    _EVAL_RESULT_SPEC = importlib.util.spec_from_file_location(
        "openevolve_evaluation_result_g_row_fallback",
        _EVAL_RESULT_PATH,
    )
    if _EVAL_RESULT_SPEC is None or _EVAL_RESULT_SPEC.loader is None:
        raise ImportError("Failed to load EvaluationResult fallback")
    _EVAL_RESULT_MODULE = importlib.util.module_from_spec(_EVAL_RESULT_SPEC)
    _EVAL_RESULT_SPEC.loader.exec_module(_EVAL_RESULT_MODULE)
    EvaluationResult = _EVAL_RESULT_MODULE.EvaluationResult

_EXAMPLE_DIR = Path(__file__).resolve().parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

from g_row_legal_search import RowSearchConfig, evaluate_priority_function


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None:
        value = default
    else:
        try:
            value = int(raw)
        except ValueError:
            value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _optional_env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def config_from_env() -> RowSearchConfig:
    """Build row-search config from LINEAR_CODE_* environment variables."""
    n = _env_int("LINEAR_CODE_N", 50, minimum=1)
    k = _env_int("LINEAR_CODE_K", 20, minimum=1)
    d = _env_int("LINEAR_CODE_D", 12, minimum=1)
    return RowSearchConfig(
        n=n,
        k=k,
        d=d,
        restarts=_env_int("LINEAR_CODE_G_ROW_RESTARTS", _env_int("LINEAR_CODE_RESTARTS", 8), minimum=1),
        max_attempts_per_step=_env_int(
            "LINEAR_CODE_G_ROW_MAX_ATTEMPTS_PER_STEP",
            50000,
            minimum=1,
        ),
        legal_pool_target=_env_int("LINEAR_CODE_G_ROW_LEGAL_POOL_TARGET", 8, minimum=1),
        seed=_env_int("LINEAR_CODE_RANDOM_SEED", 1),
        min_row_weight=_optional_env_int("LINEAR_CODE_G_ROW_MIN_ROW_WEIGHT"),
        prefer_weight=_optional_env_int("LINEAR_CODE_G_ROW_PREFER_WEIGHT"),
        near_margin_radius=_env_int("LINEAR_CODE_G_ROW_NEAR_MARGIN_RADIUS", 1, minimum=0),
        repair_events=_env_int("LINEAR_CODE_G_ROW_REPAIR_EVENTS", 4, minimum=0),
        repair_drop_count=_env_int("LINEAR_CODE_G_ROW_REPAIR_DROP_COUNT", 2, minimum=0),
        repair_strategy=os.environ.get("LINEAR_CODE_G_ROW_REPAIR_STRATEGY", "recent"),
        repair_tabu_tenure=_env_int("LINEAR_CODE_G_ROW_REPAIR_TABU_TENURE", 16, minimum=0),
    )


def _load_priority_function(program_path: Path):
    if program_path.suffix.lower() != ".py":
        raise ValueError("G-row evaluator accepts Python priority files only")
    if not program_path.exists():
        raise FileNotFoundError(str(program_path))

    example_dir = Path(__file__).resolve().parent
    for path in (example_dir, program_path.parent):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)

    module_name = f"linear_code_g_row_candidate_{abs(hash(program_path.resolve()))}"
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


def _failure_result(error_type: str, message: str, started_at: float) -> EvaluationResult:
    elapsed = time.perf_counter() - started_at
    config = config_from_env()
    return EvaluationResult(
        metrics={
            "combined_score": 0.0,
            "success_rate": 0.0,
            "constructed_rows": 0.0,
            "target_rows": float(config.k),
            "row_progress": 0.0,
            "exact_minimum_distance": 0.0,
            "exact_violation_count": 0.0,
            "total_attempts": 0.0,
            "repair_events": 0.0,
            "dropped_rows": 0.0,
            "evaluation_time_seconds": elapsed,
        },
        artifacts={
            "search_result": json.dumps(
                {
                    "success": False,
                    "error_type": error_type,
                    "error": message,
                    "config": config.__dict__,
                },
                sort_keys=True,
            )
        },
    )


def evaluate(program_path: str) -> EvaluationResult:
    """Evaluate a Python legal-row priority function."""
    started_at = time.perf_counter()
    try:
        priority_fn = _load_priority_function(Path(program_path).resolve())
        config = config_from_env()
        metrics, artifacts = evaluate_priority_function(priority_fn, config)
    except Exception as exc:
        return _failure_result(type(exc).__name__, str(exc), started_at)

    metrics["evaluation_time_seconds"] = time.perf_counter() - started_at
    return EvaluationResult(metrics=metrics, artifacts=artifacts)
