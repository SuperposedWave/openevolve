"""Compile and run evolved C priorities for binary maximum-code search."""

from __future__ import annotations

import ctypes
import importlib.util
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from queue import Empty
from typing import Sequence

try:
    from openevolve.evaluation_result import EvaluationResult
except Exception:
    _EVAL_RESULT_PATH = (
        Path(__file__).resolve().parents[2] / "openevolve" / "evaluation_result.py"
    )
    _EVAL_RESULT_SPEC = importlib.util.spec_from_file_location(
        "openevolve_evaluation_result_max_c_kernel_fallback",
        _EVAL_RESULT_PATH,
    )
    if _EVAL_RESULT_SPEC is None or _EVAL_RESULT_SPEC.loader is None:
        raise ImportError("Failed to load EvaluationResult fallback")
    _EVAL_RESULT_MODULE = importlib.util.module_from_spec(_EVAL_RESULT_SPEC)
    _EVAL_RESULT_SPEC.loader.exec_module(_EVAL_RESULT_MODULE)
    EvaluationResult = _EVAL_RESULT_MODULE.EvaluationResult

from search_core import format_word, instance_from_env, validate_code


METRIC_CODE_SIZE = 0
METRIC_VALID = 1
METRIC_CANDIDATE_COUNT = 2
METRIC_SCORED_CANDIDATES = 3
METRIC_REPAIR_ROLLOUT_EVALUATIONS = 4
METRIC_REPAIR_EVENTS = 5
METRIC_RESTART_INDEX = 6
METRIC_BLOCKED_CANDIDATES = 7
METRIC_FORBIDDEN_COUNT = 8
METRIC_CANDIDATE_GENERATION_SECONDS = 9
METRIC_CANDIDATE_SCORING_SECONDS = 10
METRIC_RESTART_SORT_SECONDS = 11
METRIC_STATE_INIT_SECONDS = 12
METRIC_GREEDY_SCAN_SECONDS = 13
METRIC_FORBIDDEN_COUNT_SECONDS = 14
METRIC_C_RUN_SECONDS = 15
METRIC_MINIMUM_DISTANCE = 16
METRIC_COUNT = 17

SELECTED_CAP = 8192
ERROR_CAP = 4096
MAX_SUPPORTED_N = 24
COMPILE_TIMEOUT_SECONDS = 20
RUN_TIMEOUT_SECONDS = 600

BANNED_SOURCE_TOKENS = (
    "system(",
    "popen(",
    "fork(",
    "execv",
    "execl",
    "socket(",
    "connect(",
    "accept(",
    "fopen(",
    "freopen(",
    "remove(",
    "rename(",
)
BASELINE_C_PROGRAM = Path(__file__).resolve().with_name("initial_program.c")
FIXED_C_SEARCH_SKELETON = Path(__file__).resolve().with_name("c_search_skeleton.c")


@dataclass(frozen=True)
class CKernelCallResult:
    """Raw result returned by the isolated C-kernel process."""

    status: str
    return_code: int = -1
    selected: tuple[int, ...] = tuple()
    metrics: tuple[float, ...] = tuple()
    error: str = ""


def evaluate_c_program_path(program_path: str) -> EvaluationResult:
    """Evaluate a C priority function with the fixed maximum-code kernel."""
    instance = instance_from_env()
    started_at = time.perf_counter()
    program = Path(program_path)

    if instance.n > MAX_SUPPORTED_N:
        return _failure_result(
            "unsupported_instance",
            f"C kernel currently supports n <= {MAX_SUPPORTED_N}; got n={instance.n}",
            started_at,
        )

    source_error = _scan_source(program)
    if source_error:
        return _failure_result("source_rejected", source_error, started_at)

    with tempfile.TemporaryDirectory(prefix="max_code_c_kernel_") as tmp_dir:
        shared_object = Path(tmp_dir) / "kernel.so"
        compile_started_at = time.perf_counter()
        compile_error = _compile_c_kernel(program, shared_object)
        compile_seconds = time.perf_counter() - compile_started_at
        if compile_error:
            return _failure_result("compile_error", compile_error, started_at)

        seed = _env_seed()
        run_started_at = time.perf_counter()
        call_result = _run_kernel_with_timeout(
            shared_object,
            instance.n,
            instance.distance,
            instance.restarts,
            seed,
        )
        run_call_seconds = time.perf_counter() - run_started_at

    elapsed_seconds = time.perf_counter() - started_at
    if call_result.status != "ok":
        return _failure_result(call_result.status, call_result.error, started_at)

    metrics = call_result.metrics + (0.0,) * max(0, METRIC_COUNT - len(call_result.metrics))
    code_size = max(0, min(int(metrics[METRIC_CODE_SIZE]), SELECTED_CAP))
    selected = tuple(int(value) for value in call_result.selected[:code_size])
    valid = bool(call_result.return_code == 0 and metrics[METRIC_VALID] >= 0.5)
    if selected:
        valid = valid and validate_code(selected, instance.distance)
    minimum_distance = int(metrics[METRIC_MINIMUM_DISTANCE])
    combined_score = float(code_size if valid else 0)

    artifacts = _success_artifacts(
        instance_name=instance.name,
        n=instance.n,
        d=instance.distance,
        restarts=instance.restarts,
        selected=selected,
        valid=valid,
        metrics=metrics,
        c_return_code=call_result.return_code,
        c_error=call_result.error,
        compile_seconds=compile_seconds,
        run_call_seconds=run_call_seconds,
        evaluation_seconds=elapsed_seconds,
    )
    return EvaluationResult(
        metrics={
            "combined_score": combined_score,
            "code_size": float(code_size),
            "valid": float(valid),
            "minimum_distance": float(minimum_distance),
            "target_distance": float(instance.distance),
            "n": float(instance.n),
            "scored_candidates": metrics[METRIC_SCORED_CANDIDATES],
            "forbidden_count": metrics[METRIC_FORBIDDEN_COUNT],
            "repair_rollout_evaluations": metrics[METRIC_REPAIR_ROLLOUT_EVALUATIONS],
            "evaluation_time_seconds": elapsed_seconds,
        },
        artifacts=artifacts,
    )


def _scan_source(program_path: Path) -> str | None:
    try:
        source = program_path.read_text(encoding="utf-8")
    except OSError as exc:
        return f"failed to read C source: {exc}"
    for token in BANNED_SOURCE_TOKENS:
        if token in source:
            return f"C source uses banned token: {token}"
    return _validate_priority_only_source(source)


def _validate_priority_only_source(source: str) -> str | None:
    """Reject variants that modify code outside the priority EVOLVE-BLOCK."""
    try:
        baseline = BASELINE_C_PROGRAM.read_text(encoding="utf-8")
    except OSError as exc:
        return f"failed to read baseline C source: {exc}"

    source_protected, source_error = _protected_source(source)
    if source_error:
        return source_error
    baseline_protected, baseline_error = _protected_source(baseline)
    if baseline_error:
        return f"baseline C source is invalid: {baseline_error}"
    if source_protected != baseline_protected:
        return "C source changed code outside the oe_max_code_priority EVOLVE-BLOCK"
    return None


def _protected_source(source: str) -> tuple[str, str | None]:
    lines = source.splitlines(keepends=True)
    protected_lines: list[str] = []
    in_block = False
    start_count = 0
    end_count = 0

    for line in lines:
        if "# EVOLVE-BLOCK-START" in line:
            if in_block:
                return "", "nested EVOLVE-BLOCK markers are not supported"
            in_block = True
            start_count += 1
            protected_lines.append(line)
            continue
        if "# EVOLVE-BLOCK-END" in line:
            if not in_block:
                return "", "EVOLVE-BLOCK-END appears before EVOLVE-BLOCK-START"
            in_block = False
            end_count += 1
            protected_lines.append(line)
            continue
        if not in_block:
            protected_lines.append(line)

    if in_block:
        return "", "unterminated EVOLVE-BLOCK"
    if start_count != 1 or end_count != 1:
        return "", "C source must contain exactly one EVOLVE-BLOCK"
    return "".join(protected_lines), None


def _compile_c_kernel(program_path: Path, shared_object: Path) -> str | None:
    compiler = os.environ.get("CC", "gcc")
    command = [
        compiler,
        "-O3",
        "-std=c99",
        "-Wall",
        "-Wextra",
        "-fPIC",
        "-shared",
        "-pthread",
        str(program_path),
        str(FIXED_C_SEARCH_SKELETON),
        "-o",
        str(shared_object),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=_env_int("MAX_CODE_C_COMPILE_TIMEOUT", COMPILE_TIMEOUT_SECONDS),
        )
    except subprocess.TimeoutExpired:
        return "C kernel compilation timed out"
    except OSError as exc:
        return f"failed to execute C compiler: {exc}"
    if completed.returncode != 0:
        return (completed.stderr or completed.stdout or "C compiler failed").strip()
    return None


def _run_kernel_with_timeout(
    shared_object: Path,
    n: int,
    d: int,
    restarts: int,
    seed: int,
) -> CKernelCallResult:
    context = get_context("fork")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_call_kernel_worker,
        args=(str(shared_object), n, d, restarts, seed, result_queue),
    )
    process.start()
    timeout = _env_int("MAX_CODE_C_RUN_TIMEOUT", RUN_TIMEOUT_SECONDS)
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(2)
        return CKernelCallResult("runtime_timeout", error="C kernel execution timed out")
    try:
        payload = result_queue.get_nowait()
    except Empty:
        return CKernelCallResult(
            "runtime_error",
            error=f"C kernel process exited without a result, exitcode={process.exitcode}",
        )
    return CKernelCallResult(**payload)


def _call_kernel_worker(
    shared_object: str,
    n: int,
    d: int,
    restarts: int,
    seed: int,
    result_queue,
) -> None:
    try:
        library = ctypes.CDLL(shared_object)
        run_fn = library.oe_max_code_run
        run_fn.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_ulonglong,
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        run_fn.restype = ctypes.c_int
        getattr(library, "oe_max_code_priority")

        selected_out = (ctypes.c_ulonglong * SELECTED_CAP)()
        metrics_out = (ctypes.c_double * METRIC_COUNT)()
        error_out = ctypes.create_string_buffer(ERROR_CAP)
        return_code = int(
            run_fn(
                n,
                d,
                restarts,
                seed,
                selected_out,
                SELECTED_CAP,
                metrics_out,
                METRIC_COUNT,
                error_out,
                ERROR_CAP,
            )
        )
        selected_count = max(0, min(int(metrics_out[METRIC_CODE_SIZE]), SELECTED_CAP))
        payload = {
            "status": "ok",
            "return_code": return_code,
            "selected": tuple(int(selected_out[i]) for i in range(selected_count)),
            "metrics": tuple(float(metrics_out[i]) for i in range(METRIC_COUNT)),
            "error": error_out.value.decode("utf-8", errors="replace"),
        }
    except AttributeError as exc:
        payload = {"status": "missing_symbol", "error": str(exc)}
    except Exception as exc:
        payload = {"status": "runtime_error", "error": str(exc)}
    result_queue.put(payload)


def _success_artifacts(
    *,
    instance_name: str,
    n: int,
    d: int,
    restarts: int,
    selected: tuple[int, ...],
    valid: bool,
    metrics: Sequence[float],
    c_return_code: int,
    c_error: str,
    compile_seconds: float,
    run_call_seconds: float,
    evaluation_seconds: float,
) -> dict[str, str]:
    codewords = [format_word(mask, n) for mask in selected]
    search_result = {
        "search_mode": "c_kernel_mcts",
        "legality_engine": "c_kernel",
        "valid": valid,
        "code_size": int(metrics[METRIC_CODE_SIZE]),
        "minimum_distance": int(metrics[METRIC_MINIMUM_DISTANCE]),
        "candidate_count": int(metrics[METRIC_CANDIDATE_COUNT]),
        "scored_candidates": int(metrics[METRIC_SCORED_CANDIDATES]),
        "blocked_candidates": int(metrics[METRIC_BLOCKED_CANDIDATES]),
        "forbidden_count": int(metrics[METRIC_FORBIDDEN_COUNT]),
        "restart_index": int(metrics[METRIC_RESTART_INDEX]),
        "repair_mode": os.environ.get("MAX_CODE_REPAIR_MODE", "greedy"),
        "repair_moves": int(metrics[METRIC_REPAIR_EVENTS]),
        "repair_gain": 0,
        "repair_rollout_evaluations": int(metrics[METRIC_REPAIR_ROLLOUT_EVALUATIONS]),
        "repair_candidate_window": _env_int("MAX_CODE_REPAIR_CANDIDATE_WINDOW", 65536),
        "dynamic_window": _env_int("MAX_CODE_DYNAMIC_WINDOW", 4096),
        "repair_mcts_simulations": _env_int("MAX_CODE_REPAIR_MCTS_SIMULATIONS", 64),
        "repair_mcts_depth": _env_int("MAX_CODE_REPAIR_MCTS_DEPTH", 4),
        "repair_mcts_drop_topk": _env_int("MAX_CODE_REPAIR_MCTS_DROP_TOPK", 2),
        "repair_mcts_workers": _env_int_min("MAX_CODE_REPAIR_MCTS_WORKERS", 1, 0),
        "max_supported_n": MAX_SUPPORTED_N,
        "c_return_code": c_return_code,
        "c_error": c_error,
        "timing_seconds": {
            "compile": compile_seconds,
            "run_call": run_call_seconds,
            "evaluation_total": evaluation_seconds,
            "c_run_total": metrics[METRIC_C_RUN_SECONDS],
            "candidate_generation": metrics[METRIC_CANDIDATE_GENERATION_SECONDS],
            "candidate_scoring": metrics[METRIC_CANDIDATE_SCORING_SECONDS],
            "restart_sort_total": metrics[METRIC_RESTART_SORT_SECONDS],
            "state_init_total": metrics[METRIC_STATE_INIT_SECONDS],
            "greedy_scan_total": metrics[METRIC_GREEDY_SCAN_SECONDS],
            "forbidden_count_total": metrics[METRIC_FORBIDDEN_COUNT_SECONDS],
        },
    }
    return {
        "instance": json.dumps(
            {"name": instance_name, "n": n, "d": d, "restarts": restarts},
            sort_keys=True,
        ),
        "codewords": json.dumps(codewords),
        "search_result": json.dumps(search_result, sort_keys=True),
    }


def _failure_result(error_type: str, message: str, started_at: float) -> EvaluationResult:
    elapsed_seconds = time.perf_counter() - started_at
    instance = instance_from_env()
    artifacts = {
        "instance": json.dumps(
            {
                "name": instance.name,
                "n": instance.n,
                "d": instance.distance,
                "restarts": instance.restarts,
            },
            sort_keys=True,
        ),
        "codewords": json.dumps([]),
        "search_result": json.dumps(
            {
                "search_mode": "c_kernel_mcts",
                "legality_engine": "c_kernel",
                "error_type": error_type,
                "error": message,
                "valid": False,
                "code_size": 0,
                "minimum_distance": 0,
                "repair_mode": os.environ.get("MAX_CODE_REPAIR_MODE", "greedy"),
                "repair_moves": 0,
                "repair_gain": 0,
                "repair_rollout_evaluations": 0,
            },
            sort_keys=True,
        ),
        "c_kernel_error": message,
    }
    return EvaluationResult(
        metrics={
            "combined_score": 0.0,
            "code_size": 0.0,
            "valid": 0.0,
            "minimum_distance": 0.0,
            "target_distance": float(instance.distance),
            "n": float(instance.n),
            "scored_candidates": 0.0,
            "forbidden_count": 0.0,
            "repair_rollout_evaluations": 0.0,
            "evaluation_time_seconds": elapsed_seconds,
        },
        artifacts=artifacts,
    )


def _env_seed() -> int:
    raw_value = os.environ.get("MAX_CODE_RANDOM_SEED", "0")
    try:
        return max(int(raw_value), 0)
    except ValueError:
        return 0


def _env_int(name: str, default: int) -> int:
    return _env_int_min(name, default, 1)


def _env_int_min(name: str, default: int, minimum: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return max(int(raw_value), minimum)
    except ValueError:
        return default
