"""OpenEvolve evaluator for C-kernel binary linear-code feasibility search."""

from pathlib import Path
import time

from c_kernel_runner import _failure_result, evaluate_c_program_path


def evaluate(program_path: str):
    """Evaluate an evolved C priority function with the fixed C search skeleton."""
    if Path(program_path).suffix.lower() != ".c":
        return _failure_result(
            "unsupported_program_type",
            "linear_code_binary_search is C-kernel-only; provide a .c priority file",
            time.perf_counter(),
        )
    return evaluate_c_program_path(program_path)
