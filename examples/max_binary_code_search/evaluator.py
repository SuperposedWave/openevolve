"""OpenEvolve evaluator for binary maximum-code search."""

from pathlib import Path

from c_kernel_runner import evaluate_c_program_path
from search_core import evaluate_program_path


def evaluate(program_path: str):
    """Evaluate an evolved priority function."""
    if Path(program_path).suffix.lower() == ".c":
        return evaluate_c_program_path(program_path)
    return evaluate_program_path(program_path)
