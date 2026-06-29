"""OpenEvolve evaluator for binary maximum-code search."""

from search_core import evaluate_program_path


def evaluate(program_path: str):
    """Evaluate an evolved priority function."""
    return evaluate_program_path(program_path)
