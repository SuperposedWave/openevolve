"""OpenEvolve evaluator for binary linear-code feasibility search."""

from search_core import evaluate_program_path


def evaluate(program_path: str):
    """Evaluate an evolved priority function against the fixed benchmark suite."""
    return evaluate_program_path(program_path)
