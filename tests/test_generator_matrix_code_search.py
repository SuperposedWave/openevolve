"""Tests for the generator-matrix binary linear-code experiment."""

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "examples" / "generator_matrix_code_search"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class TestGeneratorMatrixCodeSearch(unittest.TestCase):
    """Regression coverage for the G=[I_k|P] deficit-multicover skeleton."""

    @classmethod
    def setUpClass(cls):
        cls.search = _load_module(
            "generator_matrix_search_core",
            EXAMPLE_DIR / "search_core.py",
        )
        cls.evaluator = _load_module(
            "generator_matrix_evaluator",
            EXAMPLE_DIR / "evaluator.py",
        )
        cls.initial_program = _load_module(
            "generator_matrix_initial_program",
            EXAMPLE_DIR / "initial_program.py",
        )

    def test_deficits_exclude_zero_message(self):
        """The all-zero message must not become an impossible multicover constraint."""
        deficits = self.search.initial_deficits(k=4, d=3)
        self.assertEqual(int(deficits[0]), 0)
        self.assertEqual(int(deficits[1]), 2)
        self.assertEqual(int(deficits[0b111]), 0)

    def test_exact_distance_matches_hand_construction(self):
        """A small G=[I|P] construction should verify by exhaustive messages."""
        columns = (0b01, 0b10, 0b11)
        self.assertEqual(
            self.search.exact_minimum_distance(k=2, parity_columns=columns),
            3,
        )

    def test_search_builds_orthogonal_systematic_matrices(self):
        """The column-fill search should output compatible G and H rows."""
        instance = self.search.make_instance(5, 2, 3)
        config = self.search.ColumnSearchConfig(shortlist_size=8)
        result = self.search.search_generator_columns(instance, config)

        self.assertTrue(result.success)
        self.assertEqual(result.d_actual, 3)
        g_rows = self.search.generator_matrix_rows(instance.k, result.columns)
        h_rows = self.search.parity_check_matrix_rows(instance.k, result.columns)
        for g_row in g_rows:
            for h_row in h_rows:
                overlap = sum(
                    (int(g_bit) & int(h_bit))
                    for g_bit, h_bit in zip(g_row, h_row)
                ) % 2
                self.assertEqual(overlap, 0)

    def test_evaluator_runs_baseline_priority(self):
        """The example should expose a normal OpenEvolve evaluator entry point."""
        result = self.evaluator.evaluate(str(EXAMPLE_DIR / "initial_program.py"))
        self.assertGreaterEqual(result.metrics["combined_score"], 0.0)
        self.assertIn("search_result", result.artifacts)
        self.assertIn("generator_matrix", result.artifacts)
        self.assertTrue(callable(self.initial_program.get_priority_function()))

    def test_combined_score_is_column_progress_first(self):
        """One extra feasible column should dominate same-column tie-break metrics."""
        instance = self.search.make_instance(8, 4, 4)
        config = self.search.ColumnSearchConfig()
        weaker_more_columns = self.search.ColumnSearchResult(
            instance=instance,
            config=config,
            success=False,
            columns=(1, 2),
            column_bits=("0001", "0010"),
            row_weights=(1, 1, 0, 0),
            unsatisfied_count=20,
            remaining_deficit_sum=100,
            min_margin=-4,
            d_actual=1,
            step_records=tuple(),
            restart_index=0,
            candidate_scoring_time=0.0,
            exact_verification_time=0.0,
            total_time=0.0,
        )
        prettier_fewer_columns = self.search.ColumnSearchResult(
            instance=instance,
            config=config,
            success=False,
            columns=(1,),
            column_bits=("0001",),
            row_weights=(1, 0, 0, 0),
            unsatisfied_count=1,
            remaining_deficit_sum=1,
            min_margin=-1,
            d_actual=3,
            step_records=tuple(),
            restart_index=0,
            candidate_scoring_time=0.0,
            exact_verification_time=0.0,
            total_time=0.0,
        )

        self.assertGreater(
            self.search.metrics_from_result(weaker_more_columns)["combined_score"],
            self.search.metrics_from_result(prettier_fewer_columns)["combined_score"],
        )


if __name__ == "__main__":
    unittest.main()
