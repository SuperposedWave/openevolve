"""Tests for the binary linear-code feasibility example."""

import importlib.util
import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "examples" / "linear_code_binary_search"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class TestLinearCodeBinarySearch(unittest.TestCase):
    """Regression coverage for the new linear-code example."""

    @classmethod
    def setUpClass(cls):
        cls.search_core = _load_module("linear_code_search_core", EXAMPLE_DIR / "search_core.py")
        cls.initial_program = _load_module(
            "linear_code_initial_program",
            EXAMPLE_DIR / "initial_program.py",
        )
        cls.evaluator = _load_module("linear_code_evaluator", EXAMPLE_DIR / "evaluator.py")

    def test_initial_forbidden_matches_weight_threshold(self):
        """Systematic initialization should match the exact low-weight forbidden set."""
        r = 5
        distance = 4
        forbidden = self.search_core.initial_forbidden_masks(r, distance)
        expected = {
            mask
            for mask in range(1 << r)
            if self.search_core.popcount(mask) <= distance - 2
        }
        self.assertEqual(forbidden, expected)

    def test_can_add_matches_bruteforce_validation(self):
        """Incremental membership checks must match an independent brute-force test."""
        r = 4
        distance = 4
        selected = (7, 11)
        state = self.search_core.IncrementalForbiddenState(r, distance)
        for column_mask in selected:
            state.add(column_mask)

        for candidate_mask in self.search_core.candidate_masks(r, distance):
            expected = self.search_core.validate_free_columns(
                r,
                selected + (candidate_mask,),
                distance,
            )
            self.assertEqual(state.can_add(candidate_mask), expected)

    def test_reachable_update_matches_rebuild(self):
        """Incremental xor-layer updates must agree with a full recomputation."""
        r = 5
        distance = 4
        selected = (7, 11, 13)
        state = self.search_core.IncrementalForbiddenState(r, distance)
        for column_mask in selected:
            state.add(column_mask)

        rebuilt = self.search_core.rebuild_reachable_layers(r, distance, selected)
        self.assertEqual(state.reachable, rebuilt)
        self.assertEqual(state.forbidden, self.search_core.forbidden_masks_from_layers(rebuilt))

    def test_baseline_priority_builds_valid_code_on_default_instance(self):
        """The baseline priority heuristic should solve the default single instance."""
        result = self.search_core.evaluate_priority_function(self.initial_program.get_priority_function())
        self.assertGreater(result.metrics["combined_score"], 0.0)
        self.assertEqual(result.metrics["success_rate"], 1.0)

        search_result = json.loads(result.artifacts["search_result"])
        selected = tuple(int(bits, 2) for bits in search_result["selected_free_columns"])
        instance = self.search_core.DEFAULT_INSTANCE
        self.assertTrue(
            self.search_core.validate_free_columns(
                instance.r,
                selected,
                instance.target_distance,
            )
        )

    def test_static_priority_signature_is_supported(self):
        """The evaluator should accept a static priority(column_mask, n, k, d) interface."""
        priority_fn = self.initial_program.get_priority_function()
        score = priority_fn(0b1110, 8, 4, 4)
        self.assertIsInstance(score, (int, float))

    def test_candidate_ranking_is_deterministic(self):
        """The full candidate ordering should be stable for a fixed instance."""
        instance = self.search_core.make_instance(n=8, k=4, distance=4, restarts=1)
        priority_fn = self.initial_program.get_priority_function()
        first_order, first_scores = self.search_core.ranked_candidates(instance, priority_fn, 0)
        second_order, second_scores = self.search_core.ranked_candidates(instance, priority_fn, 0)
        self.assertEqual(first_order, second_order)
        self.assertEqual(first_scores, second_scores)

    def test_custom_instance_interface(self):
        """A caller should be able to set one custom (n, k, d) instance directly."""
        instance = self.search_core.make_instance(n=7, k=4, distance=3, restarts=2)
        result = self.search_core.evaluate_priority_function(
            self.initial_program.get_priority_function(),
            instance,
        )
        self.assertEqual(result.metrics["n"], 7)
        self.assertEqual(result.metrics["k"], 4)
        self.assertEqual(result.metrics["target_distance"], 3)
        self.assertGreater(result.metrics["combined_score"], 0.0)
        self.assertIn("top_ranked_columns", result.artifacts)

    def test_actual_minimum_distance_matches_target_for_valid_construction(self):
        """The exact d computed from constructed columns should match the target on success."""
        instance = self.search_core.make_instance(n=7, k=4, distance=3, restarts=1)
        attempt = self.search_core.best_restart_for_instance(
            instance,
            self.initial_program.get_priority_function(),
        )
        self.assertTrue(attempt.success)
        self.assertEqual(
            self.search_core.actual_minimum_distance(instance.r, attempt.selected_free_columns),
            instance.target_distance,
        )

    def test_generator_matrix_is_orthogonal_to_parity_check_matrix(self):
        """Generated G = [I_k | P] should satisfy G H^T = 0 over F_2."""
        instance = self.search_core.make_instance(n=7, k=4, distance=3, restarts=1)
        attempt = self.search_core.best_restart_for_instance(
            instance,
            self.initial_program.get_priority_function(),
        )
        g_rows = self.search_core.generator_matrix_rows(instance.r, attempt.selected_free_columns)
        h_rows = self.search_core.parity_check_matrix_rows(instance.r, attempt.selected_free_columns)
        for g_row in g_rows:
            for h_row in h_rows:
                overlap = sum(
                    (int(g_bit) & int(h_bit))
                    for g_bit, h_bit in zip(g_row, h_row)
                ) % 2
                self.assertEqual(overlap, 0)

    def test_evaluator_is_deterministic(self):
        """The evaluator should produce stable metrics for the same program."""
        program_path = str(EXAMPLE_DIR / "initial_program.py")
        first = self.evaluator.evaluate(program_path)
        second = self.evaluator.evaluate(program_path)
        self.assertEqual(first.metrics, second.metrics)
        self.assertEqual(first.artifacts, second.artifacts)


if __name__ == "__main__":
    unittest.main()
