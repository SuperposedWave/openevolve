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

    def test_baseline_priority_builds_valid_codes(self):
        """The baseline priority heuristic should solve at least part of the suite."""
        result = self.search_core.evaluate_priority_function(
            self.initial_program.get_priority_function()
        )
        self.assertGreater(result.metrics["combined_score"], 0.0)
        self.assertGreaterEqual(result.metrics["solved_instances"], 2)

        summaries = json.loads(result.artifacts["benchmark_summaries"])
        for summary in summaries:
            if summary["success"]:
                instance = next(
                    benchmark
                    for benchmark in self.search_core.BENCHMARKS
                    if benchmark.name == summary["name"]
                )
                selected = tuple(int(bits, 2) for bits in summary["selected_free_columns"])
                self.assertTrue(
                    self.search_core.validate_free_columns(
                        instance.r,
                        selected,
                        instance.target_distance,
                    )
                )

    def test_evaluator_is_deterministic(self):
        """The evaluator should produce stable metrics for the same program."""
        program_path = str(EXAMPLE_DIR / "initial_program.py")
        first = self.evaluator.evaluate(program_path)
        second = self.evaluator.evaluate(program_path)
        self.assertEqual(first.metrics, second.metrics)
        self.assertEqual(first.artifacts, second.artifacts)


if __name__ == "__main__":
    unittest.main()
