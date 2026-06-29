"""Tests for C priority-function evaluation in the linear-code example."""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "examples" / "linear_code_binary_search"
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


class TestLinearCodeCKernel(unittest.TestCase):
    """C priority files should compile and run with the fixed skeleton."""

    @classmethod
    def setUpClass(cls):
        cls.search_core = _load_module(
            "linear_code_search_core_c_kernel_tests",
            EXAMPLE_DIR / "search_core.py",
        )
        cls.evaluator = _load_module(
            "linear_code_evaluator_c_kernel_tests",
            EXAMPLE_DIR / "evaluator.py",
        )

    def test_c_baseline_solves_default_instance(self):
        """The C baseline should solve the default instance and expose viewer artifacts."""
        result = self.evaluator.evaluate(str(EXAMPLE_DIR / "initial_program.c"))

        self.assertEqual(result.metrics["success_rate"], 1.0)
        self.assertEqual(result.metrics["combined_score"], 1.0)
        search_result = json.loads(result.artifacts["search_result"])
        self.assertEqual(search_result["search_mode"], "c_kernel")
        self.assertEqual(search_result["target_free_columns"], 4)
        selected = tuple(int(bits, 2) for bits in search_result["selected_free_columns"])
        self.assertTrue(self.search_core.validate_free_columns(4, selected, 4))

    def test_c_baseline_is_deterministic_for_same_seed(self):
        """A fixed seed and instance should produce identical selected columns."""
        env = {
            "LINEAR_CODE_N": "8",
            "LINEAR_CODE_K": "4",
            "LINEAR_CODE_D": "4",
            "LINEAR_CODE_RESTARTS": "3",
            "LINEAR_CODE_RANDOM_SEED": "7",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            first = self.evaluator.evaluate(str(EXAMPLE_DIR / "initial_program.c"))
            second = self.evaluator.evaluate(str(EXAMPLE_DIR / "initial_program.c"))

        first_search = json.loads(first.artifacts["search_result"])
        second_search = json.loads(second.artifacts["search_result"])
        self.assertEqual(first_search["selected_free_columns"], second_search["selected_free_columns"])
        self.assertEqual(first.metrics["combined_score"], second.metrics["combined_score"])

    def test_mcts_repair_mode_can_recover_from_tiny_dynamic_window(self):
        """MCTS repair should be callable and able to repair a stuck local fill."""
        env = {
            "LINEAR_CODE_N": "16",
            "LINEAR_CODE_K": "8",
            "LINEAR_CODE_D": "5",
            "LINEAR_CODE_RESTARTS": "1",
            "LINEAR_CODE_DYNAMIC_WINDOW": "1",
            "LINEAR_CODE_REPAIR_EVENTS": "8",
            "LINEAR_CODE_REPAIR_MODE": "mcts",
            "LINEAR_CODE_REPAIR_MCTS_SIMULATIONS": "16",
            "LINEAR_CODE_REPAIR_MCTS_DEPTH": "3",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            result = self.evaluator.evaluate(str(EXAMPLE_DIR / "initial_program.c"))

        self.assertEqual(result.metrics["success_rate"], 1.0)
        search_result = json.loads(result.artifacts["search_result"])
        self.assertEqual(search_result["repair_mode"], "mcts")
        self.assertGreater(search_result["backtrack_events"], 0)
        selected = tuple(int(bits, 2) for bits in search_result["selected_free_columns"])
        self.assertTrue(self.search_core.validate_free_columns(8, selected, 5))

    def test_missing_required_symbols_returns_low_score(self):
        """A C file without the fixed priority block should fail cleanly."""
        source = "double unrelated(void) { return 0.0; }\n"
        with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as handle:
            handle.write(source)
            path = handle.name
        try:
            result = self.evaluator.evaluate(path)
        finally:
            os.unlink(path)

        self.assertEqual(result.metrics["combined_score"], 0.0)
        search_result = json.loads(result.artifacts["search_result"])
        self.assertEqual(search_result["error_type"], "source_rejected")

    def test_changes_outside_priority_block_are_rejected(self):
        """The C evaluator should keep the priority file wrapper fixed."""
        source = (EXAMPLE_DIR / "initial_program.c").read_text()
        source = source.replace(
            "Baseline static priority heuristic",
            "Altered static priority heuristic",
        )
        with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as handle:
            handle.write(source)
            path = handle.name
        try:
            result = self.evaluator.evaluate(path)
        finally:
            os.unlink(path)

        self.assertEqual(result.metrics["combined_score"], 0.0)
        self.assertEqual(result.metrics["success_rate"], 0.0)
        search_result = json.loads(result.artifacts["search_result"])
        self.assertEqual(search_result["error_type"], "source_rejected")

    def test_priority_block_change_is_accepted(self):
        """A variant that changes only priority logic should still be evaluated."""
        source = (EXAMPLE_DIR / "initial_program.c").read_text()
        source = source.replace(
            "return 1.2 * damage_score",
            "return 1.1 * damage_score",
        )
        with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as handle:
            handle.write(source)
            path = handle.name
        try:
            result = self.evaluator.evaluate(path)
        finally:
            os.unlink(path)

        search_result = json.loads(result.artifacts["search_result"])
        self.assertNotIn("error_type", search_result)


if __name__ == "__main__":
    unittest.main()
