"""Tests for shared utilities around the C-only binary linear-code example."""

import importlib.util
import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout
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


class TestLinearCodeBinarySearchUtilities(unittest.TestCase):
    """Regression coverage for the C-only linear-code example wrapper."""

    @classmethod
    def setUpClass(cls):
        cls.search_core = _load_module("linear_code_search_core", EXAMPLE_DIR / "search_core.py")
        cls.evaluator = _load_module("linear_code_evaluator", EXAMPLE_DIR / "evaluator.py")
        cls.run_batch = _load_module("linear_code_run_batch", EXAMPLE_DIR / "run_batch.py")
        cls.verify_distance = _load_module(
            "linear_code_verify_distance",
            EXAMPLE_DIR / "verify_distance.py",
        )

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
        """Incremental membership checks must match independent brute-force validation."""
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
        """Incremental xor-layer updates should agree with full recomputation."""
        r = 5
        distance = 4
        selected = (7, 11, 13)
        state = self.search_core.IncrementalForbiddenState(r, distance)
        for column_mask in selected:
            state.add(column_mask)

        rebuilt = self.search_core.rebuild_reachable_layers(r, distance, selected)
        self.assertEqual(state.reachable, rebuilt)
        self.assertEqual(state.forbidden, self.search_core.forbidden_masks_from_layers(rebuilt))

    def test_candidate_count_helpers_match_layers(self):
        """Candidate counts should still be available for C runner metadata checks."""
        layers = self.search_core.candidate_weight_layer_counts(r=6, distance=4)
        self.assertEqual(layers, ((3, 20), (4, 15), (5, 6), (6, 1)))
        self.assertEqual(
            self.search_core.candidate_count(6, 4),
            sum(count for _, count in layers),
        )

    def test_generator_matrix_is_orthogonal_to_parity_check_matrix(self):
        """Generated G = [I_k | P] should satisfy G H^T = 0 over F_2."""
        r = 3
        free_columns = (0b011, 0b101, 0b110, 0b111)
        g_rows = self.search_core.generator_matrix_rows(r, free_columns)
        h_rows = self.search_core.parity_check_matrix_rows(r, free_columns)
        for g_row in g_rows:
            for h_row in h_rows:
                overlap = sum(
                    (int(g_bit) & int(h_bit))
                    for g_bit, h_bit in zip(g_row, h_row)
                ) % 2
                self.assertEqual(overlap, 0)

    def test_evaluator_rejects_python_programs(self):
        """The linear-code evaluator should no longer run Python priority programs."""
        result = self.evaluator.evaluate(str(EXAMPLE_DIR / "missing_python_path.py"))
        self.assertEqual(result.metrics["combined_score"], 0.0)
        search_result = json.loads(result.artifacts["search_result"])
        self.assertEqual(search_result["error_type"], "unsupported_program_type")

    def test_load_tasks_from_record_filters_to_n_window_and_valid_k(self):
        """Batch loading should keep only n_min <= n <= n_max and 1 <= k < n."""
        tasks, skipped = self.run_batch.load_tasks_from_record(
            EXAMPLE_DIR / "Misc" / "ECCRecord.json",
            n_min=11,
            n_max=5,
            d_field="lower",
        )
        self.assertEqual(tasks, [])
        self.assertEqual(skipped, [])

        tasks, skipped = self.run_batch.load_tasks_from_record(
            EXAMPLE_DIR / "Misc" / "ECCRecord.json",
            n_min=11,
            n_max=12,
            d_field="lower",
        )
        self.assertTrue(tasks)
        self.assertTrue(all(11 <= task.n <= 12 for task in tasks))
        self.assertTrue(all(1 <= task.k < task.n for task in tasks))
        self.assertTrue(skipped)
        self.assertTrue(any(row["k"] == row["n"] for row in skipped))

    def test_resolved_config_injects_current_target(self):
        """Per-instance configs should rewrite the prompt header for the active instance."""
        config_path = EXAMPLE_DIR / "Configs" / "config_c_kernel.yaml"
        config_text = config_path.read_text()
        task = self.run_batch.SweepTask(n=18, k=7, d=7, lower=7, upper=8)
        resolved = self.run_batch.render_resolved_config(config_text, task, config_path)
        self.assertIn("- n = 18", resolved)
        self.assertIn("- k = 7", resolved)
        self.assertIn("- d = 7", resolved)
        self.assertIn("- r = n - k = 11", resolved)
        self.assertNotIn("- n = 38", resolved)

    def test_parse_verification_output_distinguishes_complete_and_partial(self):
        """Verification parsing should classify complete and partial constructions."""
        complete = self.run_batch.parse_verification_output("d_actual: 7\nH shape: 11 x 18\n")
        partial = self.run_batch.parse_verification_output("d_partial: 9\nwarning: construction is incomplete\n")
        self.assertEqual(complete["verification_status"], "complete")
        self.assertEqual(complete["distance"], 7)
        self.assertEqual(partial["verification_status"], "partial")
        self.assertEqual(partial["distance"], 9)

    def test_instance_name_is_stable_for_output_directories(self):
        """Per-instance directories should be named by n/k/d and stay deterministic."""
        task = self.run_batch.SweepTask(n=20, k=7, d=8, lower=8, upper=9)
        self.assertEqual(task.instance_name, "n20_k7_d8")

    def test_batch_defaults_use_c_entrypoint(self):
        """The batch runner should no longer enqueue Python initial programs."""
        self.assertEqual(self.run_batch.DEFAULT_CONFIG.name, "config_c_kernel.yaml")
        task = self.run_batch.SweepTask(n=8, k=4, d=4, lower=4, upper=4)

        with mock.patch.object(self.run_batch, "write_resolved_config") as write_config:
            with mock.patch.object(self.run_batch.subprocess, "run") as run:
                with mock.patch.object(self.run_batch, "write_json"):
                    write_config.return_value = EXAMPLE_DIR / "Configs" / "config_c_kernel.yaml"
                    run.return_value = mock.Mock(returncode=1, stdout="", stderr="")
                    row = self.run_batch.run_instance(
                        task,
                        EXAMPLE_DIR / "Configs" / "config_c_kernel.yaml",
                        Path(os.environ.get("TMPDIR", "/tmp")) / "linear_code_test_output",
                        iterations=1,
                        force=True,
                    )

        command = run.call_args_list[0].args[0]
        self.assertIn("initial_program.c", command)
        self.assertNotIn("initial_program.py", command)
        self.assertEqual(row["status"], "failed")

    def test_verify_distance_accepts_cli_instance_arguments(self):
        """Verification CLI should accept --N/--K/--D to override instance values."""
        program_path = str(EXAMPLE_DIR / "initial_program.c")
        stdout = io.StringIO()
        argv = [
            "verify_distance.py",
            program_path,
            "--no-progress",
            "--N",
            "7",
            "--K",
            "4",
            "--D",
            "3",
        ]
        with mock.patch.object(sys, "argv", argv):
            with redirect_stdout(stdout):
                self.verify_distance.main()
        output = stdout.getvalue()
        self.assertIn('"n": 7', output)
        self.assertIn('"k": 4', output)
        self.assertIn('"d_target": 3', output)
        self.assertIn("search_mode", output)


if __name__ == "__main__":
    unittest.main()
