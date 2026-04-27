"""Tests for the binary linear-code feasibility example."""

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
        cls.run_batch = _load_module(
            "linear_code_run_batch",
            EXAMPLE_DIR / "run_batch.py",
        )
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

    def test_lazy_can_add_matches_bruteforce_validation(self):
        """Lazy legality checks should match brute-force validation without eager masks."""
        r = 5
        distance = 4
        selected = (7, 11, 13)
        with mock.patch.dict(os.environ, {"LINEAR_CODE_EAGER_FORBIDDEN_LIMIT": "1"}, clear=False):
            state = self.search_core.IncrementalForbiddenState(r, distance)
            self.assertFalse(state.use_eager_forbidden)
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
        self.assertIn("evaluation_time_seconds", result.metrics)
        self.assertGreaterEqual(result.metrics["evaluation_time_seconds"], 0.0)

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

    def test_streaming_candidate_path_matches_materialized_search_on_small_instance(self):
        """Forced chunked processing should preserve exact greedy results on small instances."""
        instance = self.search_core.make_instance(n=8, k=4, distance=4, restarts=1)
        priority_fn = self.initial_program.get_priority_function()

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LINEAR_CODE_FORCE_STREAMING", None)
            baseline = self.search_core.best_restart_for_instance(instance, priority_fn)

        with mock.patch.dict(
            os.environ,
            {
                "LINEAR_CODE_FORCE_STREAMING": "1",
                "LINEAR_CODE_CANDIDATE_CHUNK_SIZE": "3",
            },
            clear=False,
        ):
            streamed = self.search_core.best_restart_for_instance(instance, priority_fn)

        self.assertEqual(streamed.success, baseline.success)
        self.assertEqual(streamed.selected_free_columns, baseline.selected_free_columns)
        self.assertEqual(streamed.added_free_columns, baseline.added_free_columns)
        self.assertEqual(streamed.blocked_candidate_count, baseline.blocked_candidate_count)
        self.assertEqual(streamed.chosen_weights, baseline.chosen_weights)
        self.assertEqual(streamed.candidate_count, baseline.candidate_count)
        self.assertEqual(streamed.sorted_scores[:10], baseline.sorted_scores[:10])

    def test_process_scored_streaming_matches_baseline_on_small_instance(self):
        """Optional process-based chunk scoring should preserve exact greedy results."""
        instance = self.search_core.make_instance(n=8, k=4, distance=4, restarts=1)
        priority_fn = self.initial_program.get_priority_function()

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LINEAR_CODE_FORCE_STREAMING", None)
            os.environ.pop("LINEAR_CODE_CANDIDATE_EXECUTOR", None)
            baseline = self.search_core.best_restart_for_instance(instance, priority_fn)

        with mock.patch.dict(
            os.environ,
            {
                "LINEAR_CODE_FORCE_STREAMING": "1",
                "LINEAR_CODE_CANDIDATE_CHUNK_SIZE": "3",
                "LINEAR_CODE_CANDIDATE_EXECUTOR": "process",
                "LINEAR_CODE_CANDIDATE_WORKERS": "2",
                "LINEAR_CODE_PROFILE": "1",
            },
            clear=False,
        ):
            with mock.patch.object(self.search_core.logger, "info") as mock_info:
                processed = self.search_core.best_restart_for_instance(instance, priority_fn)

        self.assertEqual(processed.success, baseline.success)
        self.assertEqual(processed.selected_free_columns, baseline.selected_free_columns)
        self.assertEqual(processed.added_free_columns, baseline.added_free_columns)
        self.assertEqual(processed.blocked_candidate_count, baseline.blocked_candidate_count)
        self.assertEqual(processed.chosen_weights, baseline.chosen_weights)
        self.assertEqual(processed.candidate_count, baseline.candidate_count)
        self.assertEqual(processed.sorted_scores[:10], baseline.sorted_scores[:10])
        profile_messages = [
            call.args[0]
            for call in mock_info.call_args_list
            if call.args and "linear_code_profile" in call.args[0]
        ]
        self.assertTrue(
            any(
                "stage=candidate_scoring" in message and "executor_mode=" in message
                for message in profile_messages
            )
        )

    def test_sampled_search_is_deterministic_for_fixed_seed(self):
        """Sampled mode should be reproducible for the same instance, restart, and seed."""
        instance = self.search_core.make_instance(n=8, k=4, distance=4, restarts=1)
        priority_fn = self.initial_program.get_priority_function()

        sampled_env = {
            "LINEAR_CODE_SEARCH_MODE": "sampled",
            "LINEAR_CODE_SAMPLE_BUDGET": "4",
            "LINEAR_CODE_SAMPLE_SEED": "123",
            "LINEAR_CODE_SAMPLE_OVERSAMPLE_FACTOR": "1",
        }
        with mock.patch.dict(os.environ, sampled_env, clear=False):
            first = self.search_core.best_restart_for_instance(instance, priority_fn)
            second = self.search_core.best_restart_for_instance(instance, priority_fn)

        self.assertEqual(first.selected_free_columns, second.selected_free_columns)
        self.assertEqual(first.sorted_scores, second.sorted_scores)
        self.assertEqual(first.search_mode, "sampled")
        self.assertEqual(first.sample_budget, 4)
        self.assertLessEqual(first.candidate_count, 4)

    def test_sampled_search_artifacts_report_sampling_metadata(self):
        """Evaluation artifacts should expose sampled-mode budget and seed details."""
        instance = self.search_core.make_instance(n=10, k=5, distance=3, restarts=1)
        sampled_env = {
            "LINEAR_CODE_SEARCH_MODE": "sampled",
            "LINEAR_CODE_SAMPLE_BUDGET": "10",
            "LINEAR_CODE_SAMPLE_SEED": "7",
            "LINEAR_CODE_SAMPLE_OVERSAMPLE_FACTOR": "2",
        }
        with mock.patch.dict(os.environ, sampled_env, clear=False):
            result = self.search_core.evaluate_priority_function(
                self.initial_program.get_priority_function(),
                instance,
            )

        search_result = json.loads(result.artifacts["search_result"])
        selected = tuple(int(bits, 2) for bits in search_result["selected_free_columns"])
        self.assertEqual(search_result["search_mode"], "sampled")
        self.assertEqual(search_result["sample_budget"], 10)
        self.assertIsInstance(search_result["sample_seed"], int)
        self.assertGreaterEqual(
            search_result["unique_sampled_candidates"],
            search_result["candidate_count"],
        )
        self.assertLessEqual(search_result["candidate_count"], 10)
        self.assertTrue(
            self.search_core.validate_free_columns(
                instance.r,
                selected,
                instance.target_distance,
            )
        )

    def test_sampled_restart_index_changes_candidate_pool(self):
        """Different restart indices should explore different deterministic samples."""
        instance = self.search_core.make_instance(n=14, k=4, distance=4, restarts=2)
        sampled_env = {
            "LINEAR_CODE_SEARCH_MODE": "sampled",
            "LINEAR_CODE_SAMPLE_BUDGET": "20",
            "LINEAR_CODE_SAMPLE_SEED": "99",
            "LINEAR_CODE_SAMPLE_OVERSAMPLE_FACTOR": "1",
        }
        with mock.patch.dict(os.environ, sampled_env, clear=False):
            first, first_seed = self.search_core.stratified_sampled_candidates(instance, 0)
            first_again, first_seed_again = self.search_core.stratified_sampled_candidates(
                instance,
                0,
            )
            second, second_seed = self.search_core.stratified_sampled_candidates(instance, 1)

        self.assertEqual(first, first_again)
        self.assertEqual(first_seed, first_seed_again)
        self.assertNotEqual(first, second)
        self.assertNotEqual(first_seed, second_seed)

    def test_final_validation_can_be_skipped_for_large_searches(self):
        """The greedy state can certify success without redundant exhaustive validation."""
        instance = self.search_core.make_instance(n=8, k=4, distance=4, restarts=1)
        with mock.patch.dict(os.environ, {"LINEAR_CODE_SKIP_FINAL_VALIDATION": "1"}, clear=False):
            with mock.patch.object(
                self.search_core,
                "validate_free_columns",
                side_effect=AssertionError("validation should be skipped"),
            ):
                attempt = self.search_core.best_restart_for_instance(
                    instance,
                    self.initial_program.get_priority_function(),
                )

        self.assertTrue(attempt.success)

    def test_stage_profile_logging_is_disabled_by_default(self):
        """Profiling logs should stay silent unless explicitly enabled."""
        instance = self.search_core.make_instance(n=8, k=4, distance=4, restarts=1)
        priority_fn = self.initial_program.get_priority_function()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LINEAR_CODE_PROFILE", None)
            with mock.patch.object(self.search_core.logger, "info") as mock_info:
                self.search_core.evaluate_priority_function(priority_fn, instance)
        profile_messages = [
            call.args[0]
            for call in mock_info.call_args_list
            if call.args and "linear_code_profile" in call.args[0]
        ]
        self.assertEqual(profile_messages, [])

    def test_stage_profile_logging_reports_major_search_phases(self):
        """Opt-in profiling should log candidate generation, sorting, and greedy scan stages."""
        instance = self.search_core.make_instance(n=8, k=4, distance=4, restarts=1)
        priority_fn = self.initial_program.get_priority_function()
        with mock.patch.dict(os.environ, {"LINEAR_CODE_PROFILE": "1"}, clear=False):
            with mock.patch.object(self.search_core.logger, "info") as mock_info:
                self.search_core.evaluate_priority_function(priority_fn, instance)
        profile_messages = [
            call.args[0]
            for call in mock_info.call_args_list
            if call.args and "linear_code_profile" in call.args[0]
        ]
        self.assertTrue(
            any("stage=candidate_generation" in message for message in profile_messages)
        )
        self.assertTrue(
            any("stage=candidate_scoring" in message for message in profile_messages)
        )
        self.assertTrue(
            any("stage=candidate_sort" in message for message in profile_messages)
        )
        self.assertTrue(
            any("stage=greedy_scan" in message for message in profile_messages)
        )
        self.assertTrue(
            any("stage=evaluation_summary" in message for message in profile_messages)
        )

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
        self.assertIn("evaluation_time_seconds", result.metrics)
        self.assertGreaterEqual(result.metrics["evaluation_time_seconds"], 0.0)
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
        self.assertIn("evaluation_time_seconds", first.metrics)
        self.assertIn("evaluation_time_seconds", second.metrics)
        comparable_first = {
            key: value
            for key, value in first.metrics.items()
            if key != "evaluation_time_seconds"
        }
        comparable_second = {
            key: value
            for key, value in second.metrics.items()
            if key != "evaluation_time_seconds"
        }
        self.assertEqual(comparable_first, comparable_second)
        self.assertEqual(first.artifacts, second.artifacts)

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
        config_text = (EXAMPLE_DIR / "config.yaml").read_text()
        task = self.run_batch.SweepTask(n=18, k=7, d=7, lower=7, upper=8)
        resolved = self.run_batch.render_resolved_config(config_text, task)
        self.assertIn("- n = 18", resolved)
        self.assertIn("- k = 7", resolved)
        self.assertIn("- d = 7", resolved)
        self.assertIn("- r = n - k = 11", resolved)
        self.assertNotIn("- n = 33", resolved)

    def test_parse_verification_output_distinguishes_complete_and_partial(self):
        """Verification parsing should classify complete and partial constructions."""
        complete = self.run_batch.parse_verification_output("d_actual: 7\nH shape: 11 x 18\n")
        partial = self.run_batch.parse_verification_output("d_partial: 9\nwarning: construction is incomplete\n")
        target_only = self.run_batch.parse_verification_output("d_at_least: 13\n")
        self.assertEqual(complete["verification_status"], "complete")
        self.assertEqual(complete["distance"], 7)
        self.assertEqual(partial["verification_status"], "partial")
        self.assertEqual(partial["distance"], 9)
        self.assertEqual(target_only["verification_status"], "target_only")
        self.assertEqual(target_only["distance_lower_bound"], 13)

    def test_instance_name_is_stable_for_output_directories(self):
        """Per-instance directories should be named by n/k/d and stay deterministic."""
        task = self.run_batch.SweepTask(n=20, k=7, d=8, lower=8, upper=9)
        self.assertEqual(task.instance_name, "n20_k7_d8")

    def test_verify_distance_accepts_cli_instance_arguments(self):
        """Verification CLI should accept --N/--K/--D to override instance values."""
        program_path = str(EXAMPLE_DIR / "initial_program.py")
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

    def test_verify_distance_target_only_skips_exact_distance(self):
        """Target-only verification should avoid exact d_actual enumeration."""
        program_path = str(EXAMPLE_DIR / "initial_program.py")
        stdout = io.StringIO()
        argv = [
            "verify_distance.py",
            program_path,
            "--no-progress",
            "--target-only",
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
        self.assertIn("d_at_least: 3", output)
        self.assertNotIn("d_actual:", output)


if __name__ == "__main__":
    unittest.main()
