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

    def test_restart_search_scores_candidates_once_across_restarts(self):
        """Restart tie-breaks should reuse static priority scores for each candidate."""
        instance = self.search_core.make_instance(n=8, k=4, distance=4, restarts=3)
        candidate_count = len(
            self.search_core.candidate_masks(instance.r, instance.target_distance)
        )
        calls = []

        def counted_priority(column_mask, n, k, d):
            calls.append(column_mask)
            return self.initial_program.get_priority_function()(column_mask, n, k, d)

        with mock.patch.dict(os.environ, {"LINEAR_CODE_RESTART_WORKERS": "3"}, clear=False):
            attempt = self.search_core.best_restart_for_instance(instance, counted_priority)

        self.assertTrue(attempt.success)
        self.assertEqual(len(calls), candidate_count)
        self.assertEqual(sorted(calls), sorted(set(calls)))

    def test_streaming_configuration_is_ignored_after_removal(self):
        """Former streaming flags should not change the materialized search path."""
        instance = self.search_core.make_instance(n=8, k=4, distance=4, restarts=3)
        candidate_count = len(
            self.search_core.candidate_masks(instance.r, instance.target_distance)
        )
        calls = []

        def counted_priority(column_mask, n, k, d):
            calls.append(column_mask)
            return self.initial_program.get_priority_function()(column_mask, n, k, d)

        with mock.patch.dict(
            os.environ,
            {
                "LINEAR_CODE_FORCE_STREAMING": "1",
                "LINEAR_CODE_CANDIDATE_CHUNK_SIZE": "3",
                "LINEAR_CODE_RESTART_WORKERS": "3",
                "LINEAR_CODE_PROFILE": "1",
            },
            clear=False,
        ):
            with mock.patch.object(self.search_core.logger, "info") as mock_info:
                attempt = self.search_core.best_restart_for_instance(instance, counted_priority)

        self.assertTrue(attempt.success)
        self.assertEqual(len(calls), candidate_count)
        self.assertEqual(sorted(calls), sorted(set(calls)))
        profile_messages = [
            call.args[0]
            for call in mock_info.call_args_list
            if call.args and "linear_code_profile" in call.args[0]
        ]
        self.assertFalse(any("chunk_index=" in message for message in profile_messages))

    def test_process_candidate_scoring_matches_thread_scoring(self):
        """Static candidate scoring should support process workers when requested."""
        instance = self.search_core.make_instance(n=8, k=4, distance=4, restarts=1)
        priority_fn = self.initial_program.get_priority_function()

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LINEAR_CODE_CANDIDATE_EXECUTOR", None)
            thread_scores = self.search_core.score_static_candidates(priority_fn=priority_fn, instance=instance)

        with mock.patch.dict(
            os.environ,
            {
                "LINEAR_CODE_CANDIDATE_EXECUTOR": "process",
                "LINEAR_CODE_CANDIDATE_WORKERS": "2",
                "LINEAR_CODE_PROFILE": "1",
            },
            clear=False,
        ):
            with mock.patch.object(self.search_core.logger, "info") as mock_info:
                process_scores = self.search_core.score_static_candidates(
                    priority_fn=priority_fn,
                    instance=instance,
                )

        self.assertEqual(process_scores, thread_scores)
        profile_messages = [
            call.args[0]
            for call in mock_info.call_args_list
            if call.args and "linear_code_profile" in call.args[0]
        ]
        self.assertTrue(
            any(
                "stage=candidate_scoring" in message and "executor_mode=process" in message
                for message in profile_messages
            )
        )

    def test_sampled_refill_search_solves_without_full_candidate_scoring(self):
        """Sampled refill mode should find a valid code while scoring only sampled candidates."""
        instance = self.search_core.make_instance(n=12, k=4, distance=4, restarts=3)
        candidate_count = self.search_core.candidate_count(
            instance.r,
            instance.target_distance,
        )
        calls = []

        def counted_priority(column_mask, n, k, d):
            calls.append(column_mask)
            return self.initial_program.get_priority_function()(column_mask, n, k, d)

        with mock.patch.dict(
            os.environ,
            {
                "LINEAR_CODE_SEARCH_MODE": "sampled_refill",
                "LINEAR_CODE_RANDOM_SEED": "1",
                "LINEAR_CODE_SAMPLE_POOL_SIZE": "64",
                "LINEAR_CODE_SAMPLE_ATTEMPTS_PER_REFILL": "1024",
                "LINEAR_CODE_SAMPLE_MAX_REFILLS": "16",
                "LINEAR_CODE_RESTART_WORKERS": "1",
            },
            clear=False,
        ):
            result = self.search_core.evaluate_priority_function(counted_priority, instance)

        self.assertEqual(result.metrics["success_rate"], 1.0)
        self.assertLess(len(calls), candidate_count * instance.restarts)
        search_result = json.loads(result.artifacts["search_result"])
        self.assertEqual(search_result["search_mode"], "sampled_refill")
        self.assertLess(search_result["scored_candidates"], candidate_count)
        selected = tuple(int(bits, 2) for bits in search_result["selected_free_columns"])
        self.assertTrue(
            self.search_core.validate_free_columns(
                instance.r,
                selected,
                instance.target_distance,
            )
        )

    def test_sample_weight_layers_use_candidate_layer_counts(self):
        """Sampled weight priors should match exact binomial layer sizes."""
        layers = self.search_core.candidate_weight_layer_counts(r=6, distance=4)
        self.assertEqual(layers, ((3, 20), (4, 15), (5, 6), (6, 1)))
        rng = self.search_core.random.Random(123)
        sampled_weights = {
            self.search_core._sample_weight(6, 4, rng)
            for _ in range(200)
        }
        self.assertTrue(sampled_weights)
        self.assertTrue(all(weight >= 3 for weight in sampled_weights))
        self.assertTrue(all(weight <= 6 for weight in sampled_weights))

    def test_sampled_refill_artifacts_use_sampled_rank_scope(self):
        """Sampled vector ranks should be labeled as pool-local rather than global ranks."""
        instance = self.search_core.make_instance(n=12, k=4, distance=4, restarts=1)
        with mock.patch.dict(
            os.environ,
            {
                "LINEAR_CODE_SEARCH_MODE": "sampled_refill",
                "LINEAR_CODE_RANDOM_SEED": "1",
                "LINEAR_CODE_SAMPLE_POOL_SIZE": "64",
                "LINEAR_CODE_SAMPLE_ATTEMPTS_PER_REFILL": "1024",
            },
            clear=False,
        ):
            result = self.search_core.evaluate_priority_function(
                self.initial_program.get_priority_function(),
                instance,
            )

        self.assertEqual(result.metrics["success_rate"], 1.0)
        vectors = json.loads(result.artifacts["successful_code_vectors"])
        summary = json.loads(result.artifacts["successful_code_summary"])
        self.assertEqual(summary["search_mode"], "sampled_refill")
        self.assertTrue(all(entry["rank_scope"] == "sampled_pool" for entry in vectors))
        self.assertEqual([entry["fill_index"] for entry in vectors], [1, 2, 3, 4])

    def test_sampled_refill_backtracking_keeps_final_vectors_consistent(self):
        """Backtracking should remove stale path columns from the final success artifacts."""
        instance = self.search_core.make_instance(n=10, k=4, distance=4, restarts=1)
        with mock.patch.dict(
            os.environ,
            {
                "LINEAR_CODE_SEARCH_MODE": "sampled_refill",
                "LINEAR_CODE_RANDOM_SEED": "5",
                "LINEAR_CODE_SAMPLE_POOL_SIZE": "1",
                "LINEAR_CODE_SAMPLE_ATTEMPTS_PER_REFILL": "1",
                "LINEAR_CODE_SAMPLE_MAX_REFILLS": "80",
                "LINEAR_CODE_SAMPLE_MAX_STALE_REFILLS": "1",
                "LINEAR_CODE_BACKTRACK_DEPTH": "1",
                "LINEAR_CODE_BACKTRACK_MAX_EVENTS": "10",
            },
            clear=False,
        ):
            result = self.search_core.evaluate_priority_function(
                self.initial_program.get_priority_function(),
                instance,
            )

        self.assertEqual(result.metrics["success_rate"], 1.0)
        search_result = json.loads(result.artifacts["search_result"])
        self.assertEqual(search_result["backtrack_events"], 1)
        self.assertEqual(search_result["backtracked_columns"], 1)
        selected = search_result["selected_free_columns"]
        vectors = json.loads(result.artifacts["successful_code_vectors"])
        self.assertEqual([entry["fill_index"] for entry in vectors], [1, 2, 3, 4])
        self.assertEqual([entry["column"] for entry in vectors], selected)
        self.assertTrue(
            self.search_core.validate_free_columns(
                instance.r,
                tuple(int(bits, 2) for bits in selected),
                instance.target_distance,
            )
        )

    def test_sampled_refill_search_is_reproducible_for_fixed_seed(self):
        """Randomized search should be reproducible when the seed and budgets are fixed."""
        instance = self.search_core.make_instance(n=12, k=4, distance=4, restarts=2)
        env = {
            "LINEAR_CODE_SEARCH_MODE": "sampled_refill",
            "LINEAR_CODE_RANDOM_SEED": "7",
            "LINEAR_CODE_SAMPLE_POOL_SIZE": "64",
            "LINEAR_CODE_SAMPLE_ATTEMPTS_PER_REFILL": "1024",
            "LINEAR_CODE_RESTART_WORKERS": "1",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            first = self.search_core.evaluate_priority_function(
                self.initial_program.get_priority_function(),
                instance,
            )
        with mock.patch.dict(os.environ, env, clear=False):
            second = self.search_core.evaluate_priority_function(
                self.initial_program.get_priority_function(),
                instance,
            )

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

    def test_sampled_beam_search_solves_and_uses_beam_rank_scope(self):
        """Sampled beam mode should solve a small instance without full enumeration."""
        instance = self.search_core.make_instance(n=12, k=4, distance=4, restarts=1)
        candidate_count = self.search_core.candidate_count(
            instance.r,
            instance.target_distance,
        )
        env = {
            "LINEAR_CODE_SEARCH_MODE": "sampled_beam",
            "LINEAR_CODE_RANDOM_SEED": "1",
            "LINEAR_CODE_BEAM_WIDTH": "4",
            "LINEAR_CODE_BEAM_BRANCHES_PER_STATE": "16",
            "LINEAR_CODE_BEAM_ATTEMPTS_PER_STATE": "256",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            first = self.search_core.evaluate_priority_function(
                self.initial_program.get_priority_function(),
                instance,
            )
        with mock.patch.dict(os.environ, env, clear=False):
            second = self.search_core.evaluate_priority_function(
                self.initial_program.get_priority_function(),
                instance,
            )

        self.assertEqual(first.metrics["success_rate"], 1.0)
        search_result = json.loads(first.artifacts["search_result"])
        self.assertEqual(search_result["search_mode"], "sampled_beam")
        self.assertEqual(search_result["beam_width"], 4)
        self.assertGreater(search_result["beam_expanded_states"], 0)
        self.assertLess(search_result["scored_candidates"], candidate_count * instance.restarts)
        vectors = json.loads(first.artifacts["successful_code_vectors"])
        self.assertTrue(all(entry["rank_scope"] == "sampled_beam_pool" for entry in vectors))

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

    def test_progress_flag_emits_sampled_restart_steps(self):
        """Progress mode should report sampled restart and refill phases."""
        instance = self.search_core.make_instance(n=12, k=4, distance=4, restarts=1)
        with mock.patch.dict(
            os.environ,
            {
                "LINEAR_CODE_SEARCH_MODE": "sampled_refill",
                "LINEAR_CODE_PROGRESS": "1",
                "LINEAR_CODE_RANDOM_SEED": "1",
                "LINEAR_CODE_SAMPLE_POOL_SIZE": "64",
                "LINEAR_CODE_SAMPLE_ATTEMPTS_PER_REFILL": "1024",
            },
            clear=False,
        ):
            with mock.patch.object(self.search_core, "_progress_message") as mock_progress:
                with mock.patch.object(
                    self.search_core,
                    "_iterate_with_progress",
                    side_effect=lambda items, description, show_progress, total=None: items,
                ):
                    result = self.search_core.evaluate_priority_function(
                        self.initial_program.get_priority_function(),
                        instance,
                    )

        self.assertEqual(result.metrics["success_rate"], 1.0)
        messages = [
            call.args[1]
            for call in mock_progress.call_args_list
            if len(call.args) >= 2 and call.args[0]
        ]
        self.assertTrue(any("restart 0: start sampled_refill" in message for message in messages))
        self.assertTrue(any("refill 1/" in message and "sample" in message for message in messages))
        self.assertTrue(any("sort_and_greedy" in message for message in messages))
        self.assertTrue(any("restart 0: finish" in message for message in messages))

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

    def test_successful_code_vector_artifacts_describe_accepted_columns(self):
        """Successful constructions should expose rank, score, and weight for each fill."""
        instance = self.search_core.make_instance(n=7, k=4, distance=3, restarts=1)
        result = self.search_core.evaluate_priority_function(
            self.initial_program.get_priority_function(),
            instance,
        )

        self.assertEqual(result.metrics["success_rate"], 1.0)
        self.assertIn("successful_code_vectors", result.artifacts)
        self.assertIn("successful_code_summary", result.artifacts)

        vectors = json.loads(result.artifacts["successful_code_vectors"])
        summary = json.loads(result.artifacts["successful_code_summary"])

        self.assertEqual(len(vectors), instance.k)
        self.assertEqual([entry["fill_index"] for entry in vectors], [1, 2, 3, 4])
        self.assertTrue(all(entry["rank"] >= 1 for entry in vectors))
        self.assertTrue(all(isinstance(entry["score"], float) for entry in vectors))
        for entry in vectors:
            self.assertEqual(entry["weight"], entry["column"].count("1"))

        ranks = [entry["rank"] for entry in vectors]
        weight_histogram = {}
        for entry in vectors:
            weight = str(entry["weight"])
            weight_histogram[weight] = weight_histogram.get(weight, 0) + 1

        self.assertEqual(summary["n"], instance.n)
        self.assertEqual(summary["k"], instance.k)
        self.assertEqual(summary["d"], instance.target_distance)
        self.assertEqual(summary["r"], instance.r)
        self.assertEqual(summary["restart"], 0)
        self.assertEqual(summary["vector_count"], instance.k)
        self.assertEqual(summary["rank_min"], min(ranks))
        self.assertEqual(summary["rank_max"], max(ranks))
        self.assertAlmostEqual(summary["rank_avg"], sum(ranks) / len(ranks))
        self.assertEqual(summary["weight_histogram"], weight_histogram)

    def test_successful_code_vector_artifacts_are_success_only(self):
        """Partial constructions should keep existing artifacts without success-only analysis."""
        instance = self.search_core.make_instance(n=16, k=15, distance=2, restarts=1)
        result = self.search_core.evaluate_priority_function(
            self.initial_program.get_priority_function(),
            instance,
        )

        self.assertEqual(result.metrics["success_rate"], 0.0)
        self.assertIn("search_result", result.artifacts)
        self.assertNotIn("successful_code_vectors", result.artifacts)
        self.assertNotIn("successful_code_summary", result.artifacts)

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
        self.assertEqual(complete["verification_status"], "complete")
        self.assertEqual(complete["distance"], 7)
        self.assertEqual(partial["verification_status"], "partial")
        self.assertEqual(partial["distance"], 9)

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


if __name__ == "__main__":
    unittest.main()
