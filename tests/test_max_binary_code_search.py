"""Tests for binary maximum-code search helpers."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "examples" / "max_binary_code_search"


def _load_module(module_name: str, path: Path):
    inserted = str(path.parent)
    sys.path.insert(0, inserted)
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        try:
            sys.path.remove(inserted)
        except ValueError:
            pass


class TestMaxBinaryCodeSearchMctsRepair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.search_core = _load_module(
            "max_binary_code_search_core_tests",
            EXAMPLE_DIR / "search_core.py",
        )
        sys.modules["search_core"] = cls.search_core
        cls.initial_program = _load_module(
            "max_binary_code_initial_tests",
            EXAMPLE_DIR / "initial_program.py",
        )
        cls.c_kernel_runner = _load_module(
            "max_binary_code_c_kernel_runner_tests",
            EXAMPLE_DIR / "c_kernel_runner.py",
        )

    def test_mcts_repair_mode_emits_valid_parity_artifact(self):
        env = {
            "MAX_CODE_REPAIR_MODE": "mcts",
            "MAX_CODE_REPAIR_MCTS_SIMULATIONS": "4",
            "MAX_CODE_REPAIR_MCTS_DEPTH": "2",
            "MAX_CODE_PARITY_FULL_SCAN_N": "12",
        }
        with patch.dict(os.environ, env, clear=False):
            instance = self.search_core.make_instance(n=10, distance=4, restarts=1)
            result = self.search_core.evaluate_priority_function(
                self.initial_program.priority,
                instance,
            )

        search_result = json.loads(result.artifacts["search_result"])
        codewords = [
            int(word, 2)
            for word in json.loads(result.artifacts["codewords"])
        ]
        self.assertEqual(search_result["search_mode"], "parity_transform_dynamic_mcts_repair")
        self.assertEqual(search_result["repair_mode"], "mcts")
        self.assertTrue(search_result["valid"])
        self.assertTrue(self.search_core.validate_code(codewords, 4))

    def test_mcts_drop_selector_returns_nonzero_selected_center(self):
        state = self.search_core.ParityTransformState(6)
        for word in (0, 0b111000, 0b100110, 0b010101):
            if state.can_add(word):
                state.add(word)

        drop_index = self.search_core._parity_choose_mcts_drop_index(
            state,
            self.initial_program.priority,
            restart_index=0,
            repair_event_index=0,
            seed=123,
            pool_size=16,
            attempts_per_refill=256,
            tabu=set(),
        )

        self.assertIsNotNone(drop_index)
        self.assertGreater(drop_index, 0)
        self.assertIn(state.codewords[drop_index], state.codewords[1:])

    def test_general_mcts_repair_mode_emits_valid_non_parity_artifact(self):
        env = {
            "MAX_CODE_REPAIR_MODE": "mcts",
            "MAX_CODE_REPAIR_EVENTS": "2",
            "MAX_CODE_REPAIR_DROP_COUNT": "1",
            "MAX_CODE_REPAIR_TABU_TENURE": "2",
            "MAX_CODE_REPAIR_CANDIDATE_WINDOW": "32",
            "MAX_CODE_REPAIR_MCTS_SIMULATIONS": "4",
            "MAX_CODE_REPAIR_MCTS_DEPTH": "2",
            "MAX_CODE_REPAIR_MCTS_DROP_TOPK": "2",
            "MAX_CODE_RANDOM_SEED": "5",
        }
        with patch.dict(os.environ, env, clear=False):
            instance = self.search_core.make_instance(n=8, distance=3, restarts=1)
            result = self.search_core.evaluate_priority_function(
                self.initial_program.priority,
                instance,
            )

        search_result = json.loads(result.artifacts["search_result"])
        codewords = [
            int(word, 2)
            for word in json.loads(result.artifacts["codewords"])
        ]
        self.assertEqual(search_result["search_mode"], "dynamic_mcts_repair")
        self.assertEqual(search_result["repair_mode"], "mcts")
        self.assertIn("repair_rollout_evaluations", search_result)
        self.assertTrue(search_result["valid"])
        self.assertTrue(self.search_core.validate_code(codewords, 3))

    def test_general_mcts_drop_selector_never_drops_zero_word(self):
        env = {
            "MAX_CODE_REPAIR_MCTS_SIMULATIONS": "4",
            "MAX_CODE_REPAIR_MCTS_DEPTH": "2",
            "MAX_CODE_REPAIR_CANDIDATE_WINDOW": "32",
            "MAX_CODE_REPAIR_MCTS_DROP_TOPK": "2",
            "MAX_CODE_RANDOM_SEED": "17",
        }
        with patch.dict(os.environ, env, clear=False):
            instance = self.search_core.make_instance(n=8, distance=3, restarts=1)
            static_scores = self.search_core.score_static_candidates(
                instance,
                self.initial_program.priority,
            )
            greedy = self.search_core.greedy_construct(
                instance,
                static_scores,
                restart_index=0,
                priority_fn=self.initial_program.priority,
            )
            ranked_scores = self.search_core._ranked_scores(static_scores, 0)
            drop_index, _evaluations = self.search_core._choose_mcts_drop_index(
                greedy.codewords,
                instance,
                ranked_scores,
                self.initial_program.priority,
                restart_index=0,
                repair_event_index=0,
                seed=17,
                dynamic_window=32,
                tabu=set(),
                local_offsets=self.search_core.sampled_exact_weight_offsets(8, 3, 16),
            )

        if drop_index is not None:
            self.assertGreater(drop_index, 0)
            self.assertNotEqual(greedy.codewords[drop_index], 0)

    def test_general_mcts_repair_is_reproducible_for_fixed_seed(self):
        env = {
            "MAX_CODE_REPAIR_MODE": "mcts",
            "MAX_CODE_REPAIR_EVENTS": "2",
            "MAX_CODE_REPAIR_DROP_COUNT": "1",
            "MAX_CODE_REPAIR_TABU_TENURE": "2",
            "MAX_CODE_REPAIR_CANDIDATE_WINDOW": "32",
            "MAX_CODE_REPAIR_MCTS_SIMULATIONS": "4",
            "MAX_CODE_REPAIR_MCTS_DEPTH": "2",
            "MAX_CODE_REPAIR_MCTS_DROP_TOPK": "2",
            "MAX_CODE_RANDOM_SEED": "23",
        }
        with patch.dict(os.environ, env, clear=False):
            instance = self.search_core.make_instance(n=8, distance=3, restarts=1)
            first = self.search_core.evaluate_priority_function(
                self.initial_program.priority,
                instance,
            )
            second = self.search_core.evaluate_priority_function(
                self.initial_program.priority,
                instance,
            )

        self.assertEqual(first.artifacts["codewords"], second.artifacts["codewords"])
        first_search = json.loads(first.artifacts["search_result"])
        second_search = json.loads(second.artifacts["search_result"])
        self.assertEqual(first_search["search_mode"], second_search["search_mode"])
        self.assertEqual(first_search["repair_moves"], second_search["repair_moves"])
        self.assertEqual(
            first_search["repair_rollout_evaluations"],
            second_search["repair_rollout_evaluations"],
        )

    def test_c_kernel_mcts_evaluator_emits_valid_artifact(self):
        env = {
            "MAX_CODE_N": "8",
            "MAX_CODE_D": "3",
            "MAX_CODE_RESTARTS": "1",
            "MAX_CODE_REPAIR_MODE": "mcts",
            "MAX_CODE_DYNAMIC_WINDOW": "64",
            "MAX_CODE_REPAIR_EVENTS": "1",
            "MAX_CODE_REPAIR_CANDIDATE_WINDOW": "128",
            "MAX_CODE_REPAIR_MCTS_SIMULATIONS": "2",
            "MAX_CODE_REPAIR_MCTS_DEPTH": "1",
            "MAX_CODE_REPAIR_MCTS_WORKERS": "1",
            "MAX_CODE_RANDOM_SEED": "31",
        }
        with patch.dict(os.environ, env, clear=False):
            result = self.c_kernel_runner.evaluate_c_program_path(
                str(EXAMPLE_DIR / "initial_program.c")
            )

        search_result = json.loads(result.artifacts["search_result"])
        codewords = [
            int(word, 2)
            for word in json.loads(result.artifacts["codewords"])
        ]
        self.assertEqual(search_result["search_mode"], "c_kernel_mcts")
        self.assertEqual(search_result["repair_mode"], "mcts")
        self.assertIn("repair_rollout_evaluations", search_result)
        self.assertGreater(result.metrics["code_size"], 0)
        self.assertEqual(result.metrics["valid"], 1.0)
        self.assertTrue(self.search_core.validate_code(codewords, 3))


if __name__ == "__main__":
    unittest.main()
