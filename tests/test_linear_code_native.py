"""Unit tests for the optional native binary linear-code legality engine."""

import importlib.util
import sys
import unittest
from pathlib import Path


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


search_core = _load_module("linear_code_search_core_native_tests", EXAMPLE_DIR / "search_core.py")
try:
    import _linear_code_native as native
except ImportError as exc:
    raise unittest.SkipTest("native extension is not built") from exc


class TestLinearCodeNative(unittest.TestCase):
    """Native state should match the existing Python exact legality engine."""

    def test_can_add_and_add_match_python_state(self):
        r = 5
        distance = 4
        py_state = search_core.IncrementalForbiddenState(r, distance)
        native_state = native.NativeForbiddenState(r, distance)
        accepted = []

        for _ in range(3):
            for candidate in search_core.candidate_masks(r, distance):
                self.assertEqual(
                    native_state.can_add(candidate),
                    py_state.can_add(candidate),
                )
            next_column = next(
                candidate
                for candidate in search_core.candidate_masks(r, distance)
                if py_state.can_add(candidate)
            )
            before_count = len(py_state.forbidden)
            py_growth = py_state.add(next_column)
            native_growth = native_state.add(next_column)
            accepted.append(next_column)
            self.assertEqual(native_growth, py_growth)
            self.assertEqual(native_growth, len(py_state.forbidden) - before_count)
            self.assertEqual(native_state.selected_columns(), tuple(accepted))
            self.assertEqual(native_state.forbidden_count(), len(py_state.forbidden))
            self.assertEqual(
                native_state.layer_counts(),
                tuple(len(layer) for layer in py_state.reachable),
            )

    def test_undo_restores_lifo_state(self):
        r = 5
        distance = 4
        columns = []
        state = native.NativeForbiddenState(r, distance)
        for candidate in search_core.candidate_masks(r, distance):
            if state.can_add(candidate):
                state.add(candidate)
                columns.append(candidate)
            if len(columns) == 2:
                break

        state.undo(1)
        expected = native.NativeForbiddenState(r, distance)
        expected.add(columns[0])
        self.assertEqual(state.selected_columns(), expected.selected_columns())
        self.assertEqual(state.forbidden_count(), expected.forbidden_count())
        self.assertEqual(state.layer_counts(), expected.layer_counts())

        for candidate in search_core.candidate_masks(r, distance):
            self.assertEqual(state.can_add(candidate), expected.can_add(candidate))

        state.undo(1)
        initial = native.NativeForbiddenState(r, distance)
        self.assertEqual(state.selected_columns(), ())
        self.assertEqual(state.forbidden_count(), initial.forbidden_count())
        self.assertEqual(state.layer_counts(), initial.layer_counts())

    def test_clone_does_not_mutate_parent(self):
        r = 5
        distance = 4
        parent = native.NativeForbiddenState(r, distance)
        first = next(iter(search_core.candidate_masks(r, distance)))
        parent.add(first)
        child = parent.clone()

        second = next(
            candidate
            for candidate in search_core.candidate_masks(r, distance)
            if candidate != first and child.can_add(candidate)
        )
        child.add(second)

        self.assertEqual(parent.selected_columns(), (first,))
        self.assertEqual(child.selected_columns(), (first, second))
        self.assertLess(parent.forbidden_count(), child.forbidden_count())

    def test_validate_columns_matches_python_validation(self):
        r = 5
        distance = 4
        columns = []
        state = native.NativeForbiddenState(r, distance)
        for candidate in search_core.candidate_masks(r, distance):
            if state.can_add(candidate):
                state.add(candidate)
                columns.append(candidate)
            if len(columns) == 3:
                break

        self.assertTrue(native.validate_columns(r, distance, tuple(columns)))
        self.assertEqual(
            native.validate_columns(r, distance, tuple(columns)),
            search_core.validate_free_columns(r, tuple(columns), distance),
        )

    def test_r_60_uses_64_bit_masks(self):
        state = native.NativeForbiddenState(60, 3)
        column = (1 << 59) | (1 << 40) | (1 << 17)
        self.assertTrue(state.can_add(column))
        state.add(column)
        self.assertEqual(state.selected_columns(), (column,))
        self.assertTrue(native.validate_columns(60, 3, (column,)))

    def test_r_limit_is_loud(self):
        with self.assertRaisesRegex(ValueError, "r <= 60"):
            native.NativeForbiddenState(61, 4)

    def test_large_initial_forbidden_layers_fail_loudly(self):
        with self.assertRaisesRegex(MemoryError, "LINEAR_CODE_NATIVE_MAX_INITIAL_VALUES"):
            native.NativeForbiddenState(60, 13)


if __name__ == "__main__":
    unittest.main()
