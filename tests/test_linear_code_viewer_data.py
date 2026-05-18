"""Tests for the binary-code matrix viewer data generator."""

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


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


class TestLinearCodeViewerData(unittest.TestCase):
    """The generated JSON should encode bounds, attempts, details, and labels."""

    @classmethod
    def setUpClass(cls):
        cls.viewer_data = _load_module(
            "linear_code_generate_viewer_data",
            EXAMPLE_DIR / "generate_viewer_data.py",
        )

    def test_build_dataset_classifies_cells_and_extracts_complete_details(self):
        with TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            record_path = tmp / "ECCRecord.json"
            record_path.write_text(
                json.dumps(
                    {
                        "3": {
                            "1": {"lower": 3, "upper": 3},
                            "2": {"lower": 2, "upper": 4},
                            "3": {"lower": 1, "upper": 1},
                        },
                        "4": {
                            "1": {"lower": 4, "upper": 4},
                            "2": {"lower": 2, "upper": 5},
                            "3": {"lower": 2, "upper": 2},
                            "4": {"lower": 1, "upper": 1},
                        },
                    }
                )
            )
            runs_root = tmp / "batch_runs"

            self._write_run(
                runs_root / "n3_k1_d3",
                status_line="d_actual: 3",
                priority_source="def priority(column_mask, n, k, d):\n    return 3\n",
                h_rows=("111", "001"),
                g_rows=("101",),
                metrics={"combined_score": 1.0, "constructed_columns": 1},
            )
            self._write_run(
                runs_root / "n3_k2_d2",
                status_line="d_actual: 2",
                priority_source="def priority(column_mask, n, k, d):\n    return 2\n",
                h_rows=("110",),
                g_rows=("10", "01"),
                metrics={"combined_score": 1.0, "constructed_columns": 2},
            )
            self._write_run(
                runs_root / "n3_k2_d4",
                status_line="d_partial: 2",
                priority_source="def priority(column_mask, n, k, d):\n    return -1\n",
                h_rows=("101",),
                g_rows=(),
                metrics={"combined_score": 0.5, "constructed_columns": 1},
            )
            self._write_run(
                runs_root / "n4_k2_d3",
                status_line="d_actual: 3",
                priority_source="def priority(column_mask, n, k, d):\n    return 30\n",
                h_rows=("1110", "0011"),
                g_rows=("1010", "0101"),
                metrics={"combined_score": 1.0, "constructed_columns": 2},
            )
            self._write_run(
                runs_root / "n4_k1_d4",
                status_line="d_partial: 3",
                priority_source="def priority(column_mask, n, k, d):\n    return 0\n",
                h_rows=("10",),
                g_rows=(),
                metrics={"combined_score": 0.5, "constructed_columns": 0},
            )

            dataset = self.viewer_data.build_dataset(record_path, [runs_root])
            cells = {(cell["n"], cell["k"]): cell for cell in dataset["cells"]}

            self.assertEqual(dataset["meta"]["totalCells"], 7)
            self.assertEqual(cells[(3, 1)]["status"], "found")
            self.assertEqual(cells[(3, 1)]["label"], "3")
            self.assertEqual(cells[(3, 2)]["status"], "upper_failed_after_found")
            self.assertEqual(cells[(3, 2)]["label"], "2-4")
            self.assertEqual(cells[(3, 2)]["attemptedTargets"], [2, 4])
            self.assertEqual(cells[(3, 2)]["bestDistance"], 2)
            self.assertEqual(cells[(4, 2)]["status"], "found")
            self.assertEqual(cells[(4, 2)]["label"], "2-3-5")
            self.assertEqual(cells[(4, 1)]["status"], "failed")
            self.assertEqual(cells[(4, 3)]["status"], "found")
            self.assertEqual(cells[(4, 3)]["bestDistance"], 2)
            self.assertEqual(cells[(4, 4)]["status"], "found")
            self.assertEqual(cells[(4, 4)]["bestDistance"], 1)

            found_detail = dataset["details"][cells[(3, 1)]["detailId"]]
            self.assertIn("return 3", found_detail["prioritySource"])
            self.assertEqual(found_detail["hRows"], ["111", "001"])
            self.assertEqual(found_detail["gRows"], ["101"])
            self.assertTrue(found_detail["completeConstruction"])
            self.assertNotIn("paths", found_detail)
            self.assertNotIn("paths", found_detail["attempts"][0])

            for key, expected_distance in [((4, 3), 2), ((4, 4), 1)]:
                trivial_detail = dataset["details"][cells[key]["detailId"]]
                self.assertFalse(trivial_detail["completeConstruction"])
                self.assertTrue(trivial_detail["trivialDistance"])
                self.assertEqual(trivial_detail["bestDistance"], expected_distance)
                self.assertEqual(trivial_detail["hRows"], [])
                self.assertEqual(trivial_detail["attempts"], [])

            failed_detail = dataset["details"][cells[(4, 1)]["detailId"]]
            self.assertFalse(failed_detail["completeConstruction"])
            self.assertEqual(failed_detail["hRows"], [])
            self.assertEqual(failed_detail["attempts"][0]["status"], "partial")

    def _write_run(
        self,
        run_dir: Path,
        *,
        status_line: str,
        priority_source: str,
        h_rows: tuple[str, ...],
        g_rows: tuple[str, ...],
        metrics: dict,
    ) -> None:
        best_dir = run_dir / "best"
        best_dir.mkdir(parents=True)
        (best_dir / "best_program.py").write_text(priority_source)
        (best_dir / "best_program_info.json").write_text(
            json.dumps({"iteration": 7, "metrics": metrics})
        )
        lines = [
            f"program_path: {best_dir / 'best_program.py'}",
            'instance: {"n": 3, "k": 1, "d_target": 3, "r": 2}',
            status_line,
            "H rows:" if status_line.startswith("d_actual") else "Partial H rows:",
            *h_rows,
        ]
        if g_rows:
            lines.extend(["G rows:", *g_rows])
        (run_dir / "matrix_verification.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    unittest.main()
