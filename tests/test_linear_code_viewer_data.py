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
        cls.code_table_db = _load_module(
            "linear_code_table_db",
            EXAMPLE_DIR / "code_table_db.py",
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

    def test_sqlite_store_imports_runs_and_exports_viewer_dataset(self):
        with TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            record_path = tmp / "ECCRecord.json"
            record_path.write_text(
                json.dumps(
                    {
                        "3": {
                            "1": {"lower": 3, "upper": 3},
                            "2": {"lower": 2, "upper": 4},
                        }
                    }
                )
            )
            runs_root = tmp / "outputs"
            self._write_run(
                runs_root / "n3_k1_d3" / "batch_1",
                status_line="d_actual: 3",
                priority_source="def priority(column_mask, n, k, d):\n    return 3\n",
                h_rows=("111", "001"),
                g_rows=("101",),
                metrics={"combined_score": 1.0, "constructed_columns": 1},
            )

            db_path = tmp / "records.sqlite"
            with self.code_table_db.connect(db_path) as conn:
                self.code_table_db.init_db(conn)
                self.assertEqual(self.code_table_db.import_bounds(conn, record_path), 2)
                self.assertEqual(self.code_table_db.import_runs(conn, [runs_root]), 1)
                dataset = self.code_table_db.build_dataset_from_db(conn)

            cells = {(cell["n"], cell["k"]): cell for cell in dataset["cells"]}
            self.assertEqual(dataset["meta"]["totalCells"], 2)
            self.assertEqual(dataset["meta"]["sqlite"]["attempts"], 1)
            self.assertEqual(cells[(3, 1)]["status"], "found")
            self.assertEqual(cells[(3, 1)]["attemptedTargets"], [3])
            self.assertEqual(cells[(3, 2)]["status"], "found")

            detail = dataset["details"][cells[(3, 1)]["detailId"]]
            self.assertTrue(detail["completeConstruction"])
            self.assertEqual(detail["hRows"], ["111", "001"])
            self.assertEqual(detail["gRows"], ["101"])
            self.assertIn("return 3", detail["prioritySource"])

    def test_build_dataset_reads_matrix_artifacts_from_best_info(self):
        with TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            record_path = tmp / "ECCRecord.json"
            record_path.write_text(
                json.dumps({"5": {"2": {"lower": 3, "upper": 3}}})
            )
            run_dir = tmp / "outputs" / "n5_k2_d3" / "run_1"
            best_dir = run_dir / "best"
            best_dir.mkdir(parents=True)
            (best_dir / "best_program.c").write_text("int oe_linear_code_priority(void) { return 1; }\n")
            (best_dir / "best_program_info.json").write_text(
                json.dumps(
                    {
                        "iteration": 11,
                        "generation": 2,
                        "metrics": {
                            "success_rate": 1.0,
                            "constructed_columns": 2,
                            "target_columns": 2,
                            "target_distance": 3,
                            "combined_score": 1.0,
                        },
                        "artifacts": {
                            "search_result": json.dumps(
                                {
                                    "success": True,
                                    "added_free_columns": 2,
                                    "selected_free_columns": ["101", "110"],
                                },
                                sort_keys=True,
                            ),
                            "matrix_summary": json.dumps(
                                {
                                    "complete": True,
                                    "selected_free_columns": ["101", "110"],
                                },
                                sort_keys=True,
                            ),
                            "parity_check_matrix": json.dumps(["10100", "01010", "11001"]),
                            "generator_matrix": json.dumps(["10101", "01011"]),
                        },
                    }
                )
            )

            dataset = self.viewer_data.build_dataset(record_path, [tmp / "outputs"])
            cell = dataset["cells"][0]
            detail = dataset["details"][cell["detailId"]]

            self.assertEqual(cell["status"], "found")
            self.assertEqual(cell["bestDistance"], 3)
            self.assertTrue(detail["completeConstruction"])
            self.assertEqual(detail["bestDistance"], 3)
            self.assertEqual(detail["hRows"], ["10100", "01010", "11001"])
            self.assertEqual(detail["gRows"], ["10101", "01011"])

    def test_sqlite_store_imports_existing_viewer_json(self):
        with TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            viewer_json = tmp / "code_table_data.json"
            viewer_json.write_text(
                json.dumps(
                    {
                        "meta": {"recordName": "synthetic.json"},
                        "cells": [
                            {
                                "n": 5,
                                "k": 2,
                                "lower": 3,
                                "upper": 4,
                                "label": "3",
                                "status": "found",
                                "bestDistance": 3,
                                "attemptedTargets": [3],
                                "detailId": "n5_k2",
                            }
                        ],
                        "details": {
                            "n5_k2": {
                                "n": 5,
                                "k": 2,
                                "lower": 3,
                                "upper": 4,
                                "completeConstruction": True,
                                "trivialDistance": False,
                                "bestDistance": 3,
                                "targetDistance": 3,
                                "prioritySource": "int priority(void) { return 5; }\n",
                                "hRows": ["11100", "00111", "01010"],
                                "gRows": ["10101", "01011"],
                                "selectedFreeColumns": ["111", "001"],
                                "metrics": {"combined_score": 2.0},
                                "sourceRoot": "outputs",
                                "sourceRun": "n5_k2_d3/batch_1",
                                "attempts": [
                                    {
                                        "targetDistance": 3,
                                        "status": "complete",
                                        "actualDistance": 3,
                                        "metrics": {"combined_score": 2.0},
                                        "iteration": 4,
                                        "generation": 5,
                                        "timestamp": "2026-01-01T00:00:00Z",
                                        "sourceRoot": "outputs",
                                        "sourceRun": "n5_k2_d3/batch_1",
                                        "method": "manual",
                                        "derivedFrom": {"n": 4, "k": 1, "actualDistance": 4},
                                    }
                                ],
                            }
                        },
                    }
                )
            )

            db_path = tmp / "records.sqlite"
            with self.code_table_db.connect(db_path) as conn:
                self.code_table_db.init_db(conn)
                bounds, attempts = self.code_table_db.import_viewer_json(conn, viewer_json)
                dataset = self.code_table_db.build_dataset_from_db(conn)

            self.assertEqual(bounds, 1)
            self.assertEqual(attempts, 1)
            detail = dataset["details"]["n5_k2"]
            self.assertEqual(detail["hRows"], ["11100", "00111", "01010"])
            self.assertEqual(detail["attempts"][0]["method"], "manual")
            self.assertEqual(detail["attempts"][0]["derivedFrom"]["n"], 4)

    def test_sqlite_store_does_not_downgrade_existing_attempt(self):
        with TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            db_path = tmp / "records.sqlite"
            complete_attempt = {
                "n": 6,
                "k": 3,
                "targetDistance": 3,
                "status": "complete",
                "actualDistance": 3,
                "selectedFreeColumns": ["111", "001", "010"],
                "hRows": ["111000", "001110", "010101"],
                "gRows": ["101000", "010100", "001010"],
                "prioritySource": "int priority(void) { return 1; }\n",
                "metrics": {"combined_score": 1.0},
                "iteration": 8,
                "generation": 1,
                "timestamp": "2026-01-01T00:00:00Z",
                "sourceRoot": "outputs",
                "sourceRun": "n6_k3_d3/batch_1",
            }
            missing_attempt = dict(complete_attempt)
            missing_attempt.update(
                {
                    "status": "missing",
                    "actualDistance": None,
                    "selectedFreeColumns": [],
                    "hRows": [],
                    "gRows": [],
                    "prioritySource": "",
                    "metrics": {"combined_score": 0.0},
                    "iteration": 9,
                }
            )

            with self.code_table_db.connect(db_path) as conn:
                self.code_table_db.init_db(conn)
                self.code_table_db.upsert_attempt(conn, complete_attempt)
                self.code_table_db.upsert_attempt(conn, missing_attempt)
                stored = self.code_table_db.attempts_from_db(conn)[(6, 3)][0]

            self.assertEqual(stored["status"], "complete")
            self.assertEqual(stored["actualDistance"], 3)
            self.assertEqual(stored["hRows"], ["111000", "001110", "010101"])

    def test_sqlite_store_does_not_replace_matrix_with_matrixless_attempt(self):
        with TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            db_path = tmp / "records.sqlite"
            complete_attempt = {
                "n": 6,
                "k": 3,
                "targetDistance": 3,
                "status": "complete",
                "actualDistance": 3,
                "selectedFreeColumns": ["111", "001", "010"],
                "hRows": ["111000", "001110", "010101"],
                "gRows": ["101000", "010100", "001010"],
                "prioritySource": "int priority(void) { return 1; }\n",
                "metrics": {"combined_score": 1.0},
                "iteration": 8,
                "generation": 1,
                "timestamp": "2026-01-01T00:00:00Z",
                "sourceRoot": "outputs",
                "sourceRun": "n6_k3_d3/batch_1",
            }
            matrixless_attempt = dict(complete_attempt)
            matrixless_attempt.update(
                {
                    "selectedFreeColumns": [],
                    "hRows": [],
                    "gRows": [],
                    "prioritySource": "int priority(void) { return 2; }\n",
                    "iteration": 9,
                }
            )

            with self.code_table_db.connect(db_path) as conn:
                self.code_table_db.init_db(conn)
                self.code_table_db.upsert_attempt(conn, complete_attempt)
                self.code_table_db.upsert_attempt(conn, matrixless_attempt)
                stored = self.code_table_db.attempts_from_db(conn)[(6, 3)][0]

            self.assertEqual(stored["iteration"], 8)
            self.assertEqual(stored["hRows"], ["111000", "001110", "010101"])

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
