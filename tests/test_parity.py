import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts import run_parity


class ParityTests(unittest.TestCase):
    def make_standard_pair(self, directory: Path):
        json_path = directory / "standard.json"
        csv_path = directory / "standard.csv"
        rows = [
            {
                "run": 1,
                "medication": "Lacosamide",
                "comparison": {"drug name": {"match": True}},
                "fields_correct": 9,
                "fields_total": 9,
                "all_fields_correct": True,
            }
        ]
        json_path.write_text(json.dumps(rows), encoding="utf-8")
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Run", "Medication", "drug name", "Fields Correct", "All Correct"])
            writer.writerow([1, "Lacosamide", "MATCH", "9/9", "YES"])
        return json_path, csv_path

    def make_tier3_pair(self, directory: Path):
        json_path = directory / "tier3.json"
        csv_path = directory / "tier3.csv"
        rows = [
            {
                "run": 1,
                "medication": "Lacosamide",
                "comparison_raw": {"drug name": {"match": True}},
                "comparison_normalized": {"drug name": {"match": True}},
                "raw_fields_correct": 9,
                "norm_fields_correct": 9,
                "raw_all_correct": True,
                "norm_all_correct": True,
            }
        ]
        json_path.write_text(json.dumps(rows), encoding="utf-8")
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Run", "Medication", "raw_drug name", "norm_drug name"])
            writer.writerow([1, "Lacosamide", "MATCH", "MATCH"])
        return json_path, csv_path

    def test_summarize_standard_pair(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path, csv_path = self.make_standard_pair(Path(tmpdir))
            summary = run_parity.summarize_result_pair(json_path, csv_path, "baseline")
            self.assertEqual(summary["metrics"]["overall_correct"], 1)
            self.assertEqual(summary["metrics"]["fields_correct"], 9)
            self.assertIn("comparison", summary["json_top_keys"])

    def test_summarize_tier3_pair(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path, csv_path = self.make_tier3_pair(Path(tmpdir))
            summary = run_parity.summarize_result_pair(json_path, csv_path, "tier3")
            self.assertEqual(summary["metrics"]["raw"]["overall_correct"], 1)
            self.assertEqual(summary["metrics"]["normalized"]["fields_correct"], 9)

    def test_compare_metrics_reports_shape_match(self):
        standard = {
            "metrics": {"overall_correct": 1, "fields_correct": 9},
            "json_top_keys": ["a", "b"],
            "csv_header": ["x"],
        }
        comparison = run_parity.compare_metrics(standard, dict(standard), "baseline")
        self.assertEqual(comparison["overall_delta"], 0)
        self.assertTrue(comparison["json_keys_match"])
        self.assertTrue(comparison["csv_header_match"])

    def test_render_markdown_contains_each_family(self):
        summary = {
            "backend": "local",
            "category": "iv_push",
            "python_bin": "/tmp/python",
            "model_name": "model",
            "cases": [
                {
                    "family": "baseline",
                    "unified": {"json_path": "u.json"},
                    "legacy": {"json_path": "l.json"},
                    "comparison": {"overall_delta": 0, "json_keys_match": True},
                    "accepted": True,
                }
            ],
        }
        text = run_parity.render_markdown(summary)
        self.assertIn("### baseline", text)
        self.assertIn("overall_delta", text)
        self.assertIn("acceptance: `pass`", text)

    def test_render_markdown_marks_skipped_slice(self):
        summary = {
            "backend": "local",
            "category": "iv_intermittent",
            "python_bin": "/tmp/python",
            "model_name": "model",
            "cases": [
                {
                    "family": "exemplar_rag",
                    "status": "skipped",
                    "reason": "both unified and legacy entrypoints failed on this slice",
                    "unified_failure": {"message": "KeyError: 'IV intermittent (16)'"},
                    "legacy_failure": {"message": "KeyError: 'IV intermittent (16)'"},
                    "accepted": True,
                    "failures": [],
                }
            ],
        }
        text = run_parity.render_markdown(summary)
        self.assertIn("acceptance: `skipped`", text)
        self.assertIn("KeyError", text)

    def test_build_remote_cases_uses_remote_commands(self):
        cases = run_parity.build_remote_cases("iv_push", "/tmp/python")
        self.assertEqual(cases[0].family, "baseline")
        self.assertIn("--backend", cases[0].unified_cmd)
        self.assertIn("remote", cases[0].unified_cmd)
        self.assertTrue(str(cases[0].legacy_results_dir).endswith("results"))


if __name__ == "__main__":
    unittest.main()
