import json
import tempfile
import unittest
from pathlib import Path

from scripts import run_single


class RunSingleTests(unittest.TestCase):
    def test_parse_input_file_supports_legacy_mode_line(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "input.txt"
            path.write_text(
                "\n".join(
                    [
                        "IV Push",
                        "Famotidine 20 mg, 2 mL of a 20 mg/2 mL vial, was administered twice daily via intravenous push.",
                        json.dumps({"drug name": "famotidine"}),
                        "both",
                    ]
                ),
                encoding="utf-8",
            )
            parsed = run_single.parse_input_file(path)
            self.assertEqual(parsed["category"], "iv_push")
            self.assertEqual(parsed["ground_truth_json"]["drug name"], "famotidine")

    def test_normalize_category_name_accepts_sheet_labels(self):
        self.assertEqual(run_single.normalize_category_name("IV push (17)"), "iv_push")
        self.assertEqual(run_single.normalize_category_name("PO liquid"), "po_liquid")

    def test_flatten_output_preserves_expected_order(self):
        text = run_single.flatten_output(
            ["drug name", "numerical dose", "frequency"],
            {"frequency": "once", "drug name": "adenosine", "numerical dose": 6},
        )
        self.assertEqual(text, "adenosine 6 once")


if __name__ == "__main__":
    unittest.main()
