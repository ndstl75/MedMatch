from __future__ import annotations

import csv
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MEDMATCH2_PO_SOLID = REPO_ROOT / "data" / "medmatch2" / "med_match - po_solid.csv"
MEDMATCH2_PO_SOLID_NO_ROUTE = REPO_ROOT / "data" / "medmatch2" / "med_match - po_solid_no_route.csv"

TARGET_FORMULATION_ROWS = {
    "Duloxetine": {
        "formulation": "delayed release capsule",
        "hyphenated_fragment": "delayed-release",
    },
    "Omeprazole": {
        "formulation": "delayed release tablet",
        "hyphenated_fragment": "delayed-release",
    },
    "Ondansetron": {
        "formulation": "orally disintegrating tablet",
        "hyphenated_fragment": "orally-disintegrating",
    },
    "Quetiapine": {
        "formulation": "extended release tablet",
        "hyphenated_fragment": "extended-release",
    },
    "Tolterodine": {
        "formulation": "extended release capsule",
        "hyphenated_fragment": "extended-release",
    },
}


class Medmatch2DatasetConsistencyTests(unittest.TestCase):
    def load_rows(self, path: Path):
        with path.open(newline="") as handle:
            return {row["Medication"]: row for row in csv.DictReader(handle)}

    def test_po_solid_prompt_text_matches_unhyphenated_formulation_rows(self):
        rows = self.load_rows(MEDMATCH2_PO_SOLID)

        for medication, expectation in TARGET_FORMULATION_ROWS.items():
            with self.subTest(medication=medication):
                row = rows[medication]
                self.assertEqual(row["formulation"], expectation["formulation"])
                self.assertNotIn(expectation["hyphenated_fragment"], row["Medication JSON (ground truth)"])
                self.assertNotIn(expectation["hyphenated_fragment"], row["Medication prompt (sentence format)"])

    def test_po_solid_no_route_rows_match_route_bearing_dosage_form_spelling(self):
        route_rows = self.load_rows(MEDMATCH2_PO_SOLID)
        no_route_rows = self.load_rows(MEDMATCH2_PO_SOLID_NO_ROUTE)

        for medication, expectation in TARGET_FORMULATION_ROWS.items():
            with self.subTest(medication=medication):
                route_row = route_rows[medication]
                no_route_row = no_route_rows[medication]
                self.assertNotIn(
                    expectation["hyphenated_fragment"],
                    no_route_row["Medication JSON without route"],
                )
                self.assertIn(
                    expectation["formulation"],
                    no_route_row["Medication JSON without route"],
                )
                self.assertIn(
                    expectation["formulation"],
                    route_row["Medication JSON (ground truth)"],
                )


if __name__ == "__main__":
    unittest.main()
