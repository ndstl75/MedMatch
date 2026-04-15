import unittest

from medmatch.core.schema import (
    BASELINE_SHEET_CONFIG,
    IV_BASELINE_SHEET_CONFIG,
    ORAL_BASELINE_SHEET_CONFIG,
    PO_SOLID_INSTRUCTION,
    IV_PUSH_INSTRUCTION,
    expected_keys_for_sheet,
)
from medmatch.experiments.common import coerce_output_object, compare_results_backend


class SchemaTests(unittest.TestCase):
    def test_sheet_config_partition_is_stable(self):
        self.assertEqual(
            set(BASELINE_SHEET_CONFIG),
            {
                "PO Solid (40)",
                "PO liquid (10)",
                "IV intermittent (16)",
                "IV push (17)",
                "IV continuous (16)",
            },
        )
        self.assertEqual(set(ORAL_BASELINE_SHEET_CONFIG), {"PO Solid (40)", "PO liquid (10)"})
        self.assertEqual(
            set(IV_BASELINE_SHEET_CONFIG),
            {"IV intermittent (16)", "IV push (17)", "IV continuous (16)"},
        )

    def test_expected_keys_are_ordered_and_complete(self):
        self.assertEqual(
            expected_keys_for_sheet("IV push (17)"),
            [
                "drug name",
                "numerical dose",
                "abbreviated unit strength of dose",
                "amount of volume",
                "volume unit of measure",
                "concentration of solution",
                "concentration unit of measure",
                "formulation",
                "frequency",
            ],
        )

    def test_core_prompt_snippets_remain_present(self):
        self.assertIn("by mouth", PO_SOLID_INSTRUCTION)
        self.assertIn("intravenous push", IV_PUSH_INSTRUCTION)
        self.assertIn("do not fabricate", PO_SOLID_INSTRUCTION.lower())

    def test_coerce_output_object_handles_aliases_and_strict_mode(self):
        parsed = {
            "Dose Unit": "mg",
            "volume unit": "mL",
            "drug name": "famotidine",
        }
        expected_keys = ["drug name", "abbreviated unit strength of dose", "volume unit of measure"]
        aliased = coerce_output_object(parsed, expected_keys, use_aliases=True)
        strict = coerce_output_object(parsed, expected_keys, use_aliases=False)

        self.assertEqual(aliased["abbreviated unit strength of dose"], "mg")
        self.assertEqual(aliased["volume unit of measure"], "mL")
        self.assertEqual(strict["abbreviated unit strength of dose"], "")
        self.assertEqual(strict["volume unit of measure"], "")

    def test_compare_results_backend_uses_expected_normalization_mode(self):
        ground_truth = {"unit of measure": "mg/hour", "drug name": "Drug-A"}
        output = {"unit of measure": "mg/hr", "drug name": "Drug—A"}

        remote = compare_results_backend(output, ground_truth, "remote")
        local = compare_results_backend(output, ground_truth, "local")

        self.assertTrue(remote["unit of measure"]["match"])
        self.assertTrue(remote["drug name"]["match"])
        self.assertFalse(local["unit of measure"]["match"])
        self.assertFalse(local["drug name"]["match"])


if __name__ == "__main__":
    unittest.main()
