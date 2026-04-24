from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from medmatch.core.schema import BASELINE_SHEET_CONFIG
from prompt_medmatch import (
    build_local_normalization_oral_instruction,
    build_local_normalization_prompt,
    build_po_solid_messages_one_shot_multi_turn,
    build_remote_normalization_oral_instruction,
    build_remote_normalization_prompt,
)


GABAPENTIN_SENTENCE = "Gabapentin 2 capsules (total dose 600mg) by mouth three times daily."
RAW_JSON = """{
  "drug name": "gabapentin",
  "numerical dose": "300",
  "abbreviated unit strength of dose": "mg",
  "amount": "2",
  "formulation": "capsules",
  "route": "by mouth",
  "frequency": "three times daily"
}"""


class POSolidTotalDosePromptingTests(unittest.TestCase):
    def test_active_schema_instruction_describes_total_dose(self):
        instruction = BASELINE_SHEET_CONFIG["PO Solid (40)"]["instruction"]
        self.assertIn("total drug amount for that administration", instruction)
        self.assertIn("2 capsules of 500 mg each -> numerical dose 1000, amount 2", instruction)

    def test_one_shot_example_teaches_multi_unit_total_dose(self):
        messages = build_po_solid_messages_one_shot_multi_turn("dummy oral prompt")
        assistant_example = messages[2]["content"]
        self.assertIn('"drug_name": "gabapentin"', assistant_example)
        self.assertIn('"numerical_dose": "600"', assistant_example)
        self.assertIn('"amount": "2"', assistant_example)

    def test_normalization_extract_instructions_match_total_dose_semantics(self):
        remote_instruction = build_remote_normalization_oral_instruction("PO Solid (40)")
        local_instruction = build_local_normalization_oral_instruction("PO Solid (40)")
        for instruction in (remote_instruction, local_instruction):
            self.assertIn("total drug amount for that administration", instruction)
            self.assertIn("2 capsules of 500 mg each -> numerical dose 1000, amount 2", instruction)

    def test_oral_normalizers_allow_explicit_total_dose_rewrite(self):
        remote_prompt = build_remote_normalization_prompt(GABAPENTIN_SENTENCE, RAW_JSON, family="oral")
        local_prompt = build_local_normalization_prompt(GABAPENTIN_SENTENCE, RAW_JSON, family="oral")
        for prompt in (remote_prompt, local_prompt):
            self.assertIn("Oral solid total dose", prompt)
            self.assertIn("set numerical dose to the total drug amount per administration", prompt)
            self.assertIn("keep amount as the dosage-unit count", prompt)
            self.assertIn("Do not infer missing strengths or counts.", prompt)


if __name__ == "__main__":
    unittest.main()
