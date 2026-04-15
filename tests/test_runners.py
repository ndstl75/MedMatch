import csv
import json
import os
import tempfile
import types
import unittest
from unittest import mock

from medmatch.experiments import cot, exemplar_rag, tier3_normalize


class FakeBackend:
    def generate_text(self, system_prompt, prompt, *, temperature=None):
        del system_prompt, temperature
        prompt_lower = prompt.lower()
        if "normalize the wording" in prompt_lower:
            if '"drug name"' in prompt:
                return '{"drug name": "lacosamide", "numerical dose": 200, "abbreviated unit strength of dose": "mg", "amount of volume": 20, "volume unit of measure": "mL", "concentration of solution": 10, "concentration unit of measure": "mg/mL", "formulation": "vial solution", "frequency": "once daily"}'
        if "analyze the following" in prompt_lower:
            return "Reasoning: extract the known IV push fields carefully."
        return '{"drug name": "lacosamide", "numerical dose": 200, "abbreviated unit strength of dose": "mg", "amount of volume": 20, "volume unit of measure": "mL", "concentration of solution": 10, "concentration unit of measure": "mg/mL", "formulation": "vial solution", "frequency": "once daily"}'

    def generate_json(self, system_prompt, prompt, expected_keys, *, temperature=None):
        del system_prompt, prompt, temperature
        obj = {
            "drug name": "lacosamide",
            "numerical dose": 200,
            "abbreviated unit strength of dose": "mg",
            "amount of volume": 20,
            "volume unit of measure": "mL",
            "concentration of solution": 10,
            "concentration unit of measure": "mg/mL",
            "formulation": "vial solution",
            "frequency": "once daily",
        }
        return {key: obj.get(key, "") for key in expected_keys}, json.dumps(obj)


def sample_iv_push_dataset():
    return {
        "IV push (17)": [
            {
                "medication": "Lacosamide",
                "prompt": "Lacosamide 200 mg in 20 mL vial solution IV push once daily.",
                "ground_truth": {
                    "drug name": "lacosamide",
                    "numerical dose": 200,
                    "abbreviated unit strength of dose": "mg",
                    "amount of volume": 20,
                    "volume unit of measure": "mL",
                    "concentration of solution": 10,
                    "concentration unit of measure": "mg/mL",
                    "formulation": "vial solution",
                    "frequency": "once daily",
                },
            }
        ]
    }


class RunnerTests(unittest.TestCase):
    def make_local_common(self):
        return types.SimpleNamespace(
            IV_PUSH_INSTRUCTION="IV push base instruction",
            IV_CONTINUOUS_INSTRUCTION="IV continuous base instruction",
            IV_EXEMPLAR_BANK={"IV push (17)": [{"label": "x", "prompt": "example", "output": {"drug name": "lacosamide"}}]},
            retrieve_topk_examples=lambda prompt, bank, k=2: bank[:k],
            render_examples=lambda examples: "Example 1 input:\nexample\nExample 1 output:\n{}",
            SHEET_CONFIG={
                "IV push (17)": {
                    "instruction": "IV push base instruction",
                    "prompt_col": 3,
                    "ground_truth_cols": {
                        "drug name": 4,
                        "numerical dose": 5,
                        "abbreviated unit strength of dose": 6,
                        "amount of volume": 7,
                        "volume unit of measure": 8,
                        "concentration of solution": 9,
                        "concentration unit of measure": 10,
                        "formulation": 11,
                        "frequency": 12,
                    },
                }
            },
            NORMALIZE_PROMPT="Normalize:\n{sentence}\n{raw_json}",
            IV_NORMALIZE_PROMPT="Normalize IV:\n{sentence}\n{raw_json}",
        )

    def test_cot_runner_writes_expected_result_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(cot, "make_backend", return_value=FakeBackend()), \
                 mock.patch.object(cot, "load_experiment_dataset", return_value=sample_iv_push_dataset()), \
                 mock.patch.object(cot.time, "sleep", return_value=None):
                cot.run_cot(
                    backend_name="local",
                    selected_sheets=["IV push (17)"],
                    num_runs=1,
                    start_dir=tmpdir,
                )

            result_files = [name for name in os.listdir(os.path.join(tmpdir, "results")) if name.endswith(".json")]
            self.assertEqual(len(result_files), 1)
            with open(os.path.join(tmpdir, "results", result_files[0]), encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(len(payload), 1)
            self.assertIn("reasoning", payload[0])
            self.assertIn("comparison", payload[0])
            self.assertEqual(payload[0]["fields_total"], 9)

    def test_exemplar_rag_runner_writes_csv_and_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(exemplar_rag, "make_backend", return_value=FakeBackend()), \
                 mock.patch.object(exemplar_rag, "load_experiment_dataset", return_value=sample_iv_push_dataset()), \
                 mock.patch.object(exemplar_rag, "load_legacy_local_module", return_value=self.make_local_common()), \
                 mock.patch.object(exemplar_rag.time, "sleep", return_value=None):
                exemplar_rag.run_exemplar_rag(
                    backend_name="local",
                    selected_sheets=["IV push (17)"],
                    num_runs=1,
                    start_dir=tmpdir,
                )

            results_dir = os.path.join(tmpdir, "results")
            self.assertTrue(any(name.endswith(".json") for name in os.listdir(results_dir)))
            csv_name = next(name for name in os.listdir(results_dir) if name.endswith(".csv"))
            with open(os.path.join(results_dir, csv_name), newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                header = next(reader)
                first_row = next(reader)
            self.assertIn("Medication", header)
            self.assertEqual(first_row[1], "Lacosamide")

    def test_tier3_runner_writes_raw_and_normalized_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(tier3_normalize, "make_backend", return_value=FakeBackend()), \
                 mock.patch.object(tier3_normalize, "load_experiment_dataset", return_value=sample_iv_push_dataset()), \
                 mock.patch.object(tier3_normalize, "load_legacy_local_module", return_value=self.make_local_common()), \
                 mock.patch.object(tier3_normalize.time, "sleep", return_value=None):
                tier3_normalize.run_tier3(
                    backend_name="local",
                    category="iv_push",
                    num_runs=1,
                    start_dir=tmpdir,
                )

            json_name = next(name for name in os.listdir(os.path.join(tmpdir, "results")) if name.endswith(".json"))
            with open(os.path.join(tmpdir, "results", json_name), encoding="utf-8") as handle:
                payload = json.load(handle)
            row = payload[0]
            self.assertIn("raw_output", row)
            self.assertIn("normalized_output", row)
            self.assertTrue(row["raw_all_correct"])
            self.assertTrue(row["norm_all_correct"])


if __name__ == "__main__":
    unittest.main()
