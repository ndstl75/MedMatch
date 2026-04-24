from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRE_STAGE2 = REPO_ROOT.parent / "MedMatch-pre-stage2"
# These parity tests intentionally pin the legacy CSV fixture so we can verify
# stage-2 behavior against the original pre-medmatch2 benchmark surface.
LEGACY_DATA_DIR = REPO_ROOT / "data" / "med_match"
SAMPLE_PROMPT = "A total of 6mg of adenosine (2 ml) of the 3 mg/ml vial solution intravenous was pushed once."
SAMPLE_REASONING = "stub reasoning"
SAMPLE_JSON = json.dumps({"drug name": "adenosine"}, indent=2)


sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from prompt_medmatch import (  # noqa: E402
    build_cot_extract_prompt,
    build_cot_reason_prompt,
    build_local_normalization_prompt,
    build_remote_normalization_oral_instruction,
    build_remote_normalization_prompt,
)
from medmatch.core.schema import BASELINE_SHEET_CONFIG  # noqa: E402
from scripts import probing_medmatch as new_runner  # noqa: E402


def pre_stage2_root() -> Path:
    root = Path(os.environ.get("MEDMATCH_PRE_STAGE2_WORKTREE", DEFAULT_PRE_STAGE2))
    if not root.exists():
        raise AssertionError(
            f"Missing pre-stage2 worktree at {root}. "
            "Create it with: git worktree add ../MedMatch-pre-stage2 pre-stage2"
        )
    return root


def run_pre_stage2_python(script: str) -> dict:
    root = pre_stage2_root()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(root / "src"), str(root), env.get("PYTHONPATH", "")])
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


class StubBackend:
    def generate_text(self, system_prompt, user_prompt, *, temperature=None):
        del system_prompt
        del temperature
        if "Analyze the following" in user_prompt:
            return SAMPLE_REASONING
        return "{}"


class Stage2ParityTests(unittest.TestCase):
    def test_prompt_strings_match_pre_stage2(self):
        # The medmatch2 cutover intentionally updated the active oral
        # normalization prompts to stop forcing hyphen insertion for dosage
        # forms. Keep parity coverage for every unchanged surface, but allow
        # those specific oral prompt helpers to diverge from the pre-stage2
        # snapshot.
        allowed_prompt_drift = {
            "norm_remote_oral_instruction:PO Solid (40)",
            "norm_remote_oral_prompt",
            "norm_local_oral_extract",
            "norm_local_oral_prompt",
        }
        current = {}
        for remote_mode in (False, True):
            for sheet_name in ["IV intermittent (16)", "IV push (17)", "IV continuous (16)"]:
                expected_keys = list(BASELINE_SHEET_CONFIG[sheet_name]["ground_truth_cols"].keys())
                current[f"cot_reason:{remote_mode}:{sheet_name}"] = build_cot_reason_prompt(
                    sheet_name,
                    SAMPLE_PROMPT,
                    remote_mode=remote_mode,
                )
                current[f"cot_extract:{remote_mode}:{sheet_name}"] = build_cot_extract_prompt(
                    sheet_name,
                    SAMPLE_REASONING,
                    SAMPLE_PROMPT,
                    BASELINE_SHEET_CONFIG[sheet_name]["instruction"],
                    expected_keys,
                    remote_mode=remote_mode,
                )

        current["norm_remote_oral_instruction:PO Solid (40)"] = build_remote_normalization_oral_instruction("PO Solid (40)")
        current["norm_remote_oral_instruction:PO liquid (10)"] = build_remote_normalization_oral_instruction("PO liquid (10)")
        current["norm_remote_oral_prompt"] = build_remote_normalization_prompt(SAMPLE_PROMPT, SAMPLE_JSON, family="oral")
        current["norm_remote_iv_prompt"] = build_remote_normalization_prompt(SAMPLE_PROMPT, SAMPLE_JSON, family="iv")

        oral_sheet_config = new_runner.get_local_normalization_resources("oral")
        iv_sheet_config = new_runner.get_local_normalization_resources("iv")
        current["norm_local_oral_extract"] = new_runner.build_normalization_extract_prompt(
            oral_sheet_config["PO Solid (40)"]["instruction"],
            SAMPLE_PROMPT,
            list(BASELINE_SHEET_CONFIG["PO Solid (40)"]["ground_truth_cols"].keys()),
        )
        current["norm_local_oral_prompt"] = build_local_normalization_prompt(
            SAMPLE_PROMPT,
            SAMPLE_JSON,
            family="oral",
        )
        current["norm_local_iv_extract"] = new_runner.build_normalization_extract_prompt(
            iv_sheet_config["IV push (17)"]["instruction"],
            SAMPLE_PROMPT,
            list(BASELINE_SHEET_CONFIG["IV push (17)"]["ground_truth_cols"].keys()),
        )
        current["norm_local_iv_prompt"] = build_local_normalization_prompt(
            SAMPLE_PROMPT,
            SAMPLE_JSON,
            family="iv",
        )

        old = run_pre_stage2_python(
            f"""
import importlib.util
import json
import os
import sys

from prompt_medmatch import (
    build_cot_extract_prompt,
    build_cot_reason_prompt,
    build_remote_normalization_oral_instruction,
    build_remote_normalization_prompt,
)
from medmatch.core.schema import BASELINE_SHEET_CONFIG

ROOT = os.getcwd()

def load_local_module(relative_path, module_name):
    path = os.path.join(ROOT, relative_path)
    module_dir = os.path.dirname(path)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

oral = load_local_module("scripts/legacy/local/oral_llm_normalize_local.py", "pre_stage2_oral_norm")
iv = load_local_module("scripts/legacy/local/iv_llm_normalize_local.py", "pre_stage2_iv_norm")

sample_prompt = {SAMPLE_PROMPT!r}
sample_reasoning = {SAMPLE_REASONING!r}
sample_json = {SAMPLE_JSON!r}

out = {{}}
for remote_mode in (False, True):
    for sheet_name in ["IV intermittent (16)", "IV push (17)", "IV continuous (16)"]:
        expected_keys = list(BASELINE_SHEET_CONFIG[sheet_name]["ground_truth_cols"].keys())
        out[f"cot_reason:{{remote_mode}}:{{sheet_name}}"] = build_cot_reason_prompt(
            sheet_name, sample_prompt, remote_mode=remote_mode
        )
        out[f"cot_extract:{{remote_mode}}:{{sheet_name}}"] = build_cot_extract_prompt(
            sheet_name,
            sample_reasoning,
            sample_prompt,
            BASELINE_SHEET_CONFIG[sheet_name]["instruction"],
            expected_keys,
            remote_mode=remote_mode,
        )

out["norm_remote_oral_instruction:PO Solid (40)"] = build_remote_normalization_oral_instruction("PO Solid (40)")
out["norm_remote_oral_instruction:PO liquid (10)"] = build_remote_normalization_oral_instruction("PO liquid (10)")
out["norm_remote_oral_prompt"] = build_remote_normalization_prompt(sample_prompt, sample_json, family="oral")
out["norm_remote_iv_prompt"] = build_remote_normalization_prompt(sample_prompt, sample_json, family="iv")
out["norm_local_oral_extract"] = (
    oral.SHEET_CONFIG["PO Solid (40)"]["instruction"]
    + "\\n\\nReturn one JSON object only.\\nDo not wrap the JSON in markdown.\\n"
    + "Use exactly these keys in this order: "
    + ", ".join(BASELINE_SHEET_CONFIG["PO Solid (40)"]["ground_truth_cols"].keys())
    + ".\\n\\nNow process this medication order:\\n"
    + sample_prompt
)
out["norm_local_oral_prompt"] = oral.NORMALIZE_PROMPT.format(sentence=sample_prompt, raw_json=sample_json)
out["norm_local_iv_extract"] = (
    iv.SHEET_CONFIG["IV push (17)"]["instruction"]
    + "\\n\\nReturn one JSON object only.\\nDo not wrap the JSON in markdown.\\n"
    + "Use exactly these keys in this order: "
    + ", ".join(BASELINE_SHEET_CONFIG["IV push (17)"]["ground_truth_cols"].keys())
    + ".\\n\\nNow process this medication order:\\n"
    + sample_prompt
)
out["norm_local_iv_prompt"] = iv.IV_NORMALIZE_PROMPT.format(sentence=sample_prompt, raw_json=sample_json)
print(json.dumps(out))
"""
        )

        comparable_current = {key: value for key, value in current.items() if key not in allowed_prompt_drift}
        comparable_old = {key: value for key, value in old.items() if key not in allowed_prompt_drift}

        self.assertEqual(comparable_current, comparable_old)

        for key in allowed_prompt_drift:
            self.assertNotEqual(current[key], old[key])

        self.assertIn(
            "Preserve the dosage-form wording and hyphenation from the source order.",
            current["norm_remote_oral_prompt"],
        )
        self.assertIn(
            "Preserve the dosage-form wording and hyphenation from the source order.",
            current["norm_local_oral_prompt"],
        )
        self.assertIn(
            "Copy the dosage-form wording from the order as closely as possible, preserving the source spelling and hyphenation.",
            current["norm_local_oral_extract"],
        )

    def test_cot_row_schema_matches_pre_stage2(self):
        row = new_runner.load_csv_rows(str(LEGACY_DATA_DIR), "iv_continuous", None)[0]
        current_record = new_runner.process_cot_entry(
            {"kind": "backend", "backend": StubBackend(), "temperature": 0.1},
            "local",
            row,
            1,
        )
        current_schema = {
            "keys": list(current_record.keys()),
            "types": {key: type(value).__name__ for key, value in current_record.items()},
        }

        old_schema = run_pre_stage2_python(
            """
import contextlib
import io
import json
import os
import tempfile

from medmatch.experiments import cot as mod

row = {
    "medication": "demo med",
    "prompt": "demo prompt",
    "ground_truth": {
        "drug name": "demo",
        "numerical dose": "1",
        "abbreviated unit strength of dose": "mg",
        "diluent volume": "100",
        "volume unit of measure": "mL",
        "compatible diluent type": "0.9% sodium chloride",
        "starting rate": "0.1",
        "unit of measure": "mg/kg/hr",
        "titration dose": "0.05",
        "titration unit of measure": "mg/kg/hr",
        "titration frequency": "every 5 minutes",
        "titration goal based on physiologic response, laboratory result, or assessment score": "RASS -1",
    },
}

tmp = tempfile.mkdtemp(prefix="pre-stage2-cot-")
mod.make_backend = lambda name: object()
mod.iv_sheet_config = lambda: {
    "IV continuous (16)": {
        "instruction": "demo instruction",
        "ground_truth_cols": {key: idx for idx, key in enumerate(row["ground_truth"].keys(), start=1)},
    }
}
mod.load_experiment_dataset = lambda *args, **kwargs: {"IV continuous (16)": [row]}
mod.ensure_results_dir = lambda start_dir: tmp
mod.timestamp_now = lambda: "schema"
mod.generate_text = lambda backend, system_prompt, prompt, temperature=None: "stub reasoning"
mod.generate_json = lambda backend, backend_name, system_prompt, prompt, expected_keys, use_aliases=True, temperature=None: ({key: "" for key in expected_keys}, "{}")
with contextlib.redirect_stdout(io.StringIO()):
    mod.run_cot(backend_name="local", selected_sheets=["IV continuous (16)"], num_runs=1, start_dir=os.getcwd())
payload = json.load(open(os.path.join(tmp, "IV_continuous_16_local_cot_schema.json"), "r", encoding="utf-8"))
record = payload[0]
print(json.dumps({"keys": list(record.keys()), "types": {key: type(value).__name__ for key, value in record.items()}}))
"""
        )

        self.assertEqual(current_schema, old_schema)

    def test_normalization_row_schema_matches_pre_stage2(self):
        runtime = {"kind": "backend", "backend": StubBackend(), "temperature": 0.1}

        oral_row = new_runner.load_csv_rows(str(LEGACY_DATA_DIR), "po_solid", None)[0]
        current_oral = new_runner.process_normalization_entry(runtime, "local", oral_row, 1)
        current_oral_schema = {
            "keys": list(current_oral.keys()),
            "types": {key: type(value).__name__ for key, value in current_oral.items()},
        }

        iv_row = new_runner.load_csv_rows(str(LEGACY_DATA_DIR), "iv_push", None)[0]
        current_iv = new_runner.process_normalization_entry(runtime, "local", iv_row, 1)
        current_iv_schema = {
            "keys": list(current_iv.keys()),
            "types": {key: type(value).__name__ for key, value in current_iv.items()},
        }

        old_schemas = run_pre_stage2_python(
            """
import contextlib
import io
import json
import os
import tempfile

from medmatch.experiments import tier3_normalize as mod

tmp = tempfile.mkdtemp(prefix="pre-stage2-norm-")
mod.make_backend = lambda name: object()
mod.ensure_results_dir = lambda start_dir: tmp
mod.timestamp_now = lambda: "schema"
mod.generate_json = lambda backend, backend_name, system_prompt, prompt, expected_keys, use_aliases=True, temperature=None: ({key: "" for key in expected_keys}, "{}")
mod.generate_text = lambda backend, system_prompt, prompt, temperature=None: "{}"

oral_gt = {
    "drug name": "demo",
    "numerical dose": "1",
    "abbreviated unit strength of dose": "mg",
    "amount": "1",
    "formulation": "tablet",
    "route": "by mouth",
    "frequency": "once daily",
}
iv_gt = {
    "drug name": "demo",
    "numerical dose": "1",
    "abbreviated unit strength of dose": "mg",
    "amount of volume": "1",
    "volume unit of measure": "mL",
    "concentration of solution": "1",
    "concentration unit of measure": "mg/mL",
    "formulation": "vial solution",
    "frequency": "once",
}

def read_schema(filename):
    payload = json.load(open(filename, "r", encoding="utf-8"))
    record = payload[0]
    return {"keys": list(record.keys()), "types": {key: type(value).__name__ for key, value in record.items()}}

mod.load_experiment_dataset = lambda *args, **kwargs: {"PO Solid (40)": [{"medication": "oral demo", "prompt": "oral prompt", "ground_truth": oral_gt}]}
with contextlib.redirect_stdout(io.StringIO()):
    mod.run_tier3(backend_name="local", category="po_solid", num_runs=1, start_dir=os.getcwd())
oral_schema = read_schema(os.path.join(tmp, "PO_Solid_40_local_llm_norm_schema.json"))

mod.load_experiment_dataset = lambda *args, **kwargs: {"IV push (17)": [{"medication": "iv demo", "prompt": "iv prompt", "ground_truth": iv_gt}]}
with contextlib.redirect_stdout(io.StringIO()):
    mod.run_tier3(backend_name="local", category="iv_push", num_runs=1, start_dir=os.getcwd())
iv_schema = read_schema(os.path.join(tmp, "IV_push_17_local_iv_llm_norm_schema.json"))

print(json.dumps({"oral": oral_schema, "iv": iv_schema}))
"""
        )

        self.assertEqual(current_oral_schema, old_schemas["oral"])
        self.assertEqual(current_iv_schema, old_schemas["iv"])


if __name__ == "__main__":
    unittest.main()
