"""
Convert MedMatch JSONL outputs to per-dataset comparison tables.

Usage:
    python convert_table.py --prompting_type zero  # Process zero-shot results
    python convert_table.py --prompting_type few   # Process few-shot results

For each dataset (po, iv_i, iv_p, iv_c):
- Load ground-truth CSV from the current MedMatch CSV directory.
- Load all JSONL outputs in results/<dataset_version>/{prompting_type}-shot/.
- Convert JSON responses to readable strings by concatenating values only.
- Pivot responses into columns per model/run (e.g., llama3 run1/run2/run3).
- Write one CSV per dataset into results/<dataset_version>/{prompting_type}-shot-tables/.

From the ``results/`` directory (or pass paths via script defaults):

    python convert_table.py --prompting_type zero
    python convert_table.py --prompting_type few
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medmatch.core.paths import current_results_root, default_data_dir

# Templates for converting JSON back to string format
STRING_TEMPLATES = {
    "po_solid": "[drug_name][numerical_dose][abbreviated_unit_strength_of_dose][amount][formulation][route][frequency]",
    "po_liquid": "[drug_name][numerical_dose][abbreviated_unit_strength_of_dose][numerical_volume][volume_unit_of_measure][concentration_of_formulation][formulation_unit_of_measure][formulation][route][frequency]",
    "iv_intermit": "[drug_name][numerical_dose][abbreviated_unit_strength_of_dose][amount_of_diluent_volume][volume_unit_of_measure][compatible_diluent_type] intravenous infused over [infusion_time] [frequency]",
    "iv_push": "[drug_name][numerical_dose][abbreviated_unit_strength_of_dose][amount_of_volume][volume_unit_of_measure] of the [concentration_of_solution][concentration_unit_of_measure][formulation] intravenous push [frequency]",
    "iv_continuous": "[drug_name][numerical_dose][abbreviated_unit_strength_of_dose] [diluent_volume][volume_unit_of_measure] in [compatible_diluent_type] continuous intravenous infusion starting at [starting_rate][unit_of_measure] titrated by [titration_dose][titration_unit_of_measure][titration_frequency] minutes to achieve a goal [titration_goal_based_on_physiologic_response_laboratory_result_or_assessment_score]",
}

# JSON responses are converted to readable strings by concatenating values only
DATASET_CSVS = {
    "po_solid": {"file": "med_match - po_solid.csv", "gt_col": "Medication JSON (ground truth)", "prompt_col": "Medication prompt (sentence format)"},
    "po_liquid": {"file": "med_match - po_liquid.csv", "gt_col": "Medication JSON (ground truth)", "prompt_col": "Medication prompt (sentence format)"},
    "iv_intermit": {"file": "med_match - iv_i.csv", "gt_col": "Medication JSON", "prompt_col": "Medication prompt (sentence format)"},
    "iv_push": {"file": "med_match - iv_p.csv", "gt_col": "Medication JSON", "prompt_col": "Medication prompt (sentence format)"},
    "iv_continuous": {"file": "med_match - iv_c.csv", "gt_col": "Medication JSON", "prompt_col": "Medication prompt (sentence format)"},
}


def jsonl_to_records(path: Path):
    """Load a JSONL file into a list of dict records."""
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_all_jsonl(base_dir: Path):
    """Aggregate all records from JSONL files in base_dir."""
    all_records = []
    for jsonl_path in sorted(base_dir.glob("*.jsonl")):
        all_records.extend(jsonl_to_records(jsonl_path))
    return pd.DataFrame(all_records) if all_records else pd.DataFrame()


def normalize_model_name(model: str) -> str:
    return model.split("/")[-1].replace(" ", "_").replace("-", "_")


def clean_response(resp: str, dataset: str = None) -> str:
    """Strip code fences/backticks or language tags from model output and normalize."""
    if not isinstance(resp, str):
        return resp
    text = resp.strip()
    # Remove BOM / zero-width / nbsp
    text = text.replace("\ufeff", "").replace("\u200b", "").replace("\u00a0", " ")
    # Full fence ```...```
    fence_pattern = re.compile(r"^```(?:\w+)?\s*(.*?)\s*```$", re.DOTALL)
    match = fence_pattern.match(text)
    if match:
        text = match.group(1).strip()
    else:
        # Leading fence without closing
        text = re.sub(r"^```(?:\w+)?\s*", "", text, flags=re.DOTALL)
        # Trailing fence without leading
        text = re.sub(r"\s*```$", "", text)
    # Drop leading language tags like "json" or "python"
    text = re.sub(r"^(json|python)\s+", "", text, flags=re.IGNORECASE)
    # Remove common prefatory markers
    text = re.sub(r"^(final answer\s*:|answer\s*:|output\s*:)\s*", "", text, flags=re.IGNORECASE)
    # Remove any remaining fenced blocks anywhere in the string
    text = re.sub(r"```(?:\w+)?", "", text)
    text = re.sub(r"```", "", text)
    # Unescape common escaped characters
    text = text.replace("\\n", " ").replace("\\t", " ").replace('\\"', '"').replace("\\'", "'")
    # Strip stray backticks
    text = text.replace("`", " ")
    # Normalize dashes (convert en/em dash to hyphen)
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    # Strip surrounding quotes/backticks if present
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    if text.startswith("`") and text.endswith("`") and len(text) >= 2:
        text = text[1:-1].strip()
    # Collapse newlines/tabs to single spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Try to parse as JSON and convert to string format if dataset is provided
    if dataset and (text.startswith("{") or text.startswith("{{")) and (text.endswith("}") or text.endswith("}}")):
        try:
            # Handle double braces that some models output
            if text.startswith("{{") and text.endswith("}}"):
                text = text[1:-1]  # Remove outer braces
            json_data = json.loads(text)
            text = json_to_string(json_data, dataset)
        except (json.JSONDecodeError, KeyError):
            # If JSON parsing fails, keep the original text
            pass

    return text


def json_to_string(json_data: dict, dataset: str) -> str:
    """Convert JSON medication data to readable string by concatenating values only (no keys)."""
    if not json_data:
        return ""

    # Create readable string from values only (no keys)
    parts = []
    for value in json_data.values():
        if value is not None and value != "":
            parts.append(str(value))

    return " ".join(parts)


def build_dataset_table(df_runs: pd.DataFrame, dataset: str, meta: dict, data_dir: Path) -> pd.DataFrame:
    """Join ground truth with pivoted model/run responses for one dataset."""
    csv_path = data_dir / meta["file"]
    gt_df = pd.read_csv(csv_path)
    gt_df["__row_order"] = range(len(gt_df))
    gt = gt_df.rename(
        columns={
            "Medication": "Medication",
            meta["gt_col"]: "Ground Truth",
            meta["prompt_col"]: "Prompt",
        }
    )

    subset = df_runs[df_runs["dataset"] == dataset].copy()
    if subset.empty:
        return pd.DataFrame()

    subset["model_run"] = subset.apply(
        lambda r: f"{normalize_model_name(r['model'])}_run{int(r['run'])}", axis=1
    )
    subset["response"] = subset["response"].apply(lambda x: clean_response(x, dataset))
    subset = subset.drop_duplicates(
        subset=["medication", "prompt", "ground_truth", "model_run"]
    ).sort_values(["medication", "prompt", "ground_truth", "model_run"])

    pivot = subset.pivot_table(
        index=["medication", "prompt", "ground_truth"],
        columns="model_run",
        values="response",
        aggfunc="first",
    ).reset_index()

    pivot = pivot.rename(
        columns={
            "medication": "Medication",
            "prompt": "Prompt",
            "ground_truth": "Ground Truth",
        }
    )

    merged = gt.merge(pivot, on=["Medication", "Prompt", "Ground Truth"], how="left")
    # Order columns: core fields, then model_run columns in deterministic order, preserve CSV row order
    fixed = ["Medication", "Ground Truth", "Prompt"]
    dynamic = sorted([c for c in merged.columns if c not in fixed and c != "__row_order"])
    merged = merged[fixed + dynamic + ["__row_order"]]
    merged = merged.sort_values("__row_order").drop(columns="__row_order").reset_index(drop=True)
    return merged


def main():
    parser = argparse.ArgumentParser(description="Convert MedMatch JSONL outputs to comparison tables.")
    parser.add_argument("--prompting_type", choices=["zero", "few"], required=True,
                       help="Prompting type: 'zero' for zero-shot, 'few' for few-shot")
    args = parser.parse_args()

    # Set up directories based on prompting type
    results_root = Path(current_results_root())
    input_dir = results_root / f"{args.prompting_type}-shot"
    output_dir = results_root / f"{args.prompting_type}-shot-tables"
    output_dir.mkdir(exist_ok=True)

    print(f"Processing {args.prompting_type}-shot results from {input_dir}")
    print(f"Output tables will be saved to {output_dir}")

    df_runs = load_all_jsonl(input_dir)
    if df_runs.empty:
        print(f"No JSONL files found in {input_dir}; nothing to convert.")
        return

    data_dir = Path(default_data_dir())

    for dataset, meta in DATASET_CSVS.items():
        table = build_dataset_table(df_runs, dataset, meta, data_dir)
        if table.empty:
            print(f"No records for dataset '{dataset}', skipping.")
            continue
        csv_path = output_dir / f"{dataset}_table.csv"
        table.to_csv(csv_path, index=False)
        print(f"Wrote {csv_path.name} ({len(table)} rows)")


if __name__ == "__main__":
    main()
