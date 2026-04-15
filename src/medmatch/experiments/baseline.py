"""Unified baseline experiment runner."""

import os
import time
from datetime import datetime

from medmatch.core.dataset import load_dataset, resolve_project_file, selected_sheets_from_env
from medmatch.core.io import build_flat_result_paths, ensure_results_dir, write_flat_results
from medmatch.core.schema import BASELINE_SHEET_CONFIG, IV_BASELINE_SHEET_CONFIG, SYSTEM_PROMPT
from medmatch.core.scorer import all_fields_match, compare_results, normalize_relaxed, normalize_strict
from medmatch.llm.config import is_remote_backend
from medmatch.experiments.common import make_backend


IV_PROMPT_ADDENDUM = {
    "IV intermittent (16)": """\
Few-shot examples:
Example 1:
Cefepime 2000 mg was delivered as a 30 minute intravenous infusion, prepared in 100 mL of 0.9% sodium chloride and administered every 8 hours.
{ "drug name": "cefepime", "numerical dose": 2000, "abbreviated unit strength of dose": "mg", "amount of diluent volume": 100, "volume unit of measure": "mL", "compatible diluent type": "0.9% sodium chloride", "infusion time": "30 minutes", "frequency": "every 8 hours" }
""",
    "IV push (17)": """\
Few-shot examples:
Example 1:
Famotidine 20 mg, 2 mL of a 20 mg/2 mL vial, was administered twice daily via intravenous push.
{ "drug name": "famotidine", "numerical dose": 20, "abbreviated unit strength of dose": "mg", "amount of volume": 2, "volume unit of measure": "mL", "concentration of solution": 10, "concentration unit of measure": "mg/mL", "formulation": "vial solution", "frequency": "twice daily" }

Example 2:
Dexamethasone 6 mg, equivalent to 0.6 mL from a 10 mg/mL vial solution, was pushed intravenously daily.
{ "drug name": "dexamethasone", "numerical dose": 6, "abbreviated unit strength of dose": "mg", "amount of volume": 0.6, "volume unit of measure": "mL", "concentration of solution": 10, "concentration unit of measure": "mg/mL", "formulation": "vial solution", "frequency": "once daily" }
""",
    "IV continuous (16)": """\
Few-shot examples:
Example 1:
The patient received a continuous intravenous infusion of midazolam at a starting rate of 0.5 mg/hr (1 mg/mL in 100 mL of 0.9% sodium chloride), and the infusion was titrated by 0.5 mg/hr every 5 minutes to achieve a RASS goal of -4 to -5.
{ "drug name": "midazolam", "numerical dose": 100, "abbreviated unit strength of dose": "mg", "diluent volume": 100, "volume unit of measure": "mL", "compatible diluent type": "0.9% sodium chloride", "starting rate": 0.5, "unit of measure": "mg/hr", "titration dose": 0.5, "titration unit of measure": "mg/hr", "titration frequency": "every 5 minutes", "titration goal based on physiologic response, laboratory result, or assessment score": "RASS of -4 to -5" }

Example 2:
Cisatracurium 200 mg/100 mL 0.9% sodium chloride was initiated at 2 mcg/kg/min and titrated by 25-50% every 2 minutes to achieve ventilator synchrony.
{ "drug name": "cisatracurium", "numerical dose": 200, "abbreviated unit strength of dose": "mg", "diluent volume": 100, "volume unit of measure": "mL", "compatible diluent type": "0.9% sodium chloride", "starting rate": 2, "unit of measure": "mcg/kg/min", "titration dose": "25-50", "titration unit of measure": "%", "titration frequency": "every 2 minutes", "titration goal based on physiologic response, laboratory result, or assessment score": "ventilator synchrony" }
""",
}
def build_instruction(sheet_name, base_instruction, expected_keys, backend_name):
    extra_lines = [
        "Return one JSON object only.",
        "Do not wrap the JSON in markdown.",
        f"Use exactly these keys in this order: {', '.join(expected_keys)}.",
    ]

    if backend_name == "local" and sheet_name == "IV push (17)":
        extra_lines.extend(
            [
                'If the source says vial or vial solution, keep the formulation as "vial solution".',
                'If the source says daily for an IV push, output frequency as "once daily" only when the sentence clearly means once per day.',
            ]
        )
    elif backend_name == "local" and sheet_name == "IV continuous (16)":
        extra_lines.extend(
            [
                "For continuous infusions, keep titration fields empty only when the sentence is truly non-titratable.",
                'Use "every 1 minute" rather than "every minute" when the source says every minute.',
                "Copy titration goals tightly and do not add extra filler wording.",
            ]
        )

    addendum = IV_PROMPT_ADDENDUM.get(sheet_name, "") if sheet_name in IV_BASELINE_SHEET_CONFIG else ""
    parts = [base_instruction, "", *extra_lines]
    if addendum:
        parts.extend(["", addendum])
    return "\n".join(parts)


def run_baseline(
    *,
    backend_name,
    selected_sheets=None,
    max_entries_per_sheet=0,
    num_runs=None,
    results_root=None,
    score_mode=None,
    start_dir=None,
):
    backend = make_backend(backend_name)
    selected = selected_sheets_from_env(selected_sheets or BASELINE_SHEET_CONFIG.keys())
    xlsx_path = resolve_project_file("MedMatch Dataset for Experiment_ Final.xlsx", start_dir=start_dir)
    dataset = load_dataset(
        xlsx_path,
        BASELINE_SHEET_CONFIG,
        selected_sheets=selected,
        max_entries_per_sheet=max_entries_per_sheet,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = ensure_results_dir(
        results_root
        or os.path.join(os.path.abspath(start_dir or os.getcwd()), "results")
    )
    run_count = int(os.environ.get("MEDMATCH_NUM_RUNS", "3") if num_runs is None else num_runs)
    sleep_between_calls = float(os.environ.get("MEDMATCH_SLEEP_SECONDS", "1"))
    normalizer = normalize_relaxed if is_remote_backend(score_mode or backend_name) else normalize_strict

    totals = {
        "entries": 0,
        "correct": 0,
        "fields": 0,
        "fields_correct": 0,
    }

    print(f"Backend: {backend_name} | Runs: {run_count}")
    print(f"Sheets: {', '.join(dataset.keys())}")

    for sheet_name, entries in dataset.items():
        config = BASELINE_SHEET_CONFIG[sheet_name]
        expected_keys = list(config["ground_truth_cols"].keys())
        rows_out = []

        print(f"\n{'=' * 72}")
        print(f"{sheet_name} ({len(entries)} entries x {run_count} runs)")
        print(f"{'=' * 72}")

        for run_index in range(1, run_count + 1):
            run_ok = 0
            print(f"\n  --- Run {run_index}/{run_count} ---")
            for index, entry in enumerate(entries, 1):
                print(f"  [{index}/{len(entries)}] {entry['medication']}...", end=" ", flush=True)
                instruction = build_instruction(
                    sheet_name,
                    config["instruction"],
                    expected_keys,
                    backend_name,
                )
                user_prompt = f"{instruction}\n\nNow process this medication order:\n{entry['prompt']}"
                llm_output, raw_response = backend.generate_json(
                    SYSTEM_PROMPT,
                    user_prompt,
                    expected_keys,
                )
                comparison = compare_results(
                    llm_output,
                    entry["ground_truth"],
                    normalizer=normalizer,
                )
                fields_correct = sum(1 for value in comparison.values() if value["match"])
                entry_all_correct = all_fields_match(comparison)
                print(
                    f"{fields_correct}/{len(expected_keys)} fields"
                    + (" [ALL CORRECT]" if entry_all_correct else "")
                )

                totals["entries"] += 1
                totals["fields"] += len(expected_keys)
                totals["fields_correct"] += fields_correct
                if entry_all_correct:
                    totals["correct"] += 1
                    run_ok += 1

                rows_out.append(
                    {
                        "run": run_index,
                        "medication": entry["medication"],
                        "prompt": entry["prompt"],
                        "ground_truth": entry["ground_truth"],
                        "llm_output": llm_output,
                        "raw_response": raw_response,
                        "comparison": comparison,
                        "fields_correct": fields_correct,
                        "fields_total": len(expected_keys),
                        "all_fields_correct": entry_all_correct,
                    }
                )
                time.sleep(sleep_between_calls)

            if entries:
                print(f"\n  Run {run_index}: {run_ok}/{len(entries)} = {run_ok / len(entries) * 100:.1f}%")

        json_path, csv_path = build_flat_result_paths(results_dir, sheet_name, f"{backend_name}_baseline", timestamp)
        write_flat_results(rows_out, expected_keys, json_path, csv_path)
        print(f"  Saved: {json_path}")
        print(f"  Saved: {csv_path}")

    field_accuracy = (totals["fields_correct"] / totals["fields"] * 100) if totals["fields"] else 0
    overall_accuracy = (totals["correct"] / totals["entries"] * 100) if totals["entries"] else 0
    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"  Field accuracy: {totals['fields_correct']}/{totals['fields']} ({field_accuracy:.1f}%)")
    print(f"  Overall accuracy: {totals['correct']}/{totals['entries']} ({overall_accuracy:.1f}%)")
