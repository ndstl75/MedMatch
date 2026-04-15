"""Unified baseline experiment runner."""

import os
import time
from datetime import datetime

from medmatch.core.paper_baseline import (
    PAPER_BASELINE_SHEET_CONFIG,
    build_zero_shot_prompt_pair,
    expected_keys_for_baseline_sheet,
    load_baseline_dataset,
)
from medmatch.core.io import build_flat_result_paths, ensure_results_dir, write_flat_results
from prompt_medmatch import SYSTEM_PROMPT
from medmatch.core.scorer import all_fields_match, compare_results, normalize_strict
from medmatch.experiments.common import make_backend


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
    selected = list(selected_sheets or PAPER_BASELINE_SHEET_CONFIG.keys())
    dataset = load_baseline_dataset(
        start_dir=start_dir,
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
    del score_mode
    normalizer = normalize_strict

    totals = {
        "entries": 0,
        "correct": 0,
        "fields": 0,
        "fields_correct": 0,
    }

    print(f"Backend: {backend_name} | Runs: {run_count}")
    print(f"Sheets: {', '.join(dataset.keys())}")

    for sheet_name, entries in dataset.items():
        expected_keys = expected_keys_for_baseline_sheet(sheet_name)
        rows_out = []

        print(f"\n{'=' * 72}")
        print(f"{sheet_name} ({len(entries)} entries x {run_count} runs)")
        print(f"{'=' * 72}")

        for run_index in range(1, run_count + 1):
            run_ok = 0
            print(f"\n  --- Run {run_index}/{run_count} ---")
            for index, entry in enumerate(entries, 1):
                print(f"  [{index}/{len(entries)}] {entry['medication']}...", end=" ", flush=True)
                system_prompt, user_prompt = build_zero_shot_prompt_pair(sheet_name, entry["prompt"])
                llm_output, raw_response = backend.generate_json(
                    system_prompt or SYSTEM_PROMPT,
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
