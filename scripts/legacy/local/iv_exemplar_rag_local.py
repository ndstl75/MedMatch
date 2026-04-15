#!/usr/bin/env python3
"""
Clean IV exemplar-RAG experiment for local Gemma/Ollama.

This script preserves the exact-match task:
- IV sheets only
- exemplar retrieval only
- no post-hoc rewriting
- strict exact comparison
"""

import csv
import json
import os
import time
from datetime import datetime

from iv_clean_common_local import (
    DEFAULT_SHEETS,
    EXPECTED_KEYS_BY_SHEET,
    IV_CONTINUOUS_INSTRUCTION,
    IV_EXEMPLAR_BANK,
    IV_PUSH_INSTRUCTION,
    MODEL_NAME,
    SLEEP_BETWEEN_CALLS,
    TEMPERATURE,
    all_fields_match,
    call_local_model,
    compare_results_exact,
    coerce_output_object,
    load_dataset,
    parse_json_response,
    render_examples,
    retrieve_topk_examples,
    selected_sheets_from_env,
)
from local_llm import resolve_project_file


NUM_RUNS = int(os.environ.get("MEDMATCH_NUM_RUNS", "3"))  # triplicate per paper
MAX_ENTRIES_PER_SHEET = int(os.environ.get("MEDMATCH_MAX_ENTRIES", "5"))
TOP_K_EXAMPLES = int(os.environ.get("MEDMATCH_RAG_TOP_K", "2"))

SYSTEM_PROMPT = (
    "You are a clinical pharmacist who formats medication orders. "
    "Only output the MedMatch JSON format."
)


def build_instruction(sheet_name, prompt, expected_keys):
    base = IV_PUSH_INSTRUCTION if sheet_name == "IV push (17)" else IV_CONTINUOUS_INSTRUCTION
    retrieved = retrieve_topk_examples(
        prompt,
        IV_EXEMPLAR_BANK[sheet_name],
        k=TOP_K_EXAMPLES,
    )
    extra_lines = [
        "Return one JSON object only.",
        "Do not wrap the JSON in markdown.",
        f"Use exactly these keys in this order: {', '.join(expected_keys)}.",
        "Do not rename keys, add keys, or explain the answer.",
    ]
    if sheet_name == "IV push (17)":
        extra_lines.extend(
            [
                'If the source says vial or vial solution, keep the formulation as "vial solution".',
                "Use the examples below as structural guidance only.",
            ]
        )
    else:
        extra_lines.extend(
            [
                "For continuous infusions, distinguish titratable and non-titratable forms from the sentence itself.",
                "Use the examples below as structural guidance only.",
            ]
        )

    examples = render_examples(retrieved)
    return f"{base}\n\n" + "\n".join(extra_lines) + f"\n\n{examples}"


def call_gemma(sheet_name, prompt, expected_keys):
    full_prompt = (
        f"{build_instruction(sheet_name, prompt, expected_keys)}\n\n"
        f"Now process this medication order:\n{prompt}"
    )
    text = call_local_model(SYSTEM_PROMPT, full_prompt)
    parsed = parse_json_response(text)
    return coerce_output_object(parsed, expected_keys), text


def main():
    xlsx_path = resolve_project_file("MedMatch Dataset for Experiment_ Final.xlsx")
    selected_sheets = selected_sheets_from_env(DEFAULT_SHEETS)
    dataset = load_dataset(xlsx_path, selected_sheets, MAX_ENTRIES_PER_SHEET)

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Model: {MODEL_NAME} | Temperature: {TEMPERATURE} | Runs: {NUM_RUNS}")
    print(f"Sheets: {', '.join(dataset.keys())}")
    print(f"RAG top-k: {TOP_K_EXAMPLES}")

    for sheet_name, entries in dataset.items():
        expected_keys = EXPECTED_KEYS_BY_SHEET[sheet_name]
        print(f"\n{'=' * 60}")
        print(f"{sheet_name} ({len(entries)} rows x {NUM_RUNS} runs)")
        print(f"{'=' * 60}")
        all_rows = []
        total_ok = 0
        total_n = 0
        total_fields_ok = 0
        total_fields_n = 0

        for run in range(1, NUM_RUNS + 1):
            run_ok = 0
            print(f"\n  --- Run {run}/{NUM_RUNS} ---")
            for idx, entry in enumerate(entries, 1):
                print(f"  [{idx}/{len(entries)}] {entry['medication']}...", end=" ", flush=True)
                llm_output, raw_response = call_gemma(sheet_name, entry["prompt"], expected_keys)
                comparison = compare_results_exact(llm_output, entry["ground_truth"])
                n_match = sum(1 for v in comparison.values() if v["match"])
                entry_all_correct = all_fields_match(comparison)
                print(f"{n_match}/{len(comparison)}" + (" [ALL]" if entry_all_correct else ""))

                total_fields_ok += n_match
                total_fields_n += len(comparison)
                total_n += 1
                if entry_all_correct:
                    total_ok += 1
                    run_ok += 1

                all_rows.append(
                    {
                        "run": run,
                        "medication": entry["medication"],
                        "prompt": entry["prompt"],
                        "ground_truth": entry["ground_truth"],
                        "llm_output": llm_output,
                        "raw_response": raw_response,
                        "comparison": comparison,
                        "fields_correct": n_match,
                        "fields_total": len(comparison),
                        "all_fields_correct": entry_all_correct,
                    }
                )
                time.sleep(SLEEP_BETWEEN_CALLS)

            print(f"  Run {run}: {run_ok}/{len(entries)} = {run_ok/len(entries)*100:.1f}%")

        safe_name = sheet_name.replace(" ", "_").replace("(", "").replace(")", "")
        json_path = os.path.join(results_dir, f"{safe_name}_rag_{timestamp}.json")
        csv_path = os.path.join(results_dir, f"{safe_name}_rag_{timestamp}.csv")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, indent=2, default=str)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Run", "Medication"] + expected_keys + ["Fields Correct", "All Correct"])
            for row in all_rows:
                out = [row["run"], row["medication"]]
                for key in expected_keys:
                    comp = row["comparison"][key]
                    out.append("MATCH" if comp["match"] else f"MISS (exp: {comp['expected']}, got: {comp['actual']})")
                out += [f"{row['fields_correct']}/{row['fields_total']}", "YES" if row["all_fields_correct"] else "NO"]
                writer.writerow(out)

        total_fields_pct = (total_fields_ok / total_fields_n * 100) if total_fields_n else 0
        overall_pct = (total_ok / total_n * 100) if total_n else 0
        print(f"Saved: {json_path}")
        print(f"Saved: {csv_path}")
        print(f"Summary: overall={overall_pct:.1f}% | field={total_fields_pct:.1f}%")


if __name__ == "__main__":
    main()
