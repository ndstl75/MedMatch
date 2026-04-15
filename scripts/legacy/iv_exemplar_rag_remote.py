#!/usr/bin/env python3
"""Strict IV exemplar-RAG experiment using the remote Google API."""

import csv
import json
import os
import re
import time
from datetime import datetime

from iv_remote_common import (
    MODEL_NAME,
    SLEEP_BETWEEN_CALLS,
    SYSTEM_PROMPT,
    IV_SHEET_CONFIG,
    all_fields_match,
    call_gemma,
    compare_results,
    exact_text,
    load_dataset,
    resolve_project_file,
)


SELECTED_SHEETS = [
    s.strip() for s in os.environ.get("MEDMATCH_SHEETS", "").split(",") if s.strip()
]
MAX_ENTRIES_PER_SHEET = int(os.environ.get("MEDMATCH_MAX_ENTRIES", "0"))
NUM_RUNS = int(os.environ.get("MEDMATCH_NUM_RUNS", "3"))  # triplicate per paper
TOP_K = int(os.environ.get("MEDMATCH_RAG_TOP_K", "2"))


def tokenize(text):
    return re.findall(r"[a-z0-9]+", exact_text(text).lower())


def score_overlap(query, candidate):
    q = set(tokenize(query))
    c = set(tokenize(candidate))
    return len(q & c)


def build_exemplar_bank(dataset):
    bank = []
    targets = {
        "IV intermittent (16)": ["Cefepime", "Bumetanide", "Iron Sucrose"],
        "IV push (17)": ["Lacosamide", "Famotidine", "Dexamethasone", "Lorazepam"],
        "IV continuous (16)": ["Cisatracurium", "Propofol", "Midazolam", "Epinephrine"],
    }
    for sheet_name, names in targets.items():
        for name in names:
            for row in dataset.get(sheet_name, []):
                if exact_text(row["medication"]).lower() == name.lower():
                    bank.append(
                        {
                            "sheet_name": sheet_name,
                            "medication": row["medication"],
                            "prompt": row["prompt"],
                            "ground_truth": row["ground_truth"],
                        }
                    )
                    break
    return bank


def format_exemplar(example):
    payload = json.dumps(example["ground_truth"], ensure_ascii=False)
    return (
        f"Example medication order:\n{example['prompt']}\n"
        f"Example MedMatch JSON:\n{payload}"
    )


def retrieve_exemplars(bank, sheet_name, prompt, current_medication):
    filtered = [
        ex for ex in bank
        if ex["sheet_name"] == sheet_name
        and exact_text(ex["medication"]).lower() != exact_text(current_medication).lower()
    ]
    ranked = sorted(
        filtered,
        key=lambda ex: (score_overlap(prompt, ex["prompt"]), exact_text(ex["medication"]).lower()),
        reverse=True,
    )
    return ranked[:TOP_K]


def build_prompt(sheet_name, base_instruction, expected_keys, prompt, exemplars, route_hint):
    parts = [
        base_instruction,
        "Return one JSON object only.",
        "Do not wrap the JSON in markdown.",
        f"Use exactly these keys in this order: {', '.join(expected_keys)}.",
        route_hint,
    ]
    if exemplars:
        parts.append("Relevant examples:")
        parts.extend(format_exemplar(ex) for ex in exemplars)
    parts.append(f"Now process this medication order:\n{prompt}")
    return "\n\n".join(parts)


def route_hint_for(sheet_name):
    if sheet_name == "IV push (17)":
        return (
            "If the source says vial or vial solution, keep formulation as vial solution. "
            "For concentrations like 20 mg/2 mL, convert to concentration of solution 10 and concentration unit of measure mg/mL. "
            "If the source says daily, output once daily."
        )
    if sheet_name == "IV continuous (16)":
        return (
            "Keep titration goals tight and do not add filler words. "
            "Normalize every minute as every 1 minute. "
            "Use the total bag amount when the sentence gives an explicit bag volume."
        )
    return (
        "Preserve full infusion time and frequency wording. "
        "Do not infer fields that are not explicitly present."
    )


def main():
    xlsx_path = resolve_project_file("MedMatch Dataset for Experiment_ Final.xlsx")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    dataset = load_dataset(xlsx_path, SELECTED_SHEETS or None, MAX_ENTRIES_PER_SHEET)
    exemplar_bank = build_exemplar_bank(dataset)

    total_entries = 0
    total_all_correct = 0
    total_fields = 0
    total_fields_correct = 0
    total_api_calls = 0

    print(f"Model: {MODEL_NAME} | Runs: {NUM_RUNS} | Strict exact-match scoring")
    print(f"Exemplar bank size: {len(exemplar_bank)}")
    print(f"Sample sheets: {', '.join(dataset.keys())}")

    for sheet_name, entries in dataset.items():
        config = IV_SHEET_CONFIG[sheet_name]
        expected_keys = list(config["ground_truth_cols"].keys())
        print(f"\n{'='*72}")
        print(f"{sheet_name} ({len(entries)} entries x {NUM_RUNS} runs)")
        print(f"{'='*72}")

        rows_out = []
        for run_idx in range(NUM_RUNS):
            run_ok = 0
            print(f"\n  --- Run {run_idx + 1}/{NUM_RUNS} ---")
            for idx, entry in enumerate(entries, 1):
                print(f"  [{idx}/{len(entries)}] {entry['medication']}...", end=" ", flush=True)
                exemplars = retrieve_exemplars(exemplar_bank, sheet_name, entry["prompt"], entry["medication"])
                user_prompt = build_prompt(
                    sheet_name,
                    config["instruction"],
                    expected_keys,
                    entry["prompt"],
                    exemplars,
                    route_hint_for(sheet_name),
                )
                llm_output, raw_response = call_gemma(
                    SYSTEM_PROMPT,
                    user_prompt,
                    expected_keys,
                )
                total_api_calls += 1

                comparison = compare_results(llm_output, entry["ground_truth"])
                n_match = sum(1 for v in comparison.values() if v["match"])
                entry_all_correct = all_fields_match(comparison)
                print(f"{n_match}/{len(expected_keys)} fields" + (" [ALL CORRECT]" if entry_all_correct else ""))

                if entry_all_correct:
                    run_ok += 1
                    total_all_correct += 1

                total_entries += 1
                total_fields += len(expected_keys)
                total_fields_correct += n_match

                rows_out.append(
                    {
                        "run": run_idx + 1,
                        "medication": entry["medication"],
                        "prompt": entry["prompt"],
                        "ground_truth": entry["ground_truth"],
                        "llm_output": llm_output,
                        "raw_response": raw_response,
                        "comparison": comparison,
                        "fields_correct": n_match,
                        "fields_total": len(expected_keys),
                        "all_fields_correct": entry_all_correct,
                    }
                )
                time.sleep(SLEEP_BETWEEN_CALLS)

            print(f"\n  Run {run_idx + 1}: {run_ok}/{len(entries)} = {run_ok/len(entries)*100:.1f}%")

        safe_name = sheet_name.replace(" ", "_").replace("(", "").replace(")", "")
        json_path = os.path.join(results_dir, f"{safe_name}_exemplar_rag_{timestamp}.json")
        csv_path = os.path.join(results_dir, f"{safe_name}_exemplar_rag_{timestamp}.csv")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rows_out, f, indent=2, default=str)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Run", "Medication"] + expected_keys + ["Fields Correct", "All Correct"])
            for row in rows_out:
                out_row = [row["run"], row["medication"]]
                for key in expected_keys:
                    c = row["comparison"][key]
                    out_row.append("MATCH" if c["match"] else f"MISS (exp: {c['expected']}, got: {c['actual']})")
                out_row.append(f"{row['fields_correct']}/{row['fields_total']}")
                out_row.append("YES" if row["all_fields_correct"] else "NO")
                writer.writerow(out_row)

        print(f"  Saved: {json_path}")
        print(f"  Saved: {csv_path}")

    field_acc = (total_fields_correct / total_fields * 100) if total_fields else 0
    overall_acc = (total_all_correct / total_entries * 100) if total_entries else 0
    print(f"\n{'='*72}")
    print(f"SUMMARY")
    print(f"  API calls: {total_api_calls}")
    print(f"  Field accuracy: {total_fields_correct}/{total_fields} ({field_acc:.1f}%)")
    print(f"  Overall accuracy: {total_all_correct}/{total_entries} ({overall_acc:.1f}%)")


if __name__ == "__main__":
    main()
