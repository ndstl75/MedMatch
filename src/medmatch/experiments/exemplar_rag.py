"""Unified exemplar-RAG experiment runner."""

import csv
import json
import os
import re
import time

from medmatch.core.scorer import all_fields_match
from medmatch.experiments.common import (
    compare_results_backend,
    ensure_results_dir,
    generate_json,
    is_remote_backend,
    iv_sheet_config,
    load_experiment_dataset,
    load_legacy_local_module,
    make_backend,
    timestamp_now,
)


REMOTE_TARGETS = {
    "IV intermittent (16)": ["Cefepime", "Bumetanide", "Iron Sucrose"],
    "IV push (17)": ["Lacosamide", "Famotidine", "Dexamethasone", "Lorazepam"],
    "IV continuous (16)": ["Cisatracurium", "Propofol", "Midazolam", "Epinephrine"],
}


def exact_text(value):
    return "" if value is None else str(value).strip()


def tokenize(text):
    return re.findall(r"[a-z0-9%/]+", exact_text(text).lower())


def build_remote_exemplar_bank(dataset):
    bank = []
    for sheet_name, names in REMOTE_TARGETS.items():
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


def retrieve_remote_exemplars(bank, sheet_name, prompt, current_medication, k):
    query = set(tokenize(prompt))
    filtered = [
        item for item in bank
        if item["sheet_name"] == sheet_name and exact_text(item["medication"]).lower() != exact_text(current_medication).lower()
    ]
    ranked = sorted(
        filtered,
        key=lambda item: (len(query & set(tokenize(item["prompt"]))), exact_text(item["medication"]).lower()),
        reverse=True,
    )
    return ranked[:k]


def format_remote_exemplar(example):
    return (
        f"Example medication order:\n{example['prompt']}\n"
        f"Example MedMatch JSON:\n{json.dumps(example['ground_truth'], ensure_ascii=False)}"
    )


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
    return "Preserve full infusion time and frequency wording. Do not infer fields that are not explicitly present."


def run_exemplar_rag(*, backend_name, selected_sheets=None, max_entries_per_sheet=0, num_runs=None, start_dir=None):
    backend = make_backend(backend_name)
    sheet_config = iv_sheet_config()
    dataset = load_experiment_dataset(
        start_dir,
        sheet_config,
        selected_sheets=selected_sheets or sheet_config.keys(),
        max_entries_per_sheet=max_entries_per_sheet,
    )
    results_dir = ensure_results_dir(start_dir)
    timestamp = timestamp_now()
    run_count = int(os.environ.get("MEDMATCH_NUM_RUNS", "3") if num_runs is None else num_runs)
    top_k = int(os.environ.get("MEDMATCH_RAG_TOP_K", "2"))
    sleep_between = float(os.environ.get("MEDMATCH_SLEEP_SECONDS", "1"))
    remote_mode = is_remote_backend(backend_name)
    use_aliases = remote_mode

    if not remote_mode:
        local_common = load_legacy_local_module(
            start_dir,
            os.path.join("scripts", "legacy", "local", "iv_clean_common_local.py"),
            "medmatch_legacy_local_common",
        )

    exemplar_bank = build_remote_exemplar_bank(dataset) if remote_mode else None

    print(f"Backend: {backend_name} | Runs: {run_count}")
    print(f"RAG top-k: {top_k}")

    total_entries = 0
    total_correct = 0
    total_fields = 0
    total_fields_ok = 0

    for sheet_name, entries in dataset.items():
        config = sheet_config[sheet_name]
        expected_keys = list(config["ground_truth_cols"].keys())
        all_rows = []
        print(f"\n{'=' * 72}")
        print(f"{sheet_name} ({len(entries)} entries x {run_count} runs)")
        print(f"{'=' * 72}")

        for run_idx in range(1, run_count + 1):
            run_ok = 0
            print(f"\n  --- Run {run_idx}/{run_count} ---")
            for idx, entry in enumerate(entries, 1):
                print(f"  [{idx}/{len(entries)}] {entry['medication']}...", end=" ", flush=True)
                if remote_mode:
                    exemplars = retrieve_remote_exemplars(exemplar_bank, sheet_name, entry["prompt"], entry["medication"], top_k)
                    parts = [
                        config["instruction"],
                        "Return one JSON object only.",
                        "Do not wrap the JSON in markdown.",
                        f"Use exactly these keys in this order: {', '.join(expected_keys)}.",
                        route_hint_for(sheet_name),
                    ]
                    if exemplars:
                        parts.append("Relevant examples:")
                        parts.extend(format_remote_exemplar(item) for item in exemplars)
                    parts.append(f"Now process this medication order:\n{entry['prompt']}")
                    user_prompt = "\n\n".join(parts)
                else:
                    base_instruction = local_common.IV_PUSH_INSTRUCTION if sheet_name == "IV push (17)" else local_common.IV_CONTINUOUS_INSTRUCTION
                    retrieved = local_common.retrieve_topk_examples(
                        entry["prompt"],
                        local_common.IV_EXEMPLAR_BANK[sheet_name],
                        k=top_k,
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
                    user_prompt = (
                        f"{base_instruction}\n\n"
                        + "\n".join(extra_lines)
                        + f"\n\n{local_common.render_examples(retrieved)}\n\nNow process this medication order:\n{entry['prompt']}"
                    )

                llm_output, raw_response = generate_json(
                    backend,
                    backend_name,
                    "You are a clinical pharmacist who formats medication orders. Only output the MedMatch JSON format.",
                    user_prompt,
                    expected_keys,
                    use_aliases=use_aliases,
                )
                time.sleep(sleep_between)

                comparison = compare_results_backend(llm_output, entry["ground_truth"], backend_name)
                fields_correct = sum(1 for value in comparison.values() if value["match"])
                entry_correct = all_fields_match(comparison)
                print(f"{fields_correct}/{len(expected_keys)} fields" + (" [ALL CORRECT]" if entry_correct else ""))

                run_ok += int(entry_correct)
                total_entries += 1
                total_correct += int(entry_correct)
                total_fields += len(expected_keys)
                total_fields_ok += fields_correct
                all_rows.append(
                    {
                        "run": run_idx,
                        "medication": entry["medication"],
                        "prompt": entry["prompt"],
                        "ground_truth": entry["ground_truth"],
                        "llm_output": llm_output,
                        "raw_response": raw_response,
                        "comparison": comparison,
                        "fields_correct": fields_correct,
                        "fields_total": len(expected_keys),
                        "all_fields_correct": entry_correct,
                    }
                )
            if entries:
                print(f"\n  Run {run_idx}: {run_ok}/{len(entries)} = {run_ok / len(entries) * 100:.1f}%")

        safe = sheet_name.replace(" ", "_").replace("(", "").replace(")", "")
        suffix = "exemplar_rag" if remote_mode else "rag"
        json_path = os.path.join(results_dir, f"{safe}_{backend_name}_{suffix}_{timestamp}.json")
        csv_path = os.path.join(results_dir, f"{safe}_{backend_name}_{suffix}_{timestamp}.csv")
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(all_rows, handle, indent=2, default=str)
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Run", "Medication"] + expected_keys + ["Fields Correct", "All Correct"])
            for row in all_rows:
                out = [row["run"], row["medication"]]
                for key in expected_keys:
                    comp = row["comparison"][key]
                    out.append("MATCH" if comp["match"] else f"MISS (exp: {comp['expected']}, got: {comp['actual']})")
                out.extend([f"{row['fields_correct']}/{row['fields_total']}", "YES" if row["all_fields_correct"] else "NO"])
                writer.writerow(out)
        print(f"  Saved: {json_path}")
        print(f"  Saved: {csv_path}")

    print(f"\n{'=' * 72}")
    print(f"SUMMARY ({backend_name} exemplar-RAG)")
    print(f"  Field accuracy: {total_fields_ok}/{total_fields} ({(total_fields_ok / total_fields * 100) if total_fields else 0:.1f}%)")
    print(f"  Overall accuracy: {total_correct}/{total_entries} ({(total_correct / total_entries * 100) if total_entries else 0:.1f}%)")
