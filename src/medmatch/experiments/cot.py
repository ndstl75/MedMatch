"""Unified CoT experiment runner."""

import csv
import os
import time

from medmatch.core.scorer import all_fields_match
from medmatch.experiments.common import (
    compare_results_backend,
    ensure_results_dir,
    generate_json,
    generate_text,
    is_remote_backend,
    iv_sheet_config,
    load_experiment_dataset,
    make_backend,
    timestamp_now,
)


REMOTE_REASON_PROMPTS = {
    "IV intermittent (16)": """\
You are a clinical pharmacist. Analyze the following IV intermittent medication order step by step.

For each step, state what you find:
1. Drug name: identify the generic or brand name (lowercase).
2. Dose: the numeric drug dose and its unit (e.g., mg, g, units).
3. Diluent: the volume of diluent (numeric, in mL) and the compatible diluent type (e.g., 0.9% sodium chloride, D5W).
4. Infusion time: how long the infusion runs (e.g., 30 minutes, 1 hour).
5. Frequency: how often the dose is given (e.g., every 8 hours, once daily). If the order says just "daily", interpret as "once daily".

Medication order:
{prompt}

Think through each step, then summarize your findings clearly.""",
    "IV push (17)": """\
You are a clinical pharmacist. Analyze the following IV push medication order step by step.

For each step, state what you find:
1. Drug name: identify the generic or brand name (lowercase).
2. Dose: the numeric drug dose and its unit (e.g., mg, mcg, units).
3. Volume: the numeric volume administered (in mL).
4. Concentration: calculate the concentration per 1 mL. If the order says "20 mg/2 mL", the concentration is 10 mg/mL. If "50 mcg/mL", it is 50 mcg/mL. Always express as [number] [unit]/mL.
5. Formulation: the dosage form — always "vial solution" if the order mentions a vial.
6. Frequency: how often the push is given. If the order says just "daily", interpret as "once daily".

Medication order:
{prompt}

Think through each step, then summarize your findings clearly.""",
    "IV continuous (16)": """\
You are a clinical pharmacist. Analyze the following IV continuous infusion medication order step by step.

For each step, state what you find:
1. Drug name: identify the generic or brand name (lowercase).
2. Total dose in the bag: This is the TOTAL amount of drug in the prepared bag, NOT the per-mL concentration.
3. Diluent volume: the total volume of the bag in mL.
4. Compatible diluent type: the IV fluid (e.g., 0.9% sodium chloride, D5W).
5. Starting rate: the initial infusion rate and its unit.
6. Titratable or non-titratable?
7. If titratable: titration dose, titration unit, titration frequency, and titration goal.

Medication order:
{prompt}

Think through each step, then summarize your findings clearly.""",
}

LOCAL_REASON_PROMPTS = {
    "IV intermittent (16)": """\
You are a clinical pharmacist. Analyze the following IV intermittent medication order step by step.

For each step, state what you find:
1. Drug name.
2. Dose (numeric value and unit).
3. Diluent volume and compatible diluent type.
4. Infusion time.
5. Frequency.

Medication order:
{prompt}

Think through each step, then summarize your findings clearly.""",
    "IV push (17)": """\
You are a clinical pharmacist. Analyze the following IV push medication order step by step.

For each step, state what you find:
1. Drug name.
2. Dose (numeric value and unit).
3. Volume administered.
4. Concentration of solution.
5. Formulation in canonical MedMatch wording.
6. Frequency in canonical MedMatch wording.

Medication order:
{prompt}

Think through each step, then summarize your findings clearly.""",
    "IV continuous (16)": """\
You are a clinical pharmacist. Analyze the following IV continuous infusion medication order step by step.

For each step, state what you find:
1. Drug name.
2. Drug dose in the prepared infusion (numeric value and unit).
3. Diluent volume.
4. Compatible diluent type.
5. Starting rate (value and unit).
6. Is this order titratable or non-titratable, based on the order text?
7. If titratable: titration dose, titration unit, titration frequency, and titration goal.

Medication order:
{prompt}

Think through each step, then summarize your findings clearly.""",
}

REMOTE_EXTRACT_TEMPLATE = """\
Based on the clinical analysis below, extract the medication information into the MedMatch JSON format.

ANALYSIS:
{reasoning}

ORIGINAL ORDER:
{prompt}

INSTRUCTIONS:
{base_instruction}

Return one JSON object only.
Do not wrap the JSON in markdown.
Use exactly these keys in this order: {keys}.
{extra_guidance}
Now produce the MedMatch JSON:"""

LOCAL_EXTRACT_TEMPLATE = """\
Based on the clinical analysis below, extract the medication information into the MedMatch JSON format.

ANALYSIS:
{reasoning}

ORIGINAL ORDER:
{prompt}

INSTRUCTIONS:
{base_instruction}

Return one JSON object only.
Do not wrap the JSON in markdown.
Use exactly these keys in this order: {keys}.
Now produce the MedMatch JSON:"""

REMOTE_EXTRACT_GUIDANCE = {
    "IV intermittent (16)": 'If frequency is just "daily", write "once daily".',
    "IV push (17)": (
        "For concentration: always convert to per-1-mL basis. "
        'Formulation should be "vial solution" when a vial is mentioned. '
        'If frequency is just "daily", write "once daily".'
    ),
    "IV continuous (16)": (
        "Use the TOTAL drug amount in the bag, not the per-mL concentration. "
        "Normalize hour to hr in rate units. "
        "For non-titratable infusions, leave titration fields as empty strings."
    ),
}

LOCAL_EXTRACT_GUIDANCE = {
    "IV push (17)": """\
Canonicalization for IV push:
- If the source says vial or vial solution, output formulation as "vial solution".
- Normalize frequency wording to MedMatch canonical form:
  - daily -> once daily
  - BID -> twice daily""",
}

REASON_SYSTEM_PROMPTS = {
    "remote": "You are a clinical pharmacist. Think step by step and be precise with numbers and units.",
    "local": "You are a clinical pharmacist. Think step by step and be precise with numbers and units.",
}


def is_titratable(ground_truth):
    tit_fields = [
        "titration dose",
        "titration unit of measure",
        "titration frequency",
        "titration goal based on physiologic response, laboratory result, or assessment score",
    ]
    return any(ground_truth.get(field) not in (None, "", 0) for field in tit_fields)


def run_cot(*, backend_name, selected_sheets=None, max_entries_per_sheet=0, num_runs=None, start_dir=None):
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
    remote_mode = is_remote_backend(backend_name)
    sleep_between = float(os.environ.get("MEDMATCH_SLEEP_SECONDS", "1" if remote_mode else "0.5"))
    reason_prompts = REMOTE_REASON_PROMPTS if remote_mode else LOCAL_REASON_PROMPTS
    extract_template = REMOTE_EXTRACT_TEMPLATE if remote_mode else LOCAL_EXTRACT_TEMPLATE
    guidance_map = REMOTE_EXTRACT_GUIDANCE if remote_mode else LOCAL_EXTRACT_GUIDANCE
    use_aliases = remote_mode

    print(f"Backend: {backend_name} | Runs: {run_count}")
    print("Pipeline: reason -> extract -> score")

    grand = {"entries": 0, "correct": 0, "fields": 0, "fields_ok": 0}
    tit_stats = {"entries": 0, "correct": 0}
    nontit_stats = {"entries": 0, "correct": 0}

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
                reasoning = generate_text(
                    backend,
                    REASON_SYSTEM_PROMPTS[backend_name],
                    reason_prompts[sheet_name].format(prompt=entry["prompt"]),
                )
                time.sleep(sleep_between)
                extract_prompt = extract_template.format(
                    reasoning=reasoning,
                    prompt=entry["prompt"],
                    base_instruction=config["instruction"],
                    keys=", ".join(expected_keys),
                    extra_guidance=guidance_map.get(sheet_name, ""),
                )
                llm_output, raw_extract = generate_json(
                    backend,
                    backend_name,
                    "You are a clinical pharmacist who formats medication orders. Only output the MedMatch JSON format.",
                    extract_prompt,
                    expected_keys,
                    use_aliases=use_aliases,
                )
                time.sleep(sleep_between)

                comparison = compare_results_backend(llm_output, entry["ground_truth"], backend_name)
                fields_correct = sum(1 for value in comparison.values() if value["match"])
                entry_correct = all_fields_match(comparison)
                entry_type = None
                if sheet_name == "IV continuous (16)":
                    if is_titratable(entry["ground_truth"]):
                        entry_type = "titratable"
                        tit_stats["entries"] += 1
                        tit_stats["correct"] += int(entry_correct)
                    else:
                        entry_type = "non-titratable"
                        nontit_stats["entries"] += 1
                        nontit_stats["correct"] += int(entry_correct)

                print(
                    f"{fields_correct}/{len(expected_keys)}"
                    + (" [ALL]" if entry_correct else "")
                    + (f" ({entry_type})" if entry_type else "")
                )

                grand["entries"] += 1
                grand["correct"] += int(entry_correct)
                grand["fields"] += len(expected_keys)
                grand["fields_ok"] += fields_correct
                run_ok += int(entry_correct)

                all_rows.append(
                    {
                        "run": run_idx,
                        "medication": entry["medication"],
                        "prompt": entry["prompt"],
                        "ground_truth": entry["ground_truth"],
                        "reasoning": reasoning,
                        "llm_output": llm_output,
                        "raw_extract_response": raw_extract,
                        "comparison": comparison,
                        "fields_correct": fields_correct,
                        "fields_total": len(expected_keys),
                        "all_fields_correct": entry_correct,
                        "entry_type": entry_type,
                    }
                )

            if entries:
                print(f"\n  Run {run_idx}: {run_ok}/{len(entries)} = {run_ok / len(entries) * 100:.1f}%")

        safe = sheet_name.replace(" ", "_").replace("(", "").replace(")", "")
        json_path = os.path.join(results_dir, f"{safe}_{backend_name}_cot_{timestamp}.json")
        csv_path = os.path.join(results_dir, f"{safe}_{backend_name}_cot_{timestamp}.csv")
        with open(json_path, "w", encoding="utf-8") as handle:
            import json
            json.dump(all_rows, handle, indent=2, default=str)
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Run", "Medication", "Type"] + expected_keys + ["Fields Correct", "All Correct"])
            for row in all_rows:
                out = [row["run"], row["medication"], row.get("entry_type", "")]
                for key in expected_keys:
                    comp = row["comparison"][key]
                    out.append("MATCH" if comp["match"] else f"MISS (exp: {comp['expected']}, got: {comp['actual']})")
                out.extend([f"{row['fields_correct']}/{row['fields_total']}", "YES" if row["all_fields_correct"] else "NO"])
                writer.writerow(out)

        print(f"  Saved: {json_path}")
        print(f"  Saved: {csv_path}")

    print(f"\n{'=' * 72}")
    print(f"GRAND SUMMARY ({backend_name} CoT)")
    print(f"  Overall accuracy: {grand['correct']}/{grand['entries']} = {grand['correct'] / grand['entries'] * 100:.1f}%")
    print(f"  Field accuracy: {grand['fields_ok']}/{grand['fields']} = {grand['fields_ok'] / grand['fields'] * 100:.1f}%")
    if tit_stats["entries"]:
        print(f"  Titratable: {tit_stats['correct']}/{tit_stats['entries']} = {tit_stats['correct'] / tit_stats['entries'] * 100:.1f}%")
    if nontit_stats["entries"]:
        print(f"  Non-titratable: {nontit_stats['correct']}/{nontit_stats['entries']} = {nontit_stats['correct'] / nontit_stats['entries'] * 100:.1f}%")
