"""Unified normalization runners."""

import csv
import json
import os
import time

from prompt_medmatch import build_remote_normalization_oral_instruction, build_remote_normalization_prompt
from medmatch.core.scorer import all_fields_match
from medmatch.experiments.common import (
    compare_results_backend,
    ensure_results_dir,
    generate_json,
    generate_text,
    is_remote_backend,
    iv_sheet_config,
    load_experiment_dataset,
    load_legacy_local_module,
    make_backend,
    oral_sheet_config,
    timestamp_now,
)


def run_tier3(*, backend_name, category, start_dir=None, selected_sheets=None, max_entries_per_sheet=0, num_runs=None):
    backend = make_backend(backend_name)
    remote_mode = is_remote_backend(backend_name)
    if category in {"po_solid", "po_liquid", "oral"}:
        family = "oral"
        if remote_mode:
            sheet_config = {
                name: {
                    "instruction": build_remote_normalization_oral_instruction(name),
                    "prompt_col": 3,
                    "ground_truth_cols": oral_sheet_config()[name]["ground_truth_cols"],
                }
                for name in oral_sheet_config()
            }
        else:
            sheet_config = oral_sheet_config()
        if category == "po_solid":
            sheet_names = ["PO Solid (40)"]
        elif category == "po_liquid":
            sheet_names = ["PO liquid (10)"]
        else:
            sheet_names = list(sheet_config.keys())
    else:
        family = "iv"
        sheet_config = {
            key: value
            for key, value in iv_sheet_config().items()
            if key in {"IV intermittent (16)", "IV push (17)"}
        }
        if category == "iv_push":
            sheet_names = ["IV push (17)"]
        elif category == "iv_intermittent":
            sheet_names = ["IV intermittent (16)"]
        else:
            sheet_names = list(sheet_config.keys())

    if not remote_mode:
        local_module = load_legacy_local_module(
            start_dir,
            os.path.join(
                "scripts",
                "legacy",
                "local",
                "oral_llm_normalize_local.py" if family == "oral" else "iv_llm_normalize_local.py",
            ),
            f"medmatch_local_{family}_tier3",
        )
        local_sheet_config = local_module.SHEET_CONFIG
        sheet_config = {name: local_sheet_config[name] for name in sheet_names}
        normalize_prompt = local_module.NORMALIZE_PROMPT if family == "oral" else local_module.IV_NORMALIZE_PROMPT
    else:
        normalize_prompt = None
        sheet_config = {name: sheet_config[name] for name in sheet_names}

    dataset = load_experiment_dataset(
        start_dir,
        sheet_config,
        selected_sheets=selected_sheets or sheet_names,
        max_entries_per_sheet=max_entries_per_sheet,
    )
    results_dir = ensure_results_dir(start_dir)
    timestamp = timestamp_now()
    run_count = int(os.environ.get("MEDMATCH_NUM_RUNS", "3") if num_runs is None else num_runs)
    sleep_between = float(os.environ.get("MEDMATCH_SLEEP_SECONDS", "1" if remote_mode else "0.5"))
    use_aliases = remote_mode

    print(f"Backend: {backend_name} | Family: {family} | Runs: {run_count}")
    print("Pipeline: extract -> normalization pass -> score")

    for sheet_name, entries in dataset.items():
        cfg = sheet_config[sheet_name]
        expected_keys = list(cfg["ground_truth_cols"].keys())
        all_rows = []
        print(f"\n{'=' * 72}")
        print(f"{sheet_name} ({len(entries)} meds x {run_count} runs, 2 calls each)")
        print(f"{'=' * 72}")

        for run_idx in range(1, run_count + 1):
            run_ok = 0
            print(f"\n  --- Run {run_idx}/{run_count} ---")
            for idx, entry in enumerate(entries, 1):
                print(f"  [{idx}/{len(entries)}] {entry['medication']}...", end=" ", flush=True)
                extract_prompt = (
                    f"{cfg['instruction']}\n\n"
                    "Return one JSON object only.\n"
                    "Do not wrap the JSON in markdown.\n"
                    f"Use exactly these keys in this order: {', '.join(expected_keys)}.\n\n"
                    f"Now process this medication order:\n{entry['prompt']}"
                )
                raw_obj, raw_text = generate_json(
                    backend,
                    backend_name,
                    "You are a clinical pharmacist who formats medication orders. Only output the MedMatch JSON format.",
                    extract_prompt,
                    expected_keys,
                    use_aliases=use_aliases,
                )
                time.sleep(sleep_between)

                normalize_text = generate_text(
                    backend,
                    "You are a clinical pharmacist who formats medication orders. Only output the MedMatch JSON format.",
                    (
                        normalize_prompt.format(sentence=entry["prompt"], raw_json=json.dumps(raw_obj, indent=2, default=str))
                        if normalize_prompt is not None
                        else build_remote_normalization_prompt(
                            entry["prompt"],
                            json.dumps(raw_obj, indent=2, default=str),
                            family=family,
                        )
                    ),
                )
                from medmatch.core.scorer import parse_json_response
                parsed = parse_json_response(normalize_text)
                if isinstance(parsed, dict):
                    norm_obj = {key: parsed.get(key, raw_obj.get(key, "")) for key in expected_keys}
                else:
                    norm_obj = dict(raw_obj)
                time.sleep(sleep_between)

                comp_raw = compare_results_backend(raw_obj, entry["ground_truth"], backend_name)
                comp_norm = compare_results_backend(norm_obj, entry["ground_truth"], backend_name)
                raw_ok = all_fields_match(comp_raw)
                norm_ok = all_fields_match(comp_norm)
                raw_fc = sum(1 for value in comp_raw.values() if value["match"])
                norm_fc = sum(1 for value in comp_norm.values() if value["match"])

                status = f"raw={raw_fc}/{len(expected_keys)} norm={norm_fc}/{len(expected_keys)}"
                if norm_ok and not raw_ok:
                    status += " [FIXED]"
                elif norm_ok:
                    status += " [ALL]"
                print(status)

                run_ok += int(norm_ok)
                all_rows.append(
                    {
                        "run": run_idx,
                        "medication": entry["medication"],
                        "prompt": entry["prompt"],
                        "ground_truth": entry["ground_truth"],
                        "raw_output": raw_obj,
                        "raw_response": raw_text,
                        "normalized_output": norm_obj,
                        "normalized_response": normalize_text,
                        "comparison_raw": comp_raw,
                        "comparison_normalized": comp_norm,
                        "raw_fields_correct": raw_fc,
                        "norm_fields_correct": norm_fc,
                        "raw_all_correct": raw_ok,
                        "norm_all_correct": norm_ok,
                    }
                )
            if entries:
                print(f"\n  Run {run_idx}: {run_ok}/{len(entries)} = {run_ok / len(entries) * 100:.1f}%")

        safe = sheet_name.replace(" ", "_").replace("(", "").replace(")", "")
        suffix = "llm_norm" if family == "oral" else "iv_llm_norm"
        json_path = os.path.join(results_dir, f"{safe}_{backend_name}_{suffix}_{timestamp}.json")
        csv_path = os.path.join(results_dir, f"{safe}_{backend_name}_{suffix}_{timestamp}.csv")
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(all_rows, handle, indent=2, default=str)
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["Run", "Medication"]
                + [f"raw_{key}" for key in expected_keys]
                + [f"norm_{key}" for key in expected_keys]
                + ["Raw Fields", "Norm Fields", "Raw All", "Norm All"]
            )
            for row in all_rows:
                out = [row["run"], row["medication"]]
                for key in expected_keys:
                    comp = row["comparison_raw"][key]
                    out.append("MATCH" if comp["match"] else f"MISS (exp: {comp['expected']}, got: {comp['actual']})")
                for key in expected_keys:
                    comp = row["comparison_normalized"][key]
                    out.append("MATCH" if comp["match"] else f"MISS (exp: {comp['expected']}, got: {comp['actual']})")
                out.extend(
                    [
                        f"{row['raw_fields_correct']}/{len(expected_keys)}",
                        f"{row['norm_fields_correct']}/{len(expected_keys)}",
                        "YES" if row["raw_all_correct"] else "NO",
                        "YES" if row["norm_all_correct"] else "NO",
                    ]
                )
                writer.writerow(out)
        print(f"  Saved: {json_path}")
        print(f"  Saved: {csv_path}")
