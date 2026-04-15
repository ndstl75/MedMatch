"""Output helpers for experiment result files."""

import csv
import json
import os


def ensure_results_dir(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def write_flat_results(rows, expected_keys, json_path, csv_path):
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, default=str)

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Run", "Medication"] + expected_keys + ["Fields Correct", "All Correct"])
        for row in rows:
            csv_row = [row["run"], row["medication"]]
            for key in expected_keys:
                comparison = row["comparison"][key]
                csv_row.append(
                    "MATCH"
                    if comparison["match"]
                    else f"MISS (exp: {comparison['expected']}, got: {comparison['actual']})"
                )
            csv_row.append(f"{row['fields_correct']}/{row['fields_total']}")
            csv_row.append("YES" if row["all_fields_correct"] else "NO")
            writer.writerow(csv_row)


def build_flat_result_paths(results_dir, sheet_name, suffix, timestamp):
    safe_name = sheet_name.replace(" ", "_").replace("(", "").replace(")", "")
    return (
        os.path.join(results_dir, f"{safe_name}_{suffix}_{timestamp}.json"),
        os.path.join(results_dir, f"{safe_name}_{suffix}_{timestamp}.csv"),
    )

