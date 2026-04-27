#!/usr/bin/env python3
"""Summarize MedMatch normalization JSONL runs under the v1 scorer artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_DATASET_KEYS = ["po_solid", "po_liquid", "iv_intermit", "iv_push", "iv_continuous"]
DEFAULT_ROW_COUNTS = {
    "po_solid": 40,
    "po_liquid": 10,
    "iv_intermit": 17,
    "iv_push": 17,
    "iv_continuous": 17,
}
CATEGORY_LABELS = {
    "po_solid": "PO solid",
    "po_liquid": "PO liquid",
    "iv_intermit": "IV intermittent",
    "iv_push": "IV push",
    "iv_continuous": "IV continuous",
}
FIELD_FAMILY_LABELS = {
    "titration goal based on physiologic response, laboratory result, or assessment score": "titration goal",
    "abbreviated unit strength of dose": "dose unit",
    "concentration of formulation": "formulation concentration",
    "formulation unit of measure": "formulation unit",
    "concentration of solution": "solution concentration",
    "volume unit of measure": "volume unit",
    "numerical dose": "numerical dose",
    "diluent volume": "diluent volume",
    "drug name": "drug name",
    "frequency": "frequency",
    "formulation": "formulation",
    "starting rate": "starting rate",
    "titration frequency": "titration frequency",
}


def parse_run_spec(raw: str) -> dict[str, str]:
    parts = raw.split("|", 4)
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "--run must be formatted as label|model_name|jsonl_path|metadata_path|note"
        )
    label, model_name, jsonl_path, metadata_path, note = parts
    return {
        "label": label,
        "model_name": model_name,
        "jsonl_path": jsonl_path,
        "metadata_path": metadata_path,
        "note": note,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_metadata(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def approximate_elapsed_seconds(jsonl_path: Path, metadata_path: Path) -> float | str:
    try:
        start = getattr(jsonl_path.stat(), "st_birthtime", jsonl_path.stat().st_mtime)
        end = metadata_path.stat().st_mtime
    except OSError:
        return ""
    return round(max(0.0, end - start), 3)


def ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def category_chunks(rows: list[dict[str, Any]], metadata: dict[str, Any]):
    dataset_keys = metadata.get("dataset_keys") or DEFAULT_DATASET_KEYS
    row_counts = metadata.get("row_counts_by_dataset") or DEFAULT_ROW_COUNTS
    cursor = 0
    for dataset_key in dataset_keys:
        count = int(row_counts[dataset_key])
        yield dataset_key, rows[cursor : cursor + count]
        cursor += count


def summarize_run(run: dict[str, str]) -> dict[str, Any]:
    jsonl_path = Path(run["jsonl_path"]).expanduser()
    metadata_path = Path(run["metadata_path"]).expanduser()
    rows = read_jsonl(jsonl_path)
    metadata = read_metadata(metadata_path)

    entries_total = len(rows)
    fields_total = sum(len(row.get("comparison_normalized", {})) for row in rows)
    entries_correct = sum(bool(row.get("norm_all_correct")) for row in rows)
    fields_correct = sum(int(row.get("norm_fields_correct", 0)) for row in rows)
    raw_entries_correct = sum(bool(row.get("raw_all_correct")) for row in rows)
    raw_fields_correct = sum(int(row.get("raw_fields_correct", 0)) for row in rows)

    overall = {
        "model": run["label"],
        "status": "completed",
        "entries_correct": entries_correct,
        "entries_total": entries_total,
        "entry_accuracy": ratio(entries_correct, entries_total),
        "fields_correct": fields_correct,
        "fields_total": fields_total,
        "field_accuracy": ratio(fields_correct, fields_total),
        "raw_entries_correct": raw_entries_correct,
        "raw_entry_accuracy": ratio(raw_entries_correct, entries_total),
        "raw_fields_correct": raw_fields_correct,
        "raw_field_accuracy": ratio(raw_fields_correct, fields_total),
    }

    category_rows = []
    residual_rows = []
    field_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})

    for dataset_key, chunk in category_chunks(rows, metadata):
        label = CATEGORY_LABELS.get(dataset_key, dataset_key)
        category_entries_total = len(chunk)
        category_fields_total = sum(len(row.get("comparison_normalized", {})) for row in chunk)
        category_entries_correct = sum(bool(row.get("norm_all_correct")) for row in chunk)
        category_fields_correct = sum(int(row.get("norm_fields_correct", 0)) for row in chunk)
        category_rows.append(
            {
                "model": run["label"],
                "category": label,
                "entries_correct": category_entries_correct,
                "entries_total": category_entries_total,
                "entry_accuracy": ratio(category_entries_correct, category_entries_total),
                "fields_correct": category_fields_correct,
                "fields_total": category_fields_total,
                "field_accuracy": ratio(category_fields_correct, category_fields_total),
            }
        )
        for row in chunk:
            for field, comparison in row.get("comparison_normalized", {}).items():
                field_totals[field]["total"] += 1
                if comparison.get("match"):
                    field_totals[field]["correct"] += 1
                else:
                    residual_rows.append(
                        {
                            "model": run["label"],
                            "category": label,
                            "medication": row.get("medication", ""),
                            "field": field,
                            "expected": comparison.get("expected", ""),
                            "actual": comparison.get("actual", ""),
                        }
                    )

    field_rows = []
    for field in sorted(field_totals):
        correct = field_totals[field]["correct"]
        total = field_totals[field]["total"]
        field_rows.append(
            {
                "model": run["label"],
                "field": field,
                "correct": correct,
                "total": total,
                "accuracy": ratio(correct, total),
                "errors": total - correct,
            }
        )

    inventory = {
        "model_label": run["label"],
        "model_name": run["model_name"],
        "status": "completed",
        "note": run["note"],
        "path": str(jsonl_path),
        "metadata": str(metadata_path),
        "rows": entries_total,
        "dataset_version": metadata.get("dataset_version", ""),
        "scorer_version": metadata.get("scorer_version", ""),
        "mode": metadata.get("mode", ""),
        "elapsed_seconds": metadata.get("elapsed_seconds", "") or approximate_elapsed_seconds(jsonl_path, metadata_path),
        "started_at": metadata.get("started_at", ""),
        "completed_at": metadata.get("completed_at", ""),
    }

    return {
        "overall": overall,
        "categories": category_rows,
        "fields": field_rows,
        "residuals": residual_rows,
        "inventory": inventory,
        "metadata": metadata,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MedMatch normalization JSONL runs.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run", action="append", type=parse_run_spec, required=True)
    parser.add_argument("--title", default="MedMatch runtime comparison")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = [summarize_run(run) for run in args.run]
    overall_rows = [summary["overall"] for summary in summaries]
    category_rows = [row for summary in summaries for row in summary["categories"]]
    field_rows = [row for summary in summaries for row in summary["fields"]]
    residual_rows = [row for summary in summaries for row in summary["residuals"]]
    inventory_rows = [summary["inventory"] for summary in summaries]

    error_counter = Counter(
        (row["model"], row["category"], FIELD_FAMILY_LABELS.get(row["field"], row["field"]))
        for row in residual_rows
    )
    error_family_rows = [
        {
            "model": model,
            "category": category,
            "field_family": field_family,
            "errors": errors,
        }
        for (model, category, field_family), errors in error_counter.most_common()
    ]

    write_csv(
        output_dir / "overall_accuracy.csv",
        overall_rows,
        [
            "model",
            "status",
            "entries_correct",
            "entries_total",
            "entry_accuracy",
            "fields_correct",
            "fields_total",
            "field_accuracy",
            "raw_entries_correct",
            "raw_entry_accuracy",
            "raw_fields_correct",
            "raw_field_accuracy",
        ],
    )
    write_csv(
        output_dir / "category_accuracy.csv",
        category_rows,
        [
            "model",
            "category",
            "entries_correct",
            "entries_total",
            "entry_accuracy",
            "fields_correct",
            "fields_total",
            "field_accuracy",
        ],
    )
    write_csv(output_dir / "field_accuracy.csv", field_rows, ["model", "field", "correct", "total", "accuracy", "errors"])
    write_csv(output_dir / "residual_errors.csv", residual_rows, ["model", "category", "medication", "field", "expected", "actual"])
    write_csv(output_dir / "error_families.csv", error_family_rows, ["model", "category", "field_family", "errors"])
    write_csv(
        output_dir / "run_inventory.csv",
        inventory_rows,
        [
            "model_label",
            "model_name",
            "status",
            "note",
            "path",
            "metadata",
            "rows",
            "dataset_version",
            "scorer_version",
            "mode",
            "elapsed_seconds",
            "started_at",
            "completed_at",
        ],
    )

    summary_json = {
        "title": args.title,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_version": sorted({row["dataset_version"] for row in inventory_rows}),
        "scorer_version": sorted({row["scorer_version"] for row in inventory_rows}),
        "overall_accuracy": overall_rows,
        "category_accuracy": category_rows,
        "field_accuracy": field_rows,
        "residual_error_count": len(residual_rows),
        "notes": [
            "These tables report v1 strict exact-match results only.",
            "Do not mix these v1 rows with any future v2 abbreviation-equivalency summaries.",
        ],
    }
    with (output_dir / "model_comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_json, handle, indent=2)

    for row in overall_rows:
        print(
            f"{row['model']}: {row['entries_correct']}/{row['entries_total']} entries, "
            f"{row['fields_correct']}/{row['fields_total']} fields"
        )


if __name__ == "__main__":
    main()
