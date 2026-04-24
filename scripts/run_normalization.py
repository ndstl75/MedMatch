#!/usr/bin/env python3
"""Compatibility normalization CLI backed by scripts/probing_medmatch.py."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
for candidate in (ROOT, SRC):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from medmatch.core.paths import current_results_root, default_data_dir
from medmatch.llm.config import SUPPORTED_BACKENDS
from probing_medmatch import run_medmatch_pipeline, sheet_safe_name


DEFAULT_DATA_DIR = default_data_dir()
DEFAULT_RESULTS_DIR = current_results_root()


def default_model_name(backend: str) -> str:
    if backend == "local":
        return os.environ.get("OLLAMA_MODEL", "gemma4:e4b")
    if backend in {"remote", "google"}:
        return os.environ.get("GOOGLE_MODEL_NAME", "gemma-3-27b-it")
    if backend == "azure":
        return os.environ.get("AZURE_OPENAI_DEPLOYMENT") or os.environ.get("AZURE_MODEL_NAME", "gpt-4o-mini")
    if backend == "qwen_local":
        return (
            os.environ.get("LOCAL_OPENAI_MODEL_NAME")
            or os.environ.get("OPENAI_MODEL")
            or os.environ.get("OPENAI_MODEL_NAME")
            or "Qwen/Qwen3.6-35B-A3B"
        )
    return os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini")


def write_legacy_outputs(results_dir: str, backend_name: str, rows_by_sheet):
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for sheet_name, rows in rows_by_sheet.items():
        if not rows:
            continue
        expected_keys = list(rows[0]["comparison_raw"].keys())
        family = "oral" if sheet_name.startswith("PO ") else "iv"
        suffix = "llm_norm" if family == "oral" else "iv_llm_norm"
        safe = sheet_safe_name(sheet_name)
        json_path = os.path.join(results_dir, f"{safe}_{backend_name}_{suffix}_{timestamp}.json")
        csv_path = os.path.join(results_dir, f"{safe}_{backend_name}_{suffix}_{timestamp}.csv")

        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2, default=str)
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["Run", "Medication"]
                + [f"raw_{key}" for key in expected_keys]
                + [f"norm_{key}" for key in expected_keys]
                + ["Raw Fields", "Norm Fields", "Raw All", "Norm All"]
            )
            for row in rows:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=SUPPORTED_BACKENDS, required=True)
    parser.add_argument(
        "--category",
        choices=["oral", "po_solid", "po_liquid", "iv", "iv_push", "iv_intermittent", "iv_continuous"],
        required=True,
    )
    parser.add_argument("--model_name")
    args = parser.parse_args()

    category_map = {
        "oral": ["po_solid", "po_liquid"],
        "po_solid": ["po_solid"],
        "po_liquid": ["po_liquid"],
        "iv": ["iv_intermit", "iv_push"],
        "iv_push": ["iv_push"],
        "iv_intermittent": ["iv_intermit"],
        "iv_continuous": ["iv_continuous"],
    }

    rows_by_sheet = run_medmatch_pipeline(
        mode=args.backend,
        model_name=args.model_name or default_model_name(args.backend),
        prompting_type="normalization",
        num_runs=int(os.environ.get("MEDMATCH_NUM_RUNS", "3")),
        temperature=float(os.environ.get("MEDMATCH_TEMPERATURE", "0.1")),
        top_p=float(os.environ.get("MEDMATCH_TOP_P", "0.95")),
        max_new_tokens=int(os.environ.get("MEDMATCH_MAX_NEW_TOKENS", "512")),
        data_dir=DEFAULT_DATA_DIR,
        output_dir=None,
        subset_size=int(os.environ.get("MEDMATCH_MAX_ENTRIES", "0")) or None,
        dataset_keys=category_map[args.category],
    )
    write_legacy_outputs(DEFAULT_RESULTS_DIR, args.backend, rows_by_sheet)


if __name__ == "__main__":
    main()
