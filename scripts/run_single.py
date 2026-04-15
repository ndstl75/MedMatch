#!/usr/bin/env python3
"""Unified single-case MedMatch debug runner."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from medmatch.llm.config import SUPPORTED_BACKENDS
from medmatch.core.schema import BASELINE_SHEET_CONFIG, SYSTEM_PROMPT
from medmatch.core.scorer import compare_results
from medmatch.experiments.baseline import build_instruction
from medmatch.experiments.common import make_backend, normalize_for_backend


CATEGORY_TO_SHEET = {
    "po_solid": "PO Solid (40)",
    "po_liquid": "PO liquid (10)",
    "iv_intermittent": "IV intermittent (16)",
    "iv_push": "IV push (17)",
    "iv_continuous": "IV continuous (16)",
}

ROUTE_LABEL_TO_CATEGORY = {
    "po solid": "po_solid",
    "po liquid": "po_liquid",
    "iv intermittent": "iv_intermittent",
    "iv push": "iv_push",
    "iv continuous": "iv_continuous",
    "po solid (40)": "po_solid",
    "po liquid (10)": "po_liquid",
    "iv intermittent (16)": "iv_intermittent",
    "iv push (17)": "iv_push",
    "iv continuous (16)": "iv_continuous",
}


def normalize_category_name(raw: str) -> str:
    key = raw.strip().lower()
    key = " ".join(key.split())
    if key not in ROUTE_LABEL_TO_CATEGORY:
        raise ValueError(f"Unsupported category/route label: {raw}")
    return ROUTE_LABEL_TO_CATEGORY[key]


def parse_input_file(path: Path) -> dict:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("input file must contain at least category and prompt on the first two non-empty lines")

    category = normalize_category_name(lines[0])
    prompt = lines[1]
    ground_truth_json = None
    ground_truth_text = None

    for line in lines[2:]:
        if line.lower() in {"zero", "one", "both"}:
            # Legacy input.txt modes are ignored by the unified runner.
            continue
        try:
            ground_truth_json = json.loads(line)
        except json.JSONDecodeError:
            ground_truth_text = line

    return {
        "category": category,
        "prompt": prompt,
        "ground_truth_json": ground_truth_json,
        "ground_truth_text": ground_truth_text,
    }


def word_set(value):
    return set(str(value).strip().lower().split())


def jaccard(a, b):
    sa, sb = word_set(a), word_set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def flatten_output(expected_keys: list[str], payload: dict) -> str:
    return " ".join(str(payload.get(key, "")) for key in expected_keys).strip()


def load_ground_truth(args) -> tuple[dict | None, str | None]:
    if args.ground_truth_json:
        return json.loads(args.ground_truth_json), None
    if args.ground_truth_text:
        return None, args.ground_truth_text
    return None, None


def print_field_comparison(payload, ground_truth, *, backend_name):
    normalizer = normalize_for_backend(backend_name)
    comparison = compare_results(payload, ground_truth, normalizer=normalizer)
    matches = sum(1 for value in comparison.values() if value["match"])
    for key, value in comparison.items():
        status = "MATCH" if value["match"] else "MISS"
        print(f"- {key}: {status} (expected: {value['expected']}, got: {value['actual']})")
    print(f"\nOverall: {matches}/{len(comparison)} fields matched")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=SUPPORTED_BACKENDS, required=True)
    parser.add_argument("--category", choices=sorted(CATEGORY_TO_SHEET))
    parser.add_argument("--prompt")
    parser.add_argument("--input-file")
    parser.add_argument("--ground-truth-json")
    parser.add_argument("--ground-truth-text")
    args = parser.parse_args()

    if not args.input_file and (not args.category or not args.prompt):
        raise SystemExit("Provide either --input-file, or both --category and --prompt.")

    category = args.category
    prompt = args.prompt
    ground_truth_json, ground_truth_text = load_ground_truth(args)

    if args.input_file:
        parsed = parse_input_file(Path(args.input_file))
        category = parsed["category"]
        prompt = parsed["prompt"]
        ground_truth_json = parsed["ground_truth_json"] if parsed["ground_truth_json"] is not None else ground_truth_json
        ground_truth_text = parsed["ground_truth_text"] if parsed["ground_truth_text"] is not None else ground_truth_text

    sheet_name = CATEGORY_TO_SHEET[category]
    config = BASELINE_SHEET_CONFIG[sheet_name]
    expected_keys = list(config["ground_truth_cols"].keys())
    instruction = build_instruction(sheet_name, config["instruction"], expected_keys, args.backend)
    user_prompt = f"{instruction}\n\nNow process this medication order:\n{prompt}"

    backend = make_backend(args.backend)
    payload, raw_response = backend.generate_json(SYSTEM_PROMPT, user_prompt, expected_keys)

    print("=" * 72)
    print(f"Backend:  {args.backend}")
    print(f"Category: {category}")
    print(f"Prompt:   {prompt}")
    print("=" * 72)
    print("\n--- Raw Response ---")
    print(raw_response)
    print("\n--- Parsed JSON ---")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if ground_truth_json:
        print("\n--- Field Comparison ---")
        print_field_comparison(payload, ground_truth_json, backend_name=args.backend)
    elif ground_truth_text:
        llm_string = flatten_output(expected_keys, payload)
        print("\n--- Jaccard Similarity ---")
        print(f"Ground truth: {ground_truth_text}")
        print(f"LLM output:   {llm_string}")
        print(f"Jaccard:      {jaccard(ground_truth_text, llm_string):.3f}")


if __name__ == "__main__":
    main()
