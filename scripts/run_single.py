#!/usr/bin/env python3
"""Single-case MedMatch debugger."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
for candidate in (ROOT, SRC):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from medmatch.core.scorer import compare_results, normalize_strict
from medmatch.llm.config import SUPPORTED_BACKENDS, canonical_backend_name
from medmatch.llm.local_ollama import LocalOllamaBackend
from medmatch.llm.remote_api import AzureOpenAIBackend, LocalQwenOpenAIBackend, OpenAICompatibleBackend
from medmatch.llm.remote_gemma import RemoteGemmaBackend
from prompt_medmatch import (
    SYSTEM_PROMPT,
    build_iv_continuous_messages_zero_shot,
    build_iv_intermit_messages_zero_shot,
    build_iv_push_messages_zero_shot,
    build_po_liquid_messages_zero_shot,
    build_po_solid_messages_zero_shot,
)


CATEGORY_CONFIG = {
    "po_solid": {
        "sheet_name": "PO Solid (40)",
        "builder": build_po_solid_messages_zero_shot,
        "expected_keys": [
            "drug_name",
            "numerical_dose",
            "abbreviated_unit_strength_of_dose",
            "amount",
            "formulation",
            "route",
            "frequency",
        ],
    },
    "po_liquid": {
        "sheet_name": "PO liquid (10)",
        "builder": build_po_liquid_messages_zero_shot,
        "expected_keys": [
            "drug_name",
            "numerical_dose",
            "abbreviated_unit_strength_of_dose",
            "numerical_volume",
            "volume_unit_of_measure",
            "concentration_of_formulation",
            "formulation_unit_of_measure",
            "formulation",
            "route",
            "frequency",
        ],
    },
    "iv_intermittent": {
        "sheet_name": "IV intermittent (16)",
        "builder": build_iv_intermit_messages_zero_shot,
        "expected_keys": [
            "drug_name",
            "numerical_dose",
            "abbreviated_unit_strength_of_dose",
            "amount_of_diluent_volume",
            "volume_unit_of_measure",
            "compatible_diluent_type",
            "infusion_time",
            "frequency",
        ],
    },
    "iv_push": {
        "sheet_name": "IV push (17)",
        "builder": build_iv_push_messages_zero_shot,
        "expected_keys": [
            "drug_name",
            "numerical_dose",
            "abbreviated_unit_strength_of_dose",
            "amount_of_volume",
            "volume_unit_of_measure",
            "concentration_of_solution",
            "concentration_unit_of_measure",
            "formulation",
            "frequency",
        ],
    },
    "iv_continuous": {
        "sheet_name": "IV continuous (16)",
        "builder": build_iv_continuous_messages_zero_shot,
        "expected_keys": [
            "drug_name",
            "numerical_dose",
            "abbreviated_unit_strength_of_dose",
            "diluent_volume",
            "volume_unit_of_measure",
            "compatible_diluent_type",
            "starting_rate",
            "unit_of_measure",
            "titration_dose",
            "titration_unit_of_measure",
            "titration_frequency",
            "titration_goal_based_on_physiologic_response_laboratory_result_or_assessment_score",
        ],
    },
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
    key = " ".join(raw.strip().lower().split())
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


def build_zero_shot_prompt_pair(category: str, medication_prompt: str) -> tuple[str, str]:
    messages = CATEGORY_CONFIG[category]["builder"](medication_prompt)
    if len(messages) != 2:
        raise ValueError(f"Expected zero-shot prompt to have 2 messages, got {len(messages)}")
    return messages[0]["content"], messages[1]["content"]


def make_backend(name: str):
    mode = canonical_backend_name(name)
    if mode == "local":
        return LocalOllamaBackend()
    if mode == "qwen_local":
        return LocalQwenOpenAIBackend()
    if mode == "openai":
        return OpenAICompatibleBackend()
    if mode == "azure":
        return AzureOpenAIBackend()
    return RemoteGemmaBackend()


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


def print_field_comparison(payload, ground_truth):
    comparison = compare_results(payload, ground_truth, normalizer=normalize_strict)
    matches = sum(1 for value in comparison.values() if value["match"])
    for key, value in comparison.items():
        status = "MATCH" if value["match"] else "MISS"
        print(f"- {key}: {status} (expected: {value['expected']}, got: {value['actual']})")
    print(f"\nOverall: {matches}/{len(comparison)} fields matched")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=SUPPORTED_BACKENDS, required=True)
    parser.add_argument("--category", choices=sorted(CATEGORY_CONFIG))
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

    expected_keys = CATEGORY_CONFIG[category]["expected_keys"]
    system_prompt, user_prompt = build_zero_shot_prompt_pair(category, prompt)

    backend = make_backend(args.backend)
    payload, raw_response = backend.generate_json(system_prompt or SYSTEM_PROMPT, user_prompt, expected_keys)

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
        print_field_comparison(payload, ground_truth_json)
    elif ground_truth_text:
        llm_string = flatten_output(expected_keys, payload)
        print("\n--- Jaccard Similarity ---")
        print(f"Ground truth: {ground_truth_text}")
        print(f"LLM output:   {llm_string}")
        print(f"Jaccard:      {jaccard(ground_truth_text, llm_string):.3f}")


if __name__ == "__main__":
    main()
