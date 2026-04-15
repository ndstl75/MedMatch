"""Shared helpers for experiment runners."""

import importlib.util
import json
import os
import sys
from datetime import datetime

from medmatch.core.dataset import load_dataset, resolve_project_file
from medmatch.core.schema import BASELINE_SHEET_CONFIG, IV_BASELINE_SHEET_CONFIG, ORAL_BASELINE_SHEET_CONFIG, SYSTEM_PROMPT
from medmatch.core.scorer import all_fields_match, normalize_relaxed, normalize_strict, parse_json_response
from medmatch.llm.local_ollama import LocalOllamaBackend
from medmatch.llm.remote_gemma import RemoteGemmaBackend


def make_backend(name):
    if name == "remote":
        return RemoteGemmaBackend()
    if name == "local":
        return LocalOllamaBackend()
    raise ValueError(f"Unsupported backend: {name}")


def load_experiment_dataset(start_dir, sheet_config, selected_sheets=None, max_entries_per_sheet=0):
    xlsx_path = resolve_project_file("MedMatch Dataset for Experiment_ Final.xlsx", start_dir=start_dir)
    return load_dataset(
        xlsx_path,
        sheet_config,
        selected_sheets=selected_sheets,
        max_entries_per_sheet=max_entries_per_sheet,
    )


def ensure_results_dir(start_dir):
    results_dir = os.path.join(start_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def timestamp_now():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def normalize_for_backend(backend_name):
    return normalize_relaxed if backend_name == "remote" else normalize_strict


def compare_results_backend(llm_output, ground_truth, backend_name):
    normalizer = normalize_for_backend(backend_name)
    results = {}
    for key, expected in ground_truth.items():
        expected_value = normalizer(expected)
        actual_value = normalizer(llm_output.get(key, "") if llm_output else "")
        results[key] = {
            "expected": expected_value,
            "actual": actual_value,
            "match": expected_value == actual_value,
        }
    return results


def coerce_output_object(parsed, expected_keys, *, use_aliases=True):
    from medmatch.core.schema import KEY_ALIASES

    if parsed is None:
        return {key: "" for key in expected_keys}
    if isinstance(parsed, list):
        if len(parsed) == 1 and isinstance(parsed[0], dict):
            parsed = parsed[0]
        else:
            parsed = dict(zip(expected_keys, parsed))
    if not isinstance(parsed, dict):
        return {key: "" for key in expected_keys}

    normalized = {}
    for key, value in parsed.items():
        normalized_key = str(key).strip().lower()
        normalized_key = " ".join(normalized_key.split())
        if use_aliases:
            normalized_key = KEY_ALIASES.get(normalized_key, normalized_key)
        normalized[normalized_key] = value
    return {key: normalized.get(key, "") for key in expected_keys}


def generate_text(backend, system_prompt, prompt, *, temperature=None):
    return backend.generate_text(system_prompt, prompt, temperature=temperature)


def generate_json(backend, backend_name, system_prompt, prompt, expected_keys, *, use_aliases=True, temperature=None):
    if backend_name == "remote" and use_aliases:
        return backend.generate_json(system_prompt, prompt, expected_keys, temperature=temperature)
    text = backend.generate_text(system_prompt, prompt, temperature=temperature)
    parsed = parse_json_response(text)
    return coerce_output_object(parsed, expected_keys, use_aliases=use_aliases), text


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def load_legacy_local_module(start_dir, relative_path, module_name):
    path = os.path.join(start_dir, relative_path)
    module_dir = os.path.dirname(path)
    added = False
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
        added = True
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if added and module_dir in sys.path:
            sys.path.remove(module_dir)


def iv_sheet_config():
    return IV_BASELINE_SHEET_CONFIG


def oral_sheet_config():
    return ORAL_BASELINE_SHEET_CONFIG


def baseline_sheet_config():
    return BASELINE_SHEET_CONFIG

