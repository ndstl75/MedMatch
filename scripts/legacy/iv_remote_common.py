#!/usr/bin/env python3
"""Compatibility exports for remote IV scripts."""

import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from medmatch.core.dataset import load_dataset as _load_dataset
from medmatch.core.dataset import resolve_project_file
from medmatch.core.schema import IV_BASELINE_SHEET_CONFIG as IV_SHEET_CONFIG
from medmatch.core.schema import IV_CONTINUOUS_INSTRUCTION
from medmatch.core.schema import IV_INTERMITTENT_INSTRUCTION
from medmatch.core.schema import IV_PUSH_INSTRUCTION
from prompt_medmatch import SYSTEM_PROMPT
from medmatch.core.scorer import all_fields_match, coerce_output_object, compare_results as _compare_results, normalize_relaxed, parse_json_response
from medmatch.llm.remote_gemma import RemoteGemmaBackend


MODEL_NAME = os.environ.get("GOOGLE_MODEL_NAME", "gemma-3-27b-it")
TEMPERATURE = float(os.environ.get("MEDMATCH_TEMPERATURE", "0.1"))
MAX_RETRIES = int(os.environ.get("MEDMATCH_MAX_RETRIES", "2"))
RETRY_DELAY = float(os.environ.get("MEDMATCH_RETRY_DELAY", "5"))
SLEEP_BETWEEN_CALLS = float(os.environ.get("MEDMATCH_SLEEP_SECONDS", "3"))

_BACKEND = RemoteGemmaBackend(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    retries=MAX_RETRIES,
    retry_delay=RETRY_DELAY,
)


def exact_text(value):
    if value is None:
        return ""
    return str(value).strip()


def load_dataset(xlsx_path, selected_sheets=None, max_entries_per_sheet=0):
    return _load_dataset(
        xlsx_path,
        IV_SHEET_CONFIG,
        selected_sheets=selected_sheets,
        max_entries_per_sheet=max_entries_per_sheet,
    )


def compare_results(llm_output, ground_truth):
    return _compare_results(llm_output, ground_truth, normalizer=normalize_relaxed)


def call_gemma(system_prompt, user_prompt, expected_keys, *, temperature=None, model=None, retries=None):
    del model, retries
    return _BACKEND.generate_json(system_prompt, user_prompt, expected_keys, temperature=temperature)
