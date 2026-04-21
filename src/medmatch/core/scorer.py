"""Scoring and JSON coercion helpers."""

import json
import re


def parse_json_response(text):
    if text is None:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3].strip()
        elif "```" in text:
            text = text[: text.rfind("```")].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return None


def normalize_key(key):
    text = str(key).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def coerce_output_object(parsed, expected_keys):
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
        normalized[normalize_key(key)] = value
    return {key: normalized.get(key, "") for key in expected_keys}


def normalize_relaxed(value):
    """Preserve the pre-refactor remote scorer normalization semantics.

    This mirrors the unit and numeric cleanup that previously lived inside the
    legacy remote IV CoT / Tier 3 / oral normalization scripts, so remote
    scoring behavior does not drift during the architecture migration.
    """
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = text.replace("–", "-").replace("—", "-").replace("≥", ">=")
    text = re.sub(r"\bmg/hour\b", "mg/hr", text)
    text = re.sub(r"\bmcg/hour\b", "mcg/hr", text)
    text = re.sub(r"\bunits/hour\b", "units/hr", text)
    text = re.sub(r"\bunit/hour\b", "units/hr", text)
    text = re.sub(r"\s*mmhg\b", " mmhg", text)
    text = re.sub(r"\s+", " ", text)
    try:
        number = float(text)
        text = str(int(number)) if number == int(number) else str(number)
    except (ValueError, OverflowError):
        pass
    return text


def normalize_strict(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def compare_results(llm_output, ground_truth, *, normalizer):
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


def all_fields_match(comparison):
    return all(value["match"] for value in comparison.values())
