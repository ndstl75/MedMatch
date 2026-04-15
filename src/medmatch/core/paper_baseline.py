"""Paper-faithful baseline prompt and CSV dataset helpers."""

from __future__ import annotations

import csv
import os

from prompt_medmatch import (
    build_iv_continuous_messages_zero_shot,
    build_iv_intermit_messages_zero_shot,
    build_iv_push_messages_zero_shot,
    build_po_liquid_messages_zero_shot,
    build_po_solid_messages_zero_shot,
)


PAPER_BASELINE_SHEET_CONFIG = {
    "PO Solid (40)": {
        "filename": "med_match - po_solid.csv",
        "builder": build_po_solid_messages_zero_shot,
        "prompt_column": "Medication prompt (sentence format)",
        "medication_column": "Medication",
        "ground_truth_columns": {
            "drug_name": "Drug name",
            "numerical_dose": "Numerical dose",
            "abbreviated_unit_strength_of_dose": "unit",
            "amount": "amount",
            "formulation": "formulation",
            "route": "route",
            "frequency": "frequency",
        },
    },
    "PO liquid (10)": {
        "filename": "med_match - po_liquid.csv",
        "builder": build_po_liquid_messages_zero_shot,
        "prompt_column": "Medication prompt (sentence format)",
        "medication_column": "Medication",
        "ground_truth_columns": {
            "drug_name": "Drug name",
            "numerical_dose": "Numerical dose",
            "abbreviated_unit_strength_of_dose": "unit",
            "numerical_volume": "volume",
            "volume_unit_of_measure": "volume unit of measure",
            "concentration_of_formulation": "concentration",
            "formulation_unit_of_measure": "formulation unit of measure",
            "formulation": "formulation",
            "route": "route",
            "frequency": "frequency",
        },
    },
    "IV intermittent (16)": {
        "filename": "med_match - iv_i.csv",
        "builder": build_iv_intermit_messages_zero_shot,
        "prompt_column": "Medication prompt (sentence format)",
        "medication_column": "Medication",
        "ground_truth_columns": {
            "drug_name": "drug",
            "numerical_dose": "dose",
            "abbreviated_unit_strength_of_dose": "unit",
            "amount_of_diluent_volume": "amount of diluent volume",
            "volume_unit_of_measure": "volume unit of measure",
            "compatible_diluent_type": "diluent",
            "infusion_time": "infusion time",
            "frequency": "frequency",
        },
    },
    "IV push (17)": {
        "filename": "med_match - iv_p.csv",
        "builder": build_iv_push_messages_zero_shot,
        "prompt_column": "Medication prompt (sentence format)",
        "medication_column": "Medication",
        "ground_truth_columns": {
            "drug_name": "drug",
            "numerical_dose": "dose",
            "abbreviated_unit_strength_of_dose": "unit",
            "amount_of_volume": "volume",
            "volume_unit_of_measure": "volume unit",
            "concentration_of_solution": "concentration",
            "concentration_unit_of_measure": "concentration unit",
            "formulation": "formulation",
            "frequency": "frequency",
        },
    },
    "IV continuous (16)": {
        "filename": "med_match - iv_c.csv",
        "builder": build_iv_continuous_messages_zero_shot,
        "medication_index": 0,
        "prompt_index": 2,
        "ground_truth_indices": {
            "drug_name": 3,
            "numerical_dose": 4,
            "abbreviated_unit_strength_of_dose": 5,
            "diluent_volume": 6,
            "volume_unit_of_measure": 7,
            "compatible_diluent_type": 8,
            "starting_rate": 9,
            "unit_of_measure": 10,
            "titration_dose": 11,
            "titration_unit_of_measure": 12,
            "titration_frequency": 13,
            "titration_goal_based_on_physiologic_response_laboratory_result_or_assessment_score": 14,
        },
    },
}


def expected_keys_for_baseline_sheet(sheet_name):
    config = PAPER_BASELINE_SHEET_CONFIG[sheet_name]
    if "ground_truth_columns" in config:
        return list(config["ground_truth_columns"].keys())
    return list(config["ground_truth_indices"].keys())


def resolve_baseline_data_dir(start_dir=None):
    base_dir = os.path.abspath(start_dir or os.getcwd())
    candidates = [
        os.path.join(base_dir, "data", "med_match"),
        os.path.join(os.path.dirname(base_dir), "data", "med_match"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


def build_zero_shot_messages(sheet_name, medication_prompt):
    return PAPER_BASELINE_SHEET_CONFIG[sheet_name]["builder"](medication_prompt)


def build_zero_shot_prompt_pair(sheet_name, medication_prompt):
    messages = build_zero_shot_messages(sheet_name, medication_prompt)
    if len(messages) != 2:
        raise ValueError(f"Expected zero-shot baseline prompt to have 2 messages, got {len(messages)}")
    return messages[0]["content"], messages[1]["content"]


def _normalize_cell(value):
    if value is None:
        return ""
    return str(value).strip()


def _load_dict_rows(path, config, max_entries_per_sheet):
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            prompt = _normalize_cell(raw.get(config["prompt_column"], ""))
            if not prompt:
                break
            ground_truth = {
                key: _normalize_cell(raw.get(column_name, ""))
                for key, column_name in config["ground_truth_columns"].items()
            }
            rows.append(
                {
                    "medication": _normalize_cell(raw.get(config["medication_column"], "")),
                    "prompt": prompt,
                    "ground_truth": ground_truth,
                }
            )
            if max_entries_per_sheet and len(rows) >= max_entries_per_sheet:
                break
    return rows


def _load_iv_continuous_rows(path, config, max_entries_per_sheet):
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for raw in reader:
            prompt = _normalize_cell(raw[config["prompt_index"]]) if len(raw) > config["prompt_index"] else ""
            if not prompt:
                break
            ground_truth = {
                key: _normalize_cell(raw[index]) if len(raw) > index else ""
                for key, index in config["ground_truth_indices"].items()
            }
            rows.append(
                {
                    "medication": _normalize_cell(raw[config["medication_index"]]) if len(raw) > config["medication_index"] else "",
                    "prompt": prompt,
                    "ground_truth": ground_truth,
                }
            )
            if max_entries_per_sheet and len(rows) >= max_entries_per_sheet:
                break
    return rows


def load_baseline_dataset(start_dir=None, selected_sheets=None, max_entries_per_sheet=0):
    data_dir = resolve_baseline_data_dir(start_dir=start_dir)
    selected = set(selected_sheets or PAPER_BASELINE_SHEET_CONFIG.keys())
    dataset = {}

    for sheet_name, config in PAPER_BASELINE_SHEET_CONFIG.items():
        if sheet_name not in selected:
            continue
        path = os.path.join(data_dir, config["filename"])
        if "ground_truth_columns" in config:
            dataset[sheet_name] = _load_dict_rows(path, config, max_entries_per_sheet)
        else:
            dataset[sheet_name] = _load_iv_continuous_rows(path, config, max_entries_per_sheet)
    return dataset
