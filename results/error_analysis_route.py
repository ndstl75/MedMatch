#!/usr/bin/env python3
"""
================================================================================
MED-MATCH ROUTE ERROR ANALYSIS SCRIPT
================================================================================

This script performs error analysis on medication route identification predictions.
It identifies incorrect predictions and outputs them with all metadata to a CSV file.

ERROR ANALYSIS OUTPUT:
---------------------
- Identifies all cases where model prediction != ground truth (case-insensitive)
- Outputs errors with complete metadata including:
  * Dataset/route type
  * Model name
  * Run number
  * Medication name
  * Prompt text (for context analysis)
  * Ground truth route
  * Predicted route
  * Normalized versions for comparison

Author: Error analysis script
================================================================================
"""

import json
import os
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medmatch.core.paths import current_results_root


# ============================================================================
# SECTION 1: CONFIGURATION
# ============================================================================

# Route type mapping from dataset names to display names
ROUTE_TYPE_MAPPING = {
    'po_solid': 'Oral solid',
    'po_liquid': 'Oral liquid',
    'iv_intermit': 'Intravenous intermittent',
    'iv_push': 'Intravenous push',
    'iv_continuous_titratable': 'Intravenous continuous infusion titratable',
    'iv_continuous_non_titratable': 'Intravenous continuous infusion non-titratable',
}

# Model name mapping from file names to display names
MODEL_NAME_MAPPING = {
    'gpt-4o-mini': 'GPT-4o-mini',
    'google/gemma-3-27b-it': 'Gemma-3-27B-IT',
    'google_gemma-3-27b-it': 'Gemma-3-27B-IT',
    'meta-llama/Llama-3.3-70B-Instruct': 'LLaMA-3.3-70B-Instruct',
    'meta-llama_Llama-3.3-70B-Instruct': 'LLaMA-3.3-70B-Instruct',
    'Qwen/Qwen3-32B': 'Qwen3-32B',
    'Qwen_Qwen3-32B': 'Qwen3-32B',
}


# ============================================================================
# SECTION 2: HELPER FUNCTIONS
# ============================================================================

def normalize_route(route: str) -> str:
    """
    Normalize route string for comparison (case-insensitive).

    Args:
        route: Route string to normalize

    Returns:
        Normalized route string (lowercase, stripped)
    """
    if not route:
        return ""
    return route.strip().lower()


def is_titratable(prompt: str) -> bool:
    """
    Determine if an IV continuous medication is titratable based on prompt.

    A medication is titratable if the prompt contains "titrate" (case-insensitive).

    Args:
        prompt: Medication prompt string

    Returns:
        True if titratable, False if non-titratable
    """
    prompt_lower = prompt.lower()
    return "titrate" in prompt_lower


def parse_model_name(filename: str) -> str:
    """
    Extract model name from JSONL filename.

    Args:
        filename: JSONL filename (e.g., "gpt-4o-mini_run1.jsonl")

    Returns:
        Normalized model name
    """
    # Remove .jsonl extension
    base = filename.replace('.jsonl', '')
    # Remove _runN suffix
    if '_run' in base:
        base = base.rsplit('_run', 1)[0]
    return base


def get_display_model_name(model_name: str) -> str:
    """
    Get display name for model.

    Args:
        model_name: Model name from file

    Returns:
        Display name for model
    """
    return MODEL_NAME_MAPPING.get(model_name, model_name)


def get_display_route_type(dataset: str, prompt: str = "") -> str:
    """
    Get display name for route type.

    Args:
        dataset: Dataset name from entry
        prompt: Prompt text (for titratable determination)

    Returns:
        Display name for route type
    """
    if dataset == 'iv_continuous':
        if is_titratable(prompt):
            return ROUTE_TYPE_MAPPING.get('iv_continuous_titratable', dataset)
        else:
            return ROUTE_TYPE_MAPPING.get('iv_continuous_non_titratable', dataset)
    return ROUTE_TYPE_MAPPING.get(dataset, dataset)


# ============================================================================
# SECTION 3: DATA LOADING
# ============================================================================

def load_jsonl_file(filepath: str) -> List[Dict]:
    """
    Load JSONL file and return list of entries.

    Args:
        filepath: Path to JSONL file

    Returns:
        List of dictionaries from JSONL file
    """
    entries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_all_data(data_dir: str) -> List[Dict]:
    """
    Load all JSONL files from data directory and flatten into single list.

    Args:
        data_dir: Directory containing JSONL files

    Returns:
        List of all entries from all JSONL files
    """
    all_entries = []

    for filename in os.listdir(data_dir):
        if not filename.endswith('.jsonl'):
            continue

        filepath = os.path.join(data_dir, filename)
        entries = load_jsonl_file(filepath)
        all_entries.extend(entries)

    return all_entries


# ============================================================================
# SECTION 4: ERROR ANALYSIS
# ============================================================================

def is_error_entry(entry: Dict) -> bool:
    """
    Determine if an entry represents an error (incorrect prediction).

    Args:
        entry: Entry dictionary with response and ground_truth

    Returns:
        True if prediction is incorrect, False if correct
    """
    response = entry.get('response', '')
    ground_truth = entry.get('ground_truth', '')
    return normalize_route(response) != normalize_route(ground_truth)


def collect_errors(all_entries: List[Dict]) -> List[Dict]:
    """
    Collect all error entries from the dataset.

    Args:
        all_entries: List of all entries

    Returns:
        List of error entries
    """
    errors = []
    for entry in all_entries:
        if is_error_entry(entry):
            # Add normalized versions for analysis
            entry_copy = entry.copy()
            entry_copy['normalized_response'] = normalize_route(entry.get('response', ''))
            entry_copy['normalized_ground_truth'] = normalize_route(entry.get('ground_truth', ''))
            entry_copy['display_model'] = get_display_model_name(entry.get('model', ''))
            entry_copy['display_route_type'] = get_display_route_type(
                entry.get('dataset', ''),
                entry.get('prompt', '')
            )
            errors.append(entry_copy)

    return errors


def save_errors_to_csv(errors: List[Dict], output_path: str):
    """
    Save error entries to CSV file with all metadata.

    Args:
        errors: List of error entries
        output_path: Path to save CSV file
    """
    if not errors:
        print("No errors found to save.")
        return

    # Define CSV columns
    fieldnames = [
        'dataset',
        'display_route_type',
        'run',
        'model',
        'display_model',
        'medication',
        'prompt',
        'ground_truth',
        'normalized_ground_truth',
        'response',
        'normalized_response'
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for error in errors:
            writer.writerow(error)

    print(f"Error analysis CSV saved to: {output_path}")
    print(f"Total errors found: {len(errors)}")


# ============================================================================
# SECTION 5: STATISTICS AND REPORTING
# ============================================================================

def generate_error_summary(errors: List[Dict]) -> str:
    """
    Generate summary statistics of errors.

    Args:
        errors: List of error entries

    Returns:
        Summary string
    """
    if not errors:
        return "No errors found in the dataset."

    # Count errors by model
    model_errors = defaultdict(int)
    # Count errors by route type
    route_errors = defaultdict(int)
    # Count errors by model and route
    model_route_errors = defaultdict(lambda: defaultdict(int))

    for error in errors:
        model = error.get('display_model', '')
        route = error.get('display_route_type', '')

        model_errors[model] += 1
        route_errors[route] += 1
        model_route_errors[model][route] += 1

    # Generate summary
    lines = []
    lines.append("ERROR ANALYSIS SUMMARY")
    lines.append("=" * 50)
    lines.append(f"Total errors: {len(errors)}")
    lines.append("")

    lines.append("Errors by Model:")
    for model in sorted(model_errors.keys()):
        lines.append(f"  {model}: {model_errors[model]}")
    lines.append("")

    lines.append("Errors by Route Type:")
    for route in sorted(route_errors.keys()):
        lines.append(f"  {route}: {route_errors[route]}")
    lines.append("")

    lines.append("Errors by Model and Route Type:")
    for model in sorted(model_route_errors.keys()):
        lines.append(f"  {model}:")
        for route in sorted(model_route_errors[model].keys()):
            lines.append(f"    {route}: {model_route_errors[model][route]}")
        lines.append("")

    return "\n".join(lines)


# ============================================================================
# SECTION 6: MAIN
# ============================================================================

def main():
    """Main error analysis function."""
    results_root = Path(current_results_root())
    data_dir = str(results_root / "route")
    output_csv = str(results_root / "route_errors.csv")

    # Load all data
    print("Loading data from:", data_dir)
    all_entries = load_all_data(data_dir)
    print(f"Loaded {len(all_entries)} total entries")

    # Collect errors
    print("Analyzing for errors...")
    errors = collect_errors(all_entries)
    print(f"Found {len(errors)} errors")

    # Generate and print summary
    summary = generate_error_summary(errors)
    print("\n" + "=" * 80)
    print(summary)
    print("=" * 80)

    # Save errors to CSV
    save_errors_to_csv(errors, output_csv)


if __name__ == "__main__":
    main()
