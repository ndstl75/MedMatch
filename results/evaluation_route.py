#!/usr/bin/env python3
"""
================================================================================
MED-MATCH ROUTE EVALUATION SCRIPT
================================================================================

This script evaluates language model performance on medication route identification
across different route types.

EVALUATION METRICS:
------------------
Average Accuracy: Average accuracy across 3 runs
- Formula: (Run 1 accuracy + Run 2 accuracy + Run 3 accuracy) / 3
- Where each run's accuracy = (Number of correct predictions) / (Total predictions) × 100%

ROUTE TYPES:
------------
- Oral solid (n=40)
- Oral liquid (n=10)
- Intravenous intermittent (n=16)
- Intravenous push (n=17)
- Intravenous continuous infusion titratable (n=11)
- Intravenous continuous infusion non-titratable (n=6)

MODELS:
-------
- GPT-4o-mini
- GPT-5-chat
- Gemma-3-27B-IT
- LLaMA-3.3-70B-Instruct
- Qwen3-32B

Author: Auto-generated evaluation script
================================================================================
"""

import json
import os
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


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
    'gpt-5-chat': 'GPT-5-chat',
    'azure-gpt-5-chat': 'GPT-5-chat',
    'google/gemma-3-27b-it': 'Gemma-3-27B-IT',
    'google_gemma-3-27b-it': 'Gemma-3-27B-IT',
    'meta-llama/Llama-3.3-70B-Instruct': 'LLaMA-3.3-70B-Instruct',
    'meta-llama_Llama-3.3-70B-Instruct': 'LLaMA-3.3-70B-Instruct',
    'Qwen/Qwen3-32B': 'Qwen3-32B',
    'Qwen_Qwen3-32B': 'Qwen3-32B',
}

# Route type order for table display
ROUTE_TYPE_ORDER = [
    'po_solid',
    'po_liquid',
    'iv_intermit',
    'iv_push',
    'iv_continuous_titratable',
    'iv_continuous_non_titratable',
]

# Model order for table display
MODEL_ORDER = [
    'GPT-4o-mini',
    'GPT-5-chat',
    'Gemma-3-27B-IT',
    'LLaMA-3.3-70B-Instruct',
    'Qwen3-32B',
]


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


def load_all_data(data_dir: str) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Load all JSONL files from data directory.
    
    Structure: {model_name: {run_number: [entries]}}
    
    Args:
        data_dir: Directory containing JSONL files
        
    Returns:
        Nested dictionary of model -> run -> entries
    """
    data = defaultdict(lambda: defaultdict(list))
    
    for filename in os.listdir(data_dir):
        if not filename.endswith('.jsonl'):
            continue
            
        filepath = os.path.join(data_dir, filename)
        model_name = parse_model_name(filename)
        
        # Extract run number from filename
        if '_run' in filename:
            run_str = filename.split('_run')[1].replace('.jsonl', '')
            try:
                run_num = int(run_str)
            except ValueError:
                continue
        else:
            continue
        
        entries = load_jsonl_file(filepath)
        data[model_name][run_num] = entries
    
    return data


# ============================================================================
# SECTION 4: EVALUATION
# ============================================================================

def evaluate_route_prediction(response: str, ground_truth: str) -> bool:
    """
    Evaluate if route prediction matches ground truth.
    
    Args:
        response: Model's predicted route
        ground_truth: Correct route
        
    Returns:
        True if prediction matches ground truth, False otherwise
    """
    return normalize_route(response) == normalize_route(ground_truth)


def calculate_accuracy_per_run(entries: List[Dict], route_type: str) -> float:
    """
    Calculate accuracy for a specific route type in a single run.
    
    Args:
        entries: List of entries from JSONL file
        route_type: Route type to filter (e.g., 'po_solid', 'iv_continuous_titratable')
        
    Returns:
        Accuracy as a percentage (0-100)
    """
    # Filter entries by route type
    if route_type.startswith('iv_continuous'):
        # For iv_continuous, need to split by titratable/non-titratable
        is_titratable_type = route_type == 'iv_continuous_titratable'
        filtered = [
            e for e in entries
            if e.get('dataset') == 'iv_continuous' and
            is_titratable(e.get('prompt', '')) == is_titratable_type
        ]
    else:
        filtered = [e for e in entries if e.get('dataset') == route_type]
    
    if not filtered:
        return 0.0
    
    # Calculate accuracy
    correct = 0
    total = len(filtered)
    
    for entry in filtered:
        response = entry.get('response', '')
        ground_truth = entry.get('ground_truth', '')
        if evaluate_route_prediction(response, ground_truth):
            correct += 1
    
    return (correct / total) * 100.0 if total > 0 else 0.0


def calculate_average_accuracy(
    data: Dict[str, Dict[int, List[Dict]]],
    model_name: str,
    route_type: str
) -> float:
    """
    Calculate average accuracy across 3 runs for a model and route type.
    
    Args:
        data: All loaded data
        model_name: Model name
        route_type: Route type
        
    Returns:
        Average accuracy as a percentage (0-100)
    """
    if model_name not in data:
        return 0.0
    
    accuracies = []
    for run_num in [1, 2, 3]:
        if run_num in data[model_name]:
            accuracy = calculate_accuracy_per_run(data[model_name][run_num], route_type)
            accuracies.append(accuracy)
    
    if not accuracies:
        return 0.0
    
    return sum(accuracies) / len(accuracies)


# ============================================================================
# SECTION 5: TABLE GENERATION
# ============================================================================

def get_sample_count(data: Dict[str, Dict[int, List[Dict]]], route_type: str) -> int:
    """
    Get sample count for a route type.
    
    Args:
        data: All loaded data
        route_type: Route type
        
    Returns:
        Sample count
    """
    sample_count = 0
    if route_type in ['po_solid', 'po_liquid', 'iv_intermit', 'iv_push']:
        # Count from first run of first model
        for model_name in data.keys():
            if 1 in data[model_name]:
                entries = data[model_name][1]
                filtered = [e for e in entries if e.get('dataset') == route_type]
                sample_count = len(filtered)
                break
    else:
        # For iv_continuous subtypes, count from first run
        for model_name in data.keys():
            if 1 in data[model_name]:
                entries = data[model_name][1]
                is_titratable_type = route_type == 'iv_continuous_titratable'
                filtered = [
                    e for e in entries
                    if e.get('dataset') == 'iv_continuous' and
                    is_titratable(e.get('prompt', '')) == is_titratable_type
                ]
                sample_count = len(filtered)
                break
    return sample_count


def generate_csv(data: Dict[str, Dict[int, List[Dict]]], output_path: str):
    """
    Generate CSV file with accuracy results.
    
    Args:
        data: All loaded data
        output_path: Path to save CSV file
    """
    # Build results dictionary: {route_type: {model: accuracy}}
    results = defaultdict(dict)
    
    for route_type in ROUTE_TYPE_ORDER:
        for model_file_name in data.keys():
            display_model_name = get_display_model_name(model_file_name)
            if display_model_name in MODEL_ORDER:
                accuracy = calculate_average_accuracy(data, model_file_name, route_type)
                results[route_type][display_model_name] = accuracy
    
    # Write CSV file
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header row
        header = ["Medication Route"]
        for model in MODEL_ORDER:
            header.append(model)
        writer.writerow(header)
        
        # Data rows
        for route_type in ROUTE_TYPE_ORDER:
            display_route = ROUTE_TYPE_MAPPING.get(route_type, route_type)
            sample_count = get_sample_count(data, route_type)
            
            row = [f"{display_route} (n={sample_count})"]
            for model in MODEL_ORDER:
                accuracy = results[route_type].get(model, 0.0)
                row.append(f"{accuracy:.1f}%")
            writer.writerow(row)
    
    print(f"CSV file saved to: {output_path}")


def generate_table(data: Dict[str, Dict[int, List[Dict]]]) -> str:
    """
    Generate formatted table with accuracy results.
    
    Args:
        data: All loaded data
        
    Returns:
        Formatted table string
    """
    # Build results dictionary: {route_type: {model: accuracy}}
    results = defaultdict(dict)
    
    for route_type in ROUTE_TYPE_ORDER:
        for model_file_name in data.keys():
            display_model_name = get_display_model_name(model_file_name)
            if display_model_name in MODEL_ORDER:
                accuracy = calculate_average_accuracy(data, model_file_name, route_type)
                results[route_type][display_model_name] = accuracy
    
    # Generate table
    lines = []
    lines.append("Table 5. LLM accuracy to identify medication route")
    lines.append("")
    
    # Header row
    header = ["Medication Route"]
    for model in MODEL_ORDER:
        header.append(model)
    lines.append(" | ".join(header))
    lines.append(" | ".join(["---"] * len(header)))
    
    # Data rows
    for route_type in ROUTE_TYPE_ORDER:
        display_route = ROUTE_TYPE_MAPPING.get(route_type, route_type)
        sample_count = get_sample_count(data, route_type)
        
        row = [f"{display_route} (n={sample_count})"]
        for model in MODEL_ORDER:
            accuracy = results[route_type].get(model, 0.0)
            row.append(f"{accuracy:.1f}%")
        lines.append(" | ".join(row))
    
    lines.append("")
    lines.append("Data reported as average accuracy.")
    lines.append("Average Accuracy = (Run 1 accuracy + Run 2 accuracy + Run 3 accuracy) / 3")
    lines.append("where each run's accuracy = (Number of correct predictions) / (Total predictions) × 100%")
    
    return "\n".join(lines)


# ============================================================================
# SECTION 6: MAIN
# ============================================================================

def main():
    """Main evaluation function."""
    results_root = Path(__file__).resolve().parent
    data_dir = str(results_root / "route")

    # Load all data
    print("Loading data from:", data_dir)
    data = load_all_data(data_dir)

    # Generate and print table
    table = generate_table(data)
    print("\n" + "=" * 80)
    print(table)
    print("=" * 80)

    # Save CSV file
    csv_output_path = str(results_root / "route_accuracy_table.csv")
    generate_csv(data, csv_output_path)

    # Also save text table to file
    txt_output_file = str(results_root / "route_accuracy_table.txt")
    with open(txt_output_file, 'w', encoding='utf-8') as f:
        f.write(table)
    print(f"Text table saved to: {txt_output_file}")


if __name__ == "__main__":
    main()
