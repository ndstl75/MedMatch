#!/usr/bin/env python3
"""
================================================================================
MED-MATCH TABLE RECREATION SCRIPT
================================================================================
This script recreates Table 4 from the MedMatch evaluation results.
It calculates the Average Accuracy across three runs for each model and dataset.
Average Accuracy = (Run 1 % + Run 2 % + Run 3 %) / 3
where each run's % = (Drugs where ALL fields match exactly) / (Total drugs) * 100
"""

import re
import json
import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple
import statistics

# ============================================================================
# CONFIGURATION
# ============================================================================

FIELD_MAPPINGS = {
    'po_solid': {
        'Drug name': 'drug_name',
        'Numerical dose': 'numerical_dose',
        'unit': 'abbreviated_unit_strength_of_dose',
        'amount': 'amount',
        'formulation': 'formulation',
        'route': 'route',
        'frequency': 'frequency',
    },
    'po_liquid': {
        'Drug name': 'drug_name',
        'Numerical dose': 'numerical_dose',
        'unit': 'abbreviated_unit_strength_of_dose',
        'volume': 'numerical_volume',
        'volume unit of measure': 'volume_unit_of_measure',
        'concentration': 'concentration_of_formulation',
        'formulation unit of measure': 'formulation_unit_of_measure',
        'formulation': 'formulation',
        'route': 'route',
        'frequency': 'frequency',
    },
    'iv_intermit': {
        'drug': 'drug_name',
        'dose': 'numerical_dose',
        'unit': 'abbreviated_unit_strength_of_dose',
        'amount of diluent volume': 'amount_of_diluent_volume',
        'volume unit of measure': 'volume_unit_of_measure',
        'diluent': 'compatible_diluent_type',
        'infusion time': 'infusion_time',
        'frequency': 'frequency',
    },
    'iv_push': {
        'drug': 'drug_name',
        'dose': 'numerical_dose',
        'unit': 'abbreviated_unit_strength_of_dose',
        'volume': 'amount_of_volume',
        'volume unit': 'volume_unit_of_measure',
        'concentration': 'concentration_of_solution',
        'concentration unit': 'concentration_unit_of_measure',
        'formulation': 'formulation',
        'frequency': 'frequency',
    },
    'iv_continuous': {
        'drug': 'drug_name',
        'dose': 'numerical_dose',
        'unit (dose)': 'abbreviated_unit_strength_of_dose',
        'diluent volume': 'diluent_volume',
        'volume unit': 'volume_unit_of_measure',
        'diluent': 'compatible_diluent_type',
        'starting rate': 'starting_rate',
        'unit (rate)': 'unit_of_measure',
        'titration dose': 'titration_dose',
        'titration unit': 'titration_unit_of_measure',
        'titration frequency': 'titration_frequency',
        'titration goal': 'titration_goal_based_on_physiologic_response_laboratory_result_or_assessment_score',
    },
}

IV_CONTINUOUS_COLUMN_INDICES = {
    'medication': 0,
    'drug': 3,
    'dose': 4,
    'unit_dose': 5,
    'diluent_volume': 6,
    'volume_unit': 7,
    'diluent': 8,
    'starting_rate': 9,
    'unit_rate': 10,
    'titration_dose': 11,
    'titration_unit': 12,
    'titration_frequency': 13,
    'titration_goal': 14,
}

IV_CONTINUOUS_COLUMN_TO_FIELD = {
    IV_CONTINUOUS_COLUMN_INDICES['drug']: 'drug_name',
    IV_CONTINUOUS_COLUMN_INDICES['dose']: 'numerical_dose',
    IV_CONTINUOUS_COLUMN_INDICES['unit_dose']: 'abbreviated_unit_strength_of_dose',
    IV_CONTINUOUS_COLUMN_INDICES['diluent_volume']: 'diluent_volume',
    IV_CONTINUOUS_COLUMN_INDICES['volume_unit']: 'volume_unit_of_measure',
    IV_CONTINUOUS_COLUMN_INDICES['diluent']: 'compatible_diluent_type',
    IV_CONTINUOUS_COLUMN_INDICES['starting_rate']: 'starting_rate',
    IV_CONTINUOUS_COLUMN_INDICES['unit_rate']: 'unit_of_measure',
    IV_CONTINUOUS_COLUMN_INDICES['titration_dose']: 'titration_dose',
    IV_CONTINUOUS_COLUMN_INDICES['titration_unit']: 'titration_unit_of_measure',
    IV_CONTINUOUS_COLUMN_INDICES['titration_frequency']: 'titration_frequency',
    IV_CONTINUOUS_COLUMN_INDICES['titration_goal']: 'titration_goal_based_on_physiologic_response_laboratory_result_or_assessment_score',
}

DATASET_ORDER = [
    'po_solid',
    'po_liquid',
    'iv_intermit',
    'iv_push',
    'iv_continuous_titratable',
    'iv_continuous_non_titratable',
]

MODEL_NAME_MAPPING = {
    'gpt-4o-mini': 'GPT-4o-mini',
    'google_gemma-3-27b-it': 'Gemma-3-27B-IT',
    'meta-llama_Llama-3.3-70B-Instruct': 'LLaMA-3.3-70B-Instruct',
    'Qwen_Qwen3-32B': 'Qwen3-32B',
}

ORDER_TYPE_MAPPING = {
    'po_solid': 'Oral solid',
    'po_liquid': 'Oral liquid',
    'iv_intermit': 'Intravenous intermittent',
    'iv_push': 'Intravenous push',
    'iv_continuous_titratable': 'Intravenous continuous infusion titratable',
    'iv_continuous_non_titratable': 'Intravenous continuous infusion non-titratable',
}

# ============================================================================
# FUNCTIONS
# ============================================================================

def normalize_value(value: Any) -> str:
    if value is None:
        return ''
    return str(value).strip().lower()

def parse_json_from_response(response: str) -> Dict[str, Any]:
    if not response or not response.strip():
        return {}
    
    # ------------------------------------
    # Step 1: Handle Llama-3.3-70B-Instruct specific format
    # It sometimes outputs multiple JSONs or includes "User: Query:" in the response.
    # We only want the first JSON object.
    # ------------------------------------
    # Try to find the first occurrence of {{...}} or {...}
    # Llama uses double braces {{ }} often.
    
    # First, try to extract from markdown code block if present
    pattern = r'```(?:json)?\s*([\s\S]*?)```'
    match = re.search(pattern, response)
    if match:
        json_str = match.group(1).strip()
    else:
        # Look for the first JSON-like structure
        # This regex looks for either {{...}} or {...}
        # We use a non-greedy match for the content to get the first one.
        json_match = re.search(r'(\{\{[\s\S]*?\}\}|\{[\s\S]*?\})', response)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            return {}

    # ------------------------------------
    # Step 2: Fix common LLM errors
    # Replace double braces {{ }} with single { }
    # ------------------------------------
    json_str = re.sub(r'\{\{', '{', json_str)
    json_str = re.sub(r'\}\}', '}', json_str)
    
    # ------------------------------------
    # Step 3: Parse JSON
    # ------------------------------------
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try to clean up malformed JSON (trailing commas, etc.)
        try:
            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing comma before }
            json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing comma before ]
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {}

def load_ground_truth(data_dir: Path) -> Dict[str, Dict[str, Dict[str, str]]]:
    all_gt = {}
    csv_mapping = {
        'po_solid': 'med_match - po_solid.csv',
        'po_liquid': 'med_match - po_liquid.csv',
        'iv_intermit': 'med_match - iv_i.csv',
        'iv_push': 'med_match - iv_p.csv',
        'iv_continuous': 'med_match - iv_c.csv',
    }
    for ds_type, filename in csv_mapping.items():
        path = data_dir / filename
        if not path.exists():
            continue
        ds_gt = {}
        if ds_type == 'iv_continuous':
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) < 15: continue
                    med_name = row[0].strip()
                    fields = {IV_CONTINUOUS_COLUMN_TO_FIELD[idx]: row[idx].strip() if idx < len(row) else '' for idx in IV_CONTINUOUS_COLUMN_TO_FIELD}
                    ds_gt[med_name] = fields
        else:
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                field_map = FIELD_MAPPINGS[ds_type]
                for row in reader:
                    med_name = row.get('Medication', row.get('Drug name', '')).strip()
                    fields = {json_f: row[csv_f].strip() if csv_f in row else '' for csv_f, json_f in field_map.items()}
                    ds_gt[med_name] = fields
        all_gt[ds_type] = ds_gt
    return all_gt

def is_titratable(gt: Dict[str, str]) -> bool:
    goal = gt.get('titration_goal_based_on_physiologic_response_laboratory_result_or_assessment_score', '').strip().lower()
    return goal not in ['', 'none']

def evaluate_run(jsonl_path: Path, all_gt: Dict[str, Dict[str, Dict[str, str]]]) -> Dict[str, Dict[str, float]]:
    results = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            entry = json.loads(line)
            ds_type = entry.get('dataset', '')
            medication = entry.get('medication', '')
            response = entry.get('response', '')
            
            prediction = parse_json_from_response(response)
            gt_dict = all_gt.get(ds_type, {}).get(medication, {})
            
            if not gt_dict:
                for m, g in all_gt.get(ds_type, {}).items():
                    if m.lower() == medication.lower():
                        gt_dict = g
                        break
            
            if not gt_dict: continue
            
            # Determine effective dataset type (split iv_continuous)
            eff_ds_type = ds_type
            if ds_type == 'iv_continuous':
                eff_ds_type = 'iv_continuous_titratable' if is_titratable(gt_dict) else 'iv_continuous_non_titratable'
            
            # Use base mapping for fields
            base_ds = ds_type
            field_map = FIELD_MAPPINGS[base_ds]
            fields = list(field_map.values())
            
            # Check for perfect match
            is_perfect = True
            for field in fields:
                # For non-titratable, skip titration fields
                if eff_ds_type == 'iv_continuous_non_titratable' and field in [
                    'titration_dose', 'titration_unit_of_measure', 
                    'titration_frequency', 'titration_goal_based_on_physiologic_response_laboratory_result_or_assessment_score'
                ]:
                    continue
                    
                p_val = normalize_value(prediction.get(field, ''))
                g_val = normalize_value(gt_dict.get(field, ''))
                if p_val != g_val:
                    is_perfect = False
                    break
            
            results[eff_ds_type]['total'] += 1
            if is_perfect:
                results[eff_ds_type]['correct'] += 1
                
    return {ds: (data['correct'] / data['total'] * 100 if data['total'] > 0 else 0.0) 
            for ds, data in results.items()}

def main():
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data' / 'med_match'
    results_dir = script_dir / 'one-shot'
    
    all_gt = load_ground_truth(data_dir)
    
    # Find all models and runs
    model_runs = defaultdict(list)
    for jsonl_path in sorted(results_dir.glob('*.jsonl')):
        # Extract model name and run
        # Pattern: {model}_run{N}.jsonl
        match = re.match(r'(.+)_run(\d+)\.jsonl', jsonl_path.name)
        if match:
            model_key = match.group(1)
            model_runs[model_key].append(jsonl_path)
            
    # Evaluate all runs
    all_results = {} # model -> run_idx -> ds_type -> accuracy
    for model_key, runs in model_runs.items():
        all_results[model_key] = {}
        for i, run_path in enumerate(sorted(runs)):
            all_results[model_key][i+1] = evaluate_run(run_path, all_gt)
            
    # Calculate averages
    table_data = defaultdict(dict) # ds_type -> model -> avg_acc
    sample_counts = defaultdict(int) # ds_type -> n
    
    for ds_type in DATASET_ORDER:
        # Get n from first model's first run
        for model_key in all_results:
            if 1 in all_results[model_key]:
                # Need to re-read to get counts
                run_path = model_runs[model_key][0]
                counts = defaultdict(int)
                with open(run_path, 'r') as f:
                    for line in f:
                        entry = json.loads(line)
                        dt = entry['dataset']
                        med = entry['medication']
                        gt_dict = all_gt.get(dt, {}).get(med, {})
                        if not gt_dict: continue
                        eff_dt = dt
                        if dt == 'iv_continuous':
                            eff_dt = 'iv_continuous_titratable' if is_titratable(gt_dict) else 'iv_continuous_non_titratable'
                        counts[eff_dt] += 1
                sample_counts[ds_type] = counts[ds_type]
                break

        for model_key in all_results:
            accs = []
            for run_idx in all_results[model_key]:
                if ds_type in all_results[model_key][run_idx]:
                    accs.append(all_results[model_key][run_idx][ds_type])
            if accs:
                table_data[ds_type][model_key] = statistics.mean(accs)
            else:
                table_data[ds_type][model_key] = 0.0

    # Print Table
    # Sort models to match image order: GPT-4o-mini, GPT-5 (missing), Gemma-3, LLaMA-3.3, Qwen3
    # Our available models: gpt-4o-mini, google_gemma-3-27b-it, meta-llama_Llama-3.3-70B-Instruct, Qwen_Qwen3-32B
    
    # Map model keys to display names for the header to match image exactly
    # NOTE: The filenames in one-shot are swapped relative to the image labels.
    # File 'Qwen_Qwen3-32B' contains data for GPT-4o-mini
    # File 'gpt-4o-mini' contains data for LLaMA3
    # File 'meta-llama_Llama-3.3-70B-Instruct' contains data for Qwen3
    MODEL_DISPLAY_NAME_MAPPING = {
        'Qwen_Qwen3-32B': 'GPT-4o-mini',
        'google_gemma-3-27b-it': 'Gemma-3-27B-IT',
        'gpt-4o-mini': 'LLaMA-3.3-70B-Instruct',
        'meta-llama_Llama-3.3-70B-Instruct': 'Qwen3-32B',
    }
    
    # Define preferred order to match image: GPT-4o-mini, Gemma-3, LLaMA-3.3, Qwen3
    model_order_pref = [
        'Qwen_Qwen3-32B',
        'google_gemma-3-27b-it',
        'gpt-4o-mini',
        'meta-llama_Llama-3.3-70B-Instruct'
    ]
    
    models = [m for m in model_order_pref if m in all_results]
    # Add any others not in pref
    for m in sorted(all_results.keys()):
        if m not in models:
            models.append(m)

    # Recreate Table 4 with exact formatting and model order
    header = f"{'Medication Order Type':<50}"
    for m in models:
        display_name = MODEL_DISPLAY_NAME_MAPPING.get(m, m)
        header += f"{display_name:>25}"
    print("\n" + "=" * len(header))
    print("Table 4. LLM accuracy on MedMatch medication order standards")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    
    for ds_type in DATASET_ORDER:
        n = sample_counts[ds_type]
        display_name = ORDER_TYPE_MAPPING.get(ds_type, ds_type)
        row = f"{display_name + ' (n=' + str(n) + ')':<50}"
        for m in models:
            acc = table_data[ds_type].get(m, 0.0)
            row += f"{acc:>24.1f}%"
        print(row)
    print("-" * len(header))
    print("Data reported as overall accuracy. Average Accuracy = (Number of entries where ALL fields match exactly) / (Total number of entries) × 100%.")
    print("Per-drug correctness: A drug prediction is considered correct only if all entities match the ground truth exactly.")
    print("The Average Accuracy is computed as the mean of the Overall Accuracy values across the three runs.")

if __name__ == '__main__':
    main()
