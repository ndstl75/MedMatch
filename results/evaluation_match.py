#!/usr/bin/env python3
"""
================================================================================
MED-MATCH EVALUATION SCRIPT
================================================================================

This script evaluates language model performance on the med_match dataset for
medication information extraction tasks.

EVALUATION METRICS:
------------------
1. MICRO-F1: Overall accuracy across ALL fields and ALL samples
   - Formula: (Total Correct Fields) / (Total Fields)
   - Treats each field equally regardless of sample

2. MACRO-F1: Average of per-field accuracies
   - Formula: Average(accuracy_field1, accuracy_field2, ...)
   - Gives equal weight to each field type

3. PERFECT MATCH (NEW): Samples where ALL components are correct
   - Per-run: How many samples have ALL fields correct in each run
   - Cross-run: How many drugs have ALL fields correct in ALL 3 runs

EVALUATION PROCESS:
------------------
Step 1: Parse JSON from model responses using regex
        - Handles ```json...``` markdown code blocks
        - Handles {{...}} double brace errors
        
Step 2: Load ground truth from CSV files for each dataset type
        - po_solid, po_liquid, iv_intermit, iv_push, iv_continuous
        
Step 3: Compare predictions with ground truth
        - Exact string matching (case-insensitive)
        - Each field: 1 = correct, 0 = incorrect
        
Step 4: Calculate metrics per run and aggregate across 3 runs

ACCURACY CALCULATION METHODS:
-----------------------------
1. MICRO-F1: Overall field-level accuracy across ALL samples and fields
   - Formula: (Total Correct Fields) / (Total Fields)
   - Treats each field prediction equally

2. MACRO-F1: Average of per-field accuracies
   - Formula: Average(accuracy_field1, accuracy_field2, ...)
   - Gives equal weight to each field type

3. DRUG-LEVEL ACCURACY: % of drugs where ALL entities are correct
   - Formula: (Drugs with ALL fields correct) / (Total drugs in dataset) × 100%
   - Holistic measure: drug is only correct if EVERY field is correct
   - Table 4: Average of three runs = (Run 1 percentage + Run 2 percentage + Run 3 percentage) / 3
     where each run's percentage = (Number of entries where ALL fields match exactly) / (Total entries) × 100%

4. ENTITY-LEVEL ACCURACY: Individual field accuracy per dataset
   - Formula: (Correct predictions for field) / (Total predictions for field) × 100%
   - Shows which specific fields are most difficult

MEASURING MATCH (EXACT STRING MATCHING PER ENTITY):
---------------------------------------------------
- EACH ENTITY is compared INDEPENDENTLY using exact string matching
- Entity = individual field (e.g., drug_name, numerical_dose, unit, frequency)
- Comparison method: prediction.strip().lower() == ground_truth.strip().lower()
- Case-insensitive: "Aspirin" matches "aspirin", "MG" matches "mg"
- Whitespace-insensitive: "  2.5  " matches "2.5"
- Empty/whitespace-only predictions are considered INCORRECT (no partial credit)
- Each entity scores: 1 = EXACT MATCH, 0 = NO MATCH (binary, no fuzzy matching)

ENTITY MATCHING EXAMPLES:
-------------------------
| Entity          | Prediction    | Ground Truth  | Result |
|-----------------|---------------|---------------|--------|
| drug_name       | "aspirin"     | "Aspirin"     | 1 (✓)  |
| numerical_dose  | "100"         | "100"         | 1 (✓)  |
| unit            | "mg"          | "MG"          | 1 (✓)  |
| frequency       | "once daily"  | "twice daily" | 0 (✗)  |
| formulation     | ""            | "tablet"      | 0 (✗)  |

Author: Auto-generated evaluation script
================================================================================
"""

import re
import json
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import statistics

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medmatch.core.paths import (
    SCORER_VERSION,
    current_results_root,
    dataset_version_for_path,
    default_data_dir,
)


# ============================================================================
# SECTION 1: CONFIGURATION
# ============================================================================
# Define field mappings between CSV columns and JSON response fields
# Each dataset type has different fields to extract
# ============================================================================

FIELD_MAPPINGS = {
    # ------------------------------------
    # PO SOLID: Oral solid medications (tablets, capsules)
    # Fields: drug_name, dose, unit, amount, formulation, route, frequency
    # ------------------------------------
    'po_solid': {
        'Drug name': 'drug_name',              # e.g., "Amlodipine"
        'Numerical dose': 'numerical_dose',     # e.g., "2.5"
        'unit': 'abbreviated_unit_strength_of_dose',  # e.g., "mg"
        'amount': 'amount',                     # e.g., "1" (number of tablets)
        'formulation': 'formulation',           # e.g., "tablet"
        'route': 'route',                       # e.g., "by mouth"
        'frequency': 'frequency',               # e.g., "once daily"
    },
    
    # ------------------------------------
    # PO LIQUID: Oral liquid medications (suspensions, solutions)
    # Additional fields: volume, concentration
    # ------------------------------------
    'po_liquid': {
        'Drug name': 'drug_name',
        'Numerical dose': 'numerical_dose',
        'unit': 'abbreviated_unit_strength_of_dose',
        'volume': 'numerical_volume',           # e.g., "20" (mL)
        'volume unit of measure': 'volume_unit_of_measure',  # e.g., "mL"
        'concentration': 'concentration_of_formulation',     # e.g., "5"
        'formulation unit of measure': 'formulation_unit_of_measure',  # e.g., "mg/mL"
        'formulation': 'formulation',           # e.g., "suspension"
        'route': 'route',
        'frequency': 'frequency',
    },
    
    # ------------------------------------
    # IV INTERMITTENT: Intermittent IV infusions
    # Fields include diluent info and infusion time
    # ------------------------------------
    'iv_intermit': {
        'drug': 'drug_name',
        'dose': 'numerical_dose',
        'unit': 'abbreviated_unit_strength_of_dose',
        'amount of diluent volume': 'amount_of_diluent_volume',  # e.g., "100"
        'volume unit of measure': 'volume_unit_of_measure',      # e.g., "mL"
        'diluent': 'compatible_diluent_type',   # e.g., "D5W"
        'infusion time': 'infusion_time',       # e.g., "60 minutes"
        'frequency': 'frequency',
    },
    
    # ------------------------------------
    # IV PUSH: IV push medications
    # Fields include vial concentration info
    # ------------------------------------
    'iv_push': {
        'drug': 'drug_name',
        'dose': 'numerical_dose',
        'unit': 'abbreviated_unit_strength_of_dose',
        'volume': 'amount_of_volume',           # e.g., "2" (mL from vial)
        'volume unit': 'volume_unit_of_measure',
        'concentration': 'concentration_of_solution',  # e.g., "10"
        'concentration unit': 'concentration_unit_of_measure',  # e.g., "mg/mL"
        'formulation': 'formulation',           # e.g., "vial solution"
        'frequency': 'frequency',
    },
    
    # ------------------------------------
    # IV CONTINUOUS: Continuous IV infusions (titratable and non-titratable)
    # Most complex - includes titration parameters for titratable drugs
    # Field names must match prompt_medmatch.py JSON keys exactly
    # ------------------------------------
    # Mapping: CSV column name -> JSON field name (what LLM produces)
    # JSON fields: drug_name | numerical_dose | abbreviated_unit_strength_of_dose | diluent_volume | volume_unit_of_measure | compatible_diluent_type | starting_rate | unit_of_measure | titration_dose | titration_unit_of_measure | titration_frequency | titration_goal_based_on_...
    # CSV columns: drug | dose | unit | diluent volume | volume unit | diluent | starting rate | unit | titration dose | titration unit | titration frequency | titration goal
    'iv_continuous': {
        'drug': 'drug_name',
        'dose': 'numerical_dose',
        'unit (dose)': 'abbreviated_unit_strength_of_dose',  # First 'unit' column in CSV (dose unit)
        'diluent volume': 'diluent_volume',
        'volume unit': 'volume_unit_of_measure',
        'diluent': 'compatible_diluent_type',
        'starting rate': 'starting_rate',
        'unit (rate)': 'unit_of_measure',  # Second 'unit' column in CSV (rate unit)
        'titration dose': 'titration_dose',
        'titration unit': 'titration_unit_of_measure',
        'titration frequency': 'titration_frequency',
        'titration goal': 'titration_goal_based_on_physiologic_response_laboratory_result_or_assessment_score',
    },
}

# ------------------------------------
# Dataset type to CSV filename mapping
# ------------------------------------
DATASET_CSV_MAPPING = {
    'po_solid': 'med_match - po_solid.csv',
    'po_liquid': 'med_match - po_liquid.csv',
    'iv_intermit': 'med_match - iv_i.csv',
    'iv_push': 'med_match - iv_p.csv',
    'iv_continuous': 'med_match - iv_c.csv',
}

# ------------------------------------
# IV Continuous CSV column mapping (for manual parsing)
# ------------------------------------
# The iv_continuous CSV has duplicate 'unit' column names, so csv.DictReader
# cannot handle it properly. We use column indices instead.
# CSV structure: [Medication, Medication JSON, Medication prompt, drug, dose, unit, ...]
IV_CONTINUOUS_COLUMN_INDICES = {
    'medication': 0,
    'medication_json': 1,
    'medication_prompt': 2,
    'drug': 3,
    'dose': 4,
    'unit_dose': 5,  # First 'unit' column - dose unit (e.g., mg)
    'diluent_volume': 6,
    'volume_unit': 7,
    'diluent': 8,
    'starting_rate': 9,
    'unit_rate': 10,  # Second 'unit' column - rate unit (e.g., mcg/kg/min)
    'titration_dose': 11,
    'titration_unit': 12,
    'titration_frequency': 13,
    'titration_goal': 14,
}

# Maps CSV column index to JSON field name (for iv_continuous)
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

# Minimum required columns for iv_continuous CSV
IV_CONTINUOUS_MIN_COLUMNS = 15

# ------------------------------------
# Dataset display order for tables
# ------------------------------------
DATASET_ORDER = [
    'po_solid',
    'po_liquid',
    'iv_intermit',
    'iv_push',
    'iv_continuous_titratable',
    'iv_continuous_non_titratable',
]


# ============================================================================
# SECTION 2: JSON PARSING FUNCTIONS
# ============================================================================
# Extract structured JSON from raw model response text
# Handle various formats: markdown blocks, double braces, malformed JSON
# ============================================================================

def parse_json_from_response(response: str) -> Dict[str, Any]:
    """
    Extract JSON from model response using regex.
    
    The model may return JSON in various formats:
    1. ```json\n{...}```  - Standard markdown code block with language tag
    2. ```{...}```        - Code block without language tag
    3. {...}              - Raw JSON without code block
    4. {{...}}            - Double braces (common LLM hallucination error)
    
    Args:
        response: Raw model response string
        
    Returns:
        Parsed JSON as dictionary
        Returns empty dict {} if parsing fails
        
    Example:
        >>> response = '```json\\n{"drug_name": "aspirin"}\\n```'
        >>> parse_json_from_response(response)
        {'drug_name': 'aspirin'}
    """
    # Handle empty response
    if not response or not response.strip():
        return {}
    
    # ------------------------------------
    # Step 1: Try to extract from markdown code block
    # Pattern matches: ```json ... ``` or ``` ... ```
    # ------------------------------------
    pattern = r'```(?:json)?\s*([\s\S]*?)```'
    match = re.search(pattern, response)
    
    if match:
        json_str = match.group(1).strip()
    else:
        # ------------------------------------
        # Step 2: Try to find raw JSON object
        # Pattern matches: { ... }
        # ------------------------------------
        json_pattern = r'\{[\s\S]*\}'
        json_match = re.search(json_pattern, response)
        if json_match:
            json_str = json_match.group(0)
        else:
            return {}
    
    # ------------------------------------
    # Step 3: Fix common LLM errors
    # Replace double braces {{ }} with single { }
    # ------------------------------------
    json_str = re.sub(r'\{\{', '{', json_str)
    json_str = re.sub(r'\}\}', '}', json_str)
    
    # ------------------------------------
    # Step 4: Parse JSON
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


# ============================================================================
# SECTION 3: GROUND TRUTH LOADING FUNCTIONS
# ============================================================================
# Load ground truth data from CSV files
# Each dataset type has its own CSV with different column structure
# ============================================================================

def load_ground_truth_from_csv(csv_path: str, dataset_type: str) -> Dict[str, Dict[str, str]]:
    """
    Load ground truth from a single CSV file.

    Args:
        csv_path: Full path to CSV file
        dataset_type: Type of dataset (po_solid, po_liquid, etc.)

    Returns:
        Dictionary mapping medication name to field values:
        {
            "Amlodipine": {"drug_name": "Amlodipine", "numerical_dose": "2.5", ...},
            "Aspirin": {...},
            ...
        }
    """
    ground_truth = {}
    field_mapping = FIELD_MAPPINGS.get(dataset_type, {})

    # Special handling for iv_continuous due to duplicate 'unit' columns
    # csv.DictReader can't handle duplicate column names properly
    if dataset_type == 'iv_continuous':
        return _load_iv_continuous_ground_truth(csv_path, field_mapping)

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Use medication name as the key
            med_name = row.get('Medication', row.get('Drug name', '')).strip()
            if not med_name:
                continue

            # Extract all mapped fields from CSV row
            fields = {}

            # Handle normal mappings
            for csv_col, json_field in field_mapping.items():
                if csv_col in row:
                    value = row[csv_col].strip() if row[csv_col] else ''
                    fields[json_field] = value

            ground_truth[med_name] = fields

    return ground_truth


def _load_iv_continuous_ground_truth(csv_path: str, field_mapping: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    """
    Load ground truth for iv_continuous dataset using raw csv.reader.
    
    This special handling is needed because the CSV has duplicate 'unit' column
    names (one for dose unit, one for rate unit), which csv.DictReader cannot
    handle properly. We use column indices instead of column names.
    
    Args:
        csv_path: Path to the iv_continuous CSV file
        field_mapping: Field mapping dictionary (unused, kept for API consistency)
        
    Returns:
        Dictionary mapping medication name to field values:
        {
            "Medication Name": {"drug_name": "...", "numerical_dose": "...", ...},
            ...
        }
    """
    ground_truth = {}
    medication_col = IV_CONTINUOUS_COLUMN_INDICES['medication']
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        
        for row in reader:
            # Skip rows with insufficient columns
            if len(row) < IV_CONTINUOUS_MIN_COLUMNS:
                continue
            
            # Extract medication name
            med_name = row[medication_col].strip()
            if not med_name:
                continue
            
            # Extract all field values using column index mapping
            fields = {}
            for col_idx, json_field in IV_CONTINUOUS_COLUMN_TO_FIELD.items():
                if col_idx < len(row) and row[col_idx]:
                    fields[json_field] = row[col_idx].strip()
                else:
                    fields[json_field] = ''
            
            ground_truth[med_name] = fields
    
    return ground_truth


def load_all_ground_truths(data_dir: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Load ground truth for all dataset types.
    
    Args:
        data_dir: Directory containing all CSV files
        
    Returns:
        Nested dictionary structure:
        {
            "po_solid": {"Amlodipine": {...}, "Aspirin": {...}},
            "po_liquid": {"Doxycycline": {...}, ...},
            ...
        }
    """
    all_ground_truths = {}
    
    for dataset_type, csv_filename in DATASET_CSV_MAPPING.items():
        csv_path = os.path.join(data_dir, csv_filename)
        if os.path.exists(csv_path):
            all_ground_truths[dataset_type] = load_ground_truth_from_csv(csv_path, dataset_type)
        else:
            print(f"Warning: CSV file not found: {csv_path}")
            all_ground_truths[dataset_type] = {}
    
    return all_ground_truths


# ============================================================================
# SECTION 4: COMPARISON AND EVALUATION FUNCTIONS
# ============================================================================
# Compare model predictions with ground truth
# Use exact string matching (case-insensitive)
# ============================================================================

def normalize_value(value: Any) -> str:
    """
    Normalize a value for comparison.
    
    Normalization steps:
    1. Convert to string
    2. Strip whitespace
    3. Convert to lowercase (case-insensitive matching)
    
    Args:
        value: Any value (string, int, float, None, etc.)
        
    Returns:
        Normalized string for comparison
        
    Example:
        >>> normalize_value("Amlodipine")
        'amlodipine'
        >>> normalize_value(2.5)
        '2.5'
        >>> normalize_value(None)
        ''
    """
    if value is None:
        return ''
    return str(value).strip().lower()


def compare_field(prediction: Any, ground_truth: str) -> int:
    """
    Compare a single field: prediction vs ground truth.
    
    Uses exact string matching (case-insensitive).
    
    Args:
        prediction: Model's predicted value for this field
        ground_truth: Ground truth value from CSV
        
    Returns:
        1 if prediction matches ground truth (CORRECT)
        0 if prediction does not match (INCORRECT)
    """
    pred_normalized = normalize_value(prediction)
    gt_normalized = normalize_value(ground_truth)
    
    return 1 if pred_normalized == gt_normalized else 0


def evaluate_single_sample(
    prediction: Dict[str, Any],
    ground_truth: Dict[str, str],
    fields: List[str]
) -> Dict[str, int]:
    """
    Evaluate a single sample (one medication) across all fields.

    Args:
        prediction: Parsed JSON from model response
        ground_truth: Ground truth field values from CSV
        fields: List of field names to evaluate

    Returns:
        Dictionary mapping each field name to correctness (0 or 1)

    Example:
        >>> prediction = {"drug_name": "aspirin", "numerical_dose": "100"}
        >>> ground_truth = {"drug_name": "Aspirin", "numerical_dose": "100"}
        >>> fields = ["drug_name", "numerical_dose"]
        >>> evaluate_single_sample(prediction, ground_truth, fields)
        {'drug_name': 1, 'numerical_dose': 1}  # Both correct (case-insensitive)
    """
    results = {}
    for field in fields:
        pred_value = prediction.get(field, '')
        gt_value = ground_truth.get(field, '')
        results[field] = compare_field(pred_value, gt_value)
    return results


def is_perfect_match(sample_results: Dict[str, int]) -> bool:
    """
    Check if ALL fields in a sample are correct (perfect match).
    
    A perfect match means the model got EVERY component correct for
    this medication.
    
    Args:
        sample_results: Dictionary of {field_name: 0/1} for one sample
        
    Returns:
        True if ALL fields == 1 (all correct)
        False if ANY field == 0 (at least one incorrect)
        
    Example:
        >>> is_perfect_match({'drug_name': 1, 'dose': 1, 'unit': 1})
        True
        >>> is_perfect_match({'drug_name': 1, 'dose': 0, 'unit': 1})
        False
    """
    if not sample_results:
        return False
    return all(v == 1 for v in sample_results.values())


def is_titratable_iv_continuous(ground_truth: Dict[str, str]) -> bool:
    """
    Determine if an IV continuous medication is titratable based on ground truth.
    
    A medication is titratable if it has a non-empty titration goal field.
    Non-titratable medications have an empty or "none" titration goal.
    
    Args:
        ground_truth: Ground truth field values for the medication
        
    Returns:
        True if titratable, False if non-titratable
    """
    titration_goal = ground_truth.get('titration_goal_based_on_physiologic_response_laboratory_result_or_assessment_score', '').strip()
    # If titration goal is empty or "none" (case-insensitive), it's non-titratable
    if not titration_goal or titration_goal.lower() == 'none':
        return False
    return True


def _get_base_dataset_type(dataset_type: str) -> str:
    """
    Get the base dataset type for field mapping lookup.
    
    For split iv_continuous datasets (titratable/non-titratable), returns
    the base 'iv_continuous' type. For all other datasets, returns the
    dataset type unchanged.
    
    Args:
        dataset_type: Dataset type identifier
        
    Returns:
        Base dataset type for field mapping lookup
        
    Example:
        >>> _get_base_dataset_type('iv_continuous_titratable')
        'iv_continuous'
        >>> _get_base_dataset_type('po_solid')
        'po_solid'
    """
    if dataset_type in ['iv_continuous_titratable', 'iv_continuous_non_titratable']:
        return 'iv_continuous'
    return dataset_type


# ============================================================================
# SECTION 5: METRICS CALCULATION FUNCTIONS
# ============================================================================
# Calculate MICRO-F1, MACRO-F1, and Perfect Match statistics
# ============================================================================

def calculate_micro_f1(results: List[Dict[str, int]], fields: List[str]) -> float:
    """
    Calculate MICRO-F1 score (overall field-level accuracy).
    
    MICRO-F1 treats each field prediction equally:
    - Aggregates ALL correct predictions across ALL samples and ALL fields
    - Divides by total number of field predictions
    
    Formula: MICRO-F1 = (Total Correct Fields) / (Total Fields)
    
    For exact matching: Micro-F1 = Micro-Precision = Micro-Recall = Accuracy
    
    Args:
        results: List of per-sample results [{field: 0/1, ...}, ...]
        fields: List of field names being evaluated
        
    Returns:
        Micro-F1 score (0.0 to 1.0)
        
    Example:
        If 100 samples × 7 fields = 700 total predictions
        If 630 are correct: Micro-F1 = 630/700 = 0.90
    """
    total_correct = 0
    total_fields = 0
    
    for sample_results in results:
        for field in fields:
            if field in sample_results:
                total_correct += sample_results[field]
                total_fields += 1
    
    if total_fields == 0:
        return 0.0
    
    return total_correct / total_fields


def calculate_macro_f1(results: List[Dict[str, int]], fields: List[str]) -> float:
    """
    Calculate MACRO-F1 score (average of per-field accuracies).
    
    MACRO-F1 gives equal weight to each field type:
    1. Calculate accuracy for each field separately
    2. Average all field accuracies
    
    Formula: MACRO-F1 = Average(accuracy_field1, accuracy_field2, ...)
    
    This is useful when some fields are harder than others.
    
    Args:
        results: List of per-sample results [{field: 0/1, ...}, ...]
        fields: List of field names being evaluated
        
    Returns:
        Macro-F1 score (0.0 to 1.0)
        
    Example:
        Field accuracies: drug_name=0.95, dose=0.90, unit=0.85
        Macro-F1 = (0.95 + 0.90 + 0.85) / 3 = 0.90
    """
    per_field_scores = {}
    
    for field in fields:
        correct = sum(r.get(field, 0) for r in results)
        total = sum(1 for r in results if field in r)
        
        if total > 0:
            per_field_scores[field] = correct / total
        else:
            per_field_scores[field] = 0.0
    
    if not per_field_scores:
        return 0.0
    
    return statistics.mean(per_field_scores.values())


def calculate_per_field_accuracy(
    results: List[Dict[str, int]], 
    fields: List[str]
) -> Dict[str, float]:
    """
    Calculate accuracy for each field separately.
    
    Useful for identifying which fields are most difficult.
    
    Args:
        results: List of per-sample results
        fields: List of field names
        
    Returns:
        Dictionary: {field_name: accuracy, ...}
        
    Example:
        {
            'drug_name': 0.975,     # 97.5% correct
            'numerical_dose': 0.95, # 95% correct
            'frequency': 0.70,      # 70% correct (hardest field)
        }
    """
    per_field_accuracy = {}
    
    for field in fields:
        correct = sum(r.get(field, 0) for r in results)
        total = sum(1 for r in results if field in r)
        
        if total > 0:
            per_field_accuracy[field] = correct / total
        else:
            per_field_accuracy[field] = 0.0
    
    return per_field_accuracy


def calculate_perfect_match_stats(
    samples: List[Dict[str, Any]]
) -> Tuple[int, int, float, List[str]]:
    """
    Calculate perfect match statistics for a single run.
    
    A "perfect match" is a sample where ALL fields are correct.
    
    Args:
        samples: List of sample dictionaries with 'medication' and 'results' keys
        
    Returns:
        Tuple of:
        - perfect_count: Number of samples with all fields correct
        - total_count: Total number of samples
        - perfect_ratio: perfect_count / total_count
        - perfect_medications: List of medication names with perfect matches
        
    Example:
        >>> samples = [
        ...     {'medication': 'Aspirin', 'results': {'drug': 1, 'dose': 1}},
        ...     {'medication': 'Tylenol', 'results': {'drug': 1, 'dose': 0}},
        ... ]
        >>> calculate_perfect_match_stats(samples)
        (1, 2, 0.5, ['Aspirin'])
    """
    perfect_count = 0
    perfect_medications = []
    total_count = len(samples)
    
    for sample in samples:
        medication = sample.get('medication', 'Unknown')
        results = sample.get('results', {})
        
        if is_perfect_match(results):
            perfect_count += 1
            perfect_medications.append(medication)
    
    perfect_ratio = perfect_count / total_count if total_count > 0 else 0.0
    
    return perfect_count, total_count, perfect_ratio, perfect_medications


# ============================================================================
# SECTION 6: MAIN EVALUATION FUNCTIONS
# ============================================================================
# Core evaluation logic: process JSONL files and aggregate results
# ============================================================================

def evaluate_jsonl_file(
    jsonl_path: str,
    all_ground_truths: Dict[str, Dict[str, Dict[str, str]]]
) -> Dict[str, Any]:
    """
    Evaluate a single JSONL file (one run for one model).
    
    Process each line in the JSONL file:
    1. Parse the JSON response from model
    2. Look up ground truth for this medication
    3. Compare each field
    4. Track perfect matches
    
    Args:
        jsonl_path: Path to JSONL file with model predictions
        all_ground_truths: Ground truth data for all datasets
        
    Returns:
        Dictionary with comprehensive evaluation results:
        {
            'file': filename,
            'total_samples': N,
            'parse_failures': N,
            'overall_micro_f1': float,
            'overall_macro_f1': float,
            'perfect_count': N,           # NEW: samples with all correct
            'perfect_ratio': float,       # NEW: perfect_count / total
            'perfect_medications': [...], # NEW: list of perfect matches
            'metrics_by_dataset': {...},
            'detailed_results': {...},
        }
    """
    # ------------------------------------
    # Initialize result storage
    # ------------------------------------
    results_by_dataset = defaultdict(list)
    total_samples = 0
    parse_failures = 0
    
    # ------------------------------------
    # Process each line in JSONL file
    # ------------------------------------
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            # Parse JSONL entry
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                parse_failures += 1
                continue
            
            total_samples += 1
            
            # Extract fields from entry
            dataset_type = entry.get('dataset', '')
            medication = entry.get('medication', '')
            response = entry.get('response', '')
            
            # ------------------------------------
            # Step 1: Parse model's JSON response
            # ------------------------------------
            prediction = parse_json_from_response(response)
            
            # ------------------------------------
            # Step 2: Get ground truth for this medication
            # ------------------------------------
            dataset_gt = all_ground_truths.get(dataset_type, {})
            sample_gt = dataset_gt.get(medication, {})
            
            # Try case-insensitive lookup if exact match fails
            if not sample_gt:
                for med_name, gt in dataset_gt.items():
                    if med_name.lower() == medication.lower():
                        sample_gt = gt
                        break
            
            # ------------------------------------
            # Step 3: Evaluate this sample
            # ------------------------------------
            field_mapping = FIELD_MAPPINGS.get(dataset_type, {})
            fields = list(field_mapping.values())

            sample_results = evaluate_single_sample(prediction, sample_gt, fields)
            
            # Store results
            # For iv_continuous, also track whether it's titratable or non-titratable
            sample_data = {
                'medication': medication,
                'results': sample_results,
                'prediction': prediction,
                'ground_truth': sample_gt,
            }
            
            # Split iv_continuous into titratable and non-titratable
            if dataset_type == 'iv_continuous':
                is_titratable = is_titratable_iv_continuous(sample_gt)
                sample_data['is_titratable'] = is_titratable
                # Store in a sub-dataset key
                sub_dataset = 'iv_continuous_titratable' if is_titratable else 'iv_continuous_non_titratable'
                if sub_dataset not in results_by_dataset:
                    results_by_dataset[sub_dataset] = []
                results_by_dataset[sub_dataset].append(sample_data)
            else:
                results_by_dataset[dataset_type].append(sample_data)
    
    # ------------------------------------
    # Calculate metrics per dataset
    # ------------------------------------
    metrics_by_dataset = {}
    for dataset_type, samples in results_by_dataset.items():
        base_dataset_type = _get_base_dataset_type(dataset_type)
        field_mapping = FIELD_MAPPINGS.get(base_dataset_type, {})
        fields = list(field_mapping.values())
        
        sample_results = [s['results'] for s in samples]
        
        # Calculate F1 metrics
        micro_f1 = calculate_micro_f1(sample_results, fields)
        macro_f1 = calculate_macro_f1(sample_results, fields)
        per_field_acc = calculate_per_field_accuracy(sample_results, fields)
        
        # Calculate perfect match stats for this dataset
        perfect_count, total_count, perfect_ratio, perfect_meds = calculate_perfect_match_stats(samples)
        
        metrics_by_dataset[dataset_type] = {
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'per_field_accuracy': per_field_acc,
            'num_samples': len(samples),
            'perfect_count': perfect_count,
            'perfect_ratio': perfect_ratio,
            'perfect_medications': perfect_meds,
        }
    
    # ------------------------------------
    # Calculate overall metrics (across all datasets)
    # ------------------------------------
    all_results = []
    all_fields = set()
    all_samples = []
    
    for dataset_type, samples in results_by_dataset.items():
        # For split iv_continuous datasets, use the original field mapping
        base_dataset_type = dataset_type
        if dataset_type in ['iv_continuous_titratable', 'iv_continuous_non_titratable']:
            base_dataset_type = 'iv_continuous'
        
        field_mapping = FIELD_MAPPINGS.get(base_dataset_type, {})
        all_fields.update(field_mapping.values())
        for sample in samples:
            all_results.append(sample['results'])
            all_samples.append(sample)
    
    overall_micro_f1 = calculate_micro_f1(all_results, list(all_fields))
    overall_macro_f1 = calculate_macro_f1(all_results, list(all_fields))
    
    # Calculate overall perfect match stats
    overall_perfect_count, _, overall_perfect_ratio, overall_perfect_meds = calculate_perfect_match_stats(all_samples)
    
    return {
        'file': os.path.basename(jsonl_path),
        'total_samples': total_samples,
        'parse_failures': parse_failures,
        'overall_micro_f1': overall_micro_f1,
        'overall_macro_f1': overall_macro_f1,
        'perfect_count': overall_perfect_count,
        'perfect_ratio': overall_perfect_ratio,
        'perfect_medications': overall_perfect_meds,
        'metrics_by_dataset': metrics_by_dataset,
        'detailed_results': dict(results_by_dataset),
    }


def evaluate_all_files(results_dir: str, data_dir: str) -> Dict[str, Any]:
    """
    Evaluate all JSONL files in the results directory.
    
    Groups results by model and calculates:
    1. Per-run metrics (MICRO-F1, MACRO-F1, perfect matches)
    2. Cross-run aggregated metrics (mean ± std)
    3. Drugs that are perfect in ALL runs (consistent performance)
    
    Args:
        results_dir: Directory containing JSONL files
        data_dir: Directory containing ground truth CSV files
        
    Returns:
        Aggregated results by model:
        {
            "model_name": {
                'num_runs': 3,
                'micro_f1_mean': 0.88,
                'micro_f1_std': 0.01,
                ...
                'perfect_in_all_runs_count': N,    # Drugs perfect in ALL runs
                'perfect_in_all_runs_drugs': [...], # List of these drugs
                'runs': [run1_results, run2_results, run3_results],
            }
        }
    """
    # ------------------------------------
    # Load all ground truths
    # ------------------------------------
    print("Loading ground truth data...")
    all_ground_truths = load_all_ground_truths(data_dir)
    
    # ------------------------------------
    # Find all JSONL files
    # ------------------------------------
    jsonl_files = list(Path(results_dir).glob('*.jsonl'))
    print(f"Found {len(jsonl_files)} JSONL files to evaluate")
    
    # ------------------------------------
    # Evaluate each file and group by model
    # ------------------------------------
    model_results = defaultdict(list)
    
    for jsonl_path in sorted(jsonl_files):
        print(f"Evaluating: {jsonl_path.name}")
        
        # Extract model name and run number from filename
        # Example: "google_gemma-3-27b-it_run1.jsonl" -> model="google_gemma-3-27b-it", run="1"
        filename = jsonl_path.stem  # Remove .jsonl
        parts = filename.rsplit('_run', 1)
        model_name = parts[0] if len(parts) > 1 else filename
        run_num = parts[1] if len(parts) > 1 else '1'
        
        # Evaluate this file
        results = evaluate_jsonl_file(str(jsonl_path), all_ground_truths)
        results['model'] = model_name
        results['run'] = run_num
        
        model_results[model_name].append(results)
    
    # ------------------------------------
    # Aggregate results per model
    # ------------------------------------
    aggregated = {}
    
    for model_name, runs in model_results.items():
        # Sort runs by run number
        runs = sorted(runs, key=lambda x: x['run'])
        
        # Extract metrics from each run
        micro_f1_scores = [r['overall_micro_f1'] for r in runs]
        macro_f1_scores = [r['overall_macro_f1'] for r in runs]
        perfect_counts = [r['perfect_count'] for r in runs]
        perfect_ratios = [r['perfect_ratio'] for r in runs]
        
        # ------------------------------------
        # Calculate cross-run perfect matches
        # Find drugs that are PERFECT in ALL runs
        # ------------------------------------
        perfect_sets = [set(r['perfect_medications']) for r in runs]
        
        if perfect_sets:
            # Intersection: drugs perfect in ALL runs
            perfect_in_all_runs = set.intersection(*perfect_sets)
            # Union: drugs perfect in at least one run
            perfect_in_any_run = set.union(*perfect_sets)
        else:
            perfect_in_all_runs = set()
            perfect_in_any_run = set()
        
        aggregated[model_name] = {
            'num_runs': len(runs),
            
            # MICRO-F1 statistics
            'micro_f1_mean': statistics.mean(micro_f1_scores) if micro_f1_scores else 0,
            'micro_f1_std': statistics.stdev(micro_f1_scores) if len(micro_f1_scores) > 1 else 0,
            
            # MACRO-F1 statistics
            'macro_f1_mean': statistics.mean(macro_f1_scores) if macro_f1_scores else 0,
            'macro_f1_std': statistics.stdev(macro_f1_scores) if len(macro_f1_scores) > 1 else 0,
            
            # Perfect match statistics (per-run average)
            'perfect_count_mean': statistics.mean(perfect_counts) if perfect_counts else 0,
            'perfect_count_std': statistics.stdev(perfect_counts) if len(perfect_counts) > 1 else 0,
            'perfect_ratio_mean': statistics.mean(perfect_ratios) if perfect_ratios else 0,
            'perfect_ratio_std': statistics.stdev(perfect_ratios) if len(perfect_ratios) > 1 else 0,
            
            # Cross-run perfect matches (drugs consistent across ALL runs)
            'perfect_in_all_runs_count': len(perfect_in_all_runs),
            'perfect_in_all_runs_drugs': sorted(list(perfect_in_all_runs)),
            'perfect_in_any_run_count': len(perfect_in_any_run),
            'perfect_in_any_run_drugs': sorted(list(perfect_in_any_run)),
            
            # Individual run results
            'runs': runs,
        }
    
    return aggregated


# ============================================================================
# SECTION 7: REPORTING FUNCTIONS
# ============================================================================
# Print formatted reports and save results to JSON
# ============================================================================

def print_evaluation_report(aggregated_results: Dict[str, Any]) -> None:
    """
    Print a formatted evaluation report to console.
    
    Report includes:
    - Overall MICRO-F1 and MACRO-F1 (mean ± std across runs)
    - Perfect match statistics per run
    - Cross-run perfect match analysis
    - Per-dataset breakdown
    
    Args:
        aggregated_results: Results from evaluate_all_files()
    """
    print("\n" + "=" * 80)
    print("MED-MATCH EVALUATION REPORT")
    print("=" * 80)
    
    for model_name, model_data in aggregated_results.items():
        print(f"\n{'=' * 60}")
        print(f"MODEL: {model_name}")
        print(f"Number of runs: {model_data['num_runs']}")
        print(f"{'=' * 60}")
        
        # ------------------------------------
        # Overall F1 metrics
        # ------------------------------------
        print(f"\n--- OVERALL METRICS (averaged across {model_data['num_runs']} runs) ---")
        print(f"  MICRO-F1: {model_data['micro_f1_mean']:.4f} ± {model_data['micro_f1_std']:.4f}")
        print(f"  MACRO-F1: {model_data['macro_f1_mean']:.4f} ± {model_data['macro_f1_std']:.4f}")
        
        # ------------------------------------
        # Perfect match statistics (COMMENTED OUT - not needed)
        # ------------------------------------
        # print(f"\n--- PERFECT MATCH STATISTICS ---")
        # print(f"  Per-run average:")
        # print(f"    Perfect samples: {model_data['perfect_count_mean']:.1f} ± {model_data['perfect_count_std']:.1f}")
        # print(f"    Perfect ratio: {model_data['perfect_ratio_mean']:.4f} ± {model_data['perfect_ratio_std']:.4f}")
        #
        # print(f"\n  Cross-run consistency:")
        # print(f"    Drugs perfect in ALL {model_data['num_runs']} runs: {model_data['perfect_in_all_runs_count']}")
        # print(f"    Drugs perfect in ANY run: {model_data['perfect_in_any_run_count']}")
        #
        # if model_data['perfect_in_all_runs_drugs']:
        #     print(f"\n    Drugs perfect in ALL runs:")
        #     for drug in model_data['perfect_in_all_runs_drugs']:
        #         print(f"      - {drug}")
        
        # ------------------------------------
        # Drug-level accuracy table (NEW)
        # Shows accuracy where each drug is correct only if ALL entities are correct
        # ------------------------------------
        print(f"\n--- DRUG-LEVEL ACCURACY TABLE ---")
        print(f"Shows % of drugs where ALL entities/fields are correctly matched")
        print(f"Each drug is 'correct' only if EVERY entity matches ground truth")
        print(f"Average Acc = average accuracy across the three runs")
        print()

        # Collect all dataset names and use predefined order
        all_datasets_set = set()
        for run_data in model_data['runs']:
            all_datasets_set.update(run_data['metrics_by_dataset'].keys())
        # Use predefined order, filtering to only include datasets that exist in the data
        all_datasets = [ds for ds in DATASET_ORDER if ds in all_datasets_set]

        # Print table header
        header = "Dataset".ljust(15) + "Run 1".rjust(8) + "Run 2".rjust(8) + "Run 3".rjust(8) + "Average Acc".rjust(12)
        print(header)
        print("-" * len(header))

        # Print each dataset row
        for dataset in all_datasets:
            row_data = dataset.ljust(15)

            # Get perfect ratios for each run
            run_ratios = []
            for run_data in model_data['runs']:
                metrics = run_data['metrics_by_dataset'].get(dataset, {})
                perfect_ratio = metrics.get('perfect_ratio', 0.0)
                run_ratios.append(perfect_ratio)
                row_data += f"{perfect_ratio:.1%}".rjust(8)

            # Calculate average
            avg_ratio = statistics.mean(run_ratios) if run_ratios else 0.0
            row_data += f"{avg_ratio:.1%}".rjust(12)

            print(row_data)

        # ------------------------------------
        # Entity-level accuracy table (NEW)
        # Shows accuracy for each entity/field per dataset
        # ------------------------------------
        print(f"\n--- ENTITY-LEVEL ACCURACY TABLE ---")
        print(f"Accuracy for each individual entity/field across all drugs in the dataset")
        print()

        for dataset in all_datasets:
            print(f"Dataset: {dataset}")
            print("-" * 60)

            # Get all entities for this dataset (preserve CSV column order)
            base_dataset = _get_base_dataset_type(dataset)
            field_mapping = FIELD_MAPPINGS.get(base_dataset, {})
            entities = list(field_mapping.values())
            # Preserve order as defined in FIELD_MAPPINGS (matches CSV column order)

            # Print entity table header
            entity_header = "Entity".ljust(25) + "Run 1".rjust(8) + "Run 2".rjust(8) + "Run 3".rjust(8) + "Average".rjust(8)
            print(entity_header)
            print("-" * len(entity_header))

            # Print each entity row
            for entity in entities:
                row_data = entity.ljust(25)

                # Get accuracy for this entity across runs
                run_accuracies = []
                for run_data in model_data['runs']:
                    metrics = run_data['metrics_by_dataset'].get(dataset, {})
                    per_field_acc = metrics.get('per_field_accuracy', {})
                    accuracy = per_field_acc.get(entity, 0.0)
                    run_accuracies.append(accuracy)
                    row_data += f"{accuracy:.1%}".rjust(8)

                # Calculate average
                avg_accuracy = statistics.mean(run_accuracies) if run_accuracies else 0.0
                row_data += f"{avg_accuracy:.1%}".rjust(8)

                print(row_data)

            print()  # Blank line between datasets

        # ------------------------------------
        # Per-run details (COMMENTED OUT - not needed)
        # ------------------------------------
        # print(f"\n--- PER-RUN DETAILS ---")
        # for run_data in model_data['runs']:
        #     print(f"\n  Run {run_data['run']}:")
        #     print(f"    Total samples: {run_data['total_samples']}")
        #     print(f"    Parse failures: {run_data['parse_failures']}")
        #     print(f"    MICRO-F1: {run_data['overall_micro_f1']:.4f}")
        #     print(f"    MACRO-F1: {run_data['overall_macro_f1']:.4f}")
        #     print(f"    Perfect matches: {run_data['perfect_count']}/{run_data['total_samples']} ({run_data['perfect_ratio']:.2%})")
        #
        #     # Per-dataset metrics
        #     print(f"\n    Per-dataset breakdown:")
        #     for dataset, metrics in run_data['metrics_by_dataset'].items():
        #         print(f"      {dataset}:")
        #         print(f"        Samples: {metrics['num_samples']}")
        #         print(f"        Micro-F1: {metrics['micro_f1']:.4f}")
        #         print(f"        Perfect: {metrics['perfect_count']}/{metrics['num_samples']} ({metrics['perfect_ratio']:.2%})")

def save_results_to_json(aggregated_results: Dict[str, Any], output_path: str) -> None:
    """
    Save evaluation results to JSON file.
    
    Saves a summary without detailed per-sample results to keep file size small.
    
    Args:
        aggregated_results: Results from evaluate_all_files()
        output_path: Path to output JSON file
    """
    # Convert to serializable format
    serializable = {}
    
    for model_name, model_data in aggregated_results.items():
        runs_summary = []
        for run in model_data['runs']:
            run_summary = {
                'file': run['file'],
                'run': run['run'],
                'total_samples': run['total_samples'],
                'parse_failures': run['parse_failures'],
                'overall_micro_f1': run['overall_micro_f1'],
                'overall_macro_f1': run['overall_macro_f1'],
                'perfect_count': run['perfect_count'],
                'perfect_ratio': run['perfect_ratio'],
                'perfect_medications': run['perfect_medications'],
                'metrics_by_dataset': run['metrics_by_dataset'],
            }
            runs_summary.append(run_summary)
        
        serializable[model_name] = {
            'num_runs': model_data['num_runs'],
            'micro_f1_mean': model_data['micro_f1_mean'],
            'micro_f1_std': model_data['micro_f1_std'],
            'macro_f1_mean': model_data['macro_f1_mean'],
            'macro_f1_std': model_data['macro_f1_std'],
            'perfect_count_mean': model_data['perfect_count_mean'],
            'perfect_count_std': model_data['perfect_count_std'],
            'perfect_ratio_mean': model_data['perfect_ratio_mean'],
            'perfect_ratio_std': model_data['perfect_ratio_std'],
            'perfect_in_all_runs_count': model_data['perfect_in_all_runs_count'],
            'perfect_in_all_runs_drugs': model_data['perfect_in_all_runs_drugs'],
            'perfect_in_any_run_count': model_data['perfect_in_any_run_count'],
            'perfect_in_any_run_drugs': model_data['perfect_in_any_run_drugs'],
            'runs': runs_summary,
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def save_metadata_to_json(results_dir: Path, data_dir: Path, output_path: Path) -> None:
    metadata = {
        "dataset_version": dataset_version_for_path(data_dir),
        "dataset_dir": str(data_dir.resolve()),
        "results_dir": str(results_dir.resolve()),
        "scorer_version": SCORER_VERSION,
        "comparison_notice": (
            "Do not compare MedMatch results across dataset versions without explicit "
            "dataset-version disclosure."
        ),
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to: {output_path}")


def print_overall_results_table(aggregated_results: Dict[str, Any]) -> None:
    """
    Print overall results table matching Table 4 format.
    
    Shows Average Accuracy = (Run 1 percentage + Run 2 percentage + Run 3 percentage) / 3
    where each run's percentage = (Number of entries where ALL fields match exactly) / (Total entries) × 100%
    
    For each medication order type and LLM model.
    
    Args:
        aggregated_results: Results from evaluate_all_files()
    """
    # Model name mapping: internal name -> display name
    MODEL_NAME_MAPPING = {
        'gpt-4o-mini': 'GPT-4o-mini',
        'azure-gpt-5-chat': 'GPT-5 Chat',
        'google_gemma-3-27b-it': 'Gemma-3-27B-IT',
        'meta-llama_Llama-3.3-70B-Instruct': 'LLaMA-3.3-70B-Instruct',
        'Qwen_Qwen3-32B': 'Qwen3-32B',
    }
    
    # Medication order type mapping: dataset type -> display name
    ORDER_TYPE_MAPPING = {
        'po_solid': 'Oral solid',
        'po_liquid': 'Oral liquid',
        'iv_intermit': 'Intravenous intermittent',
        'iv_push': 'Intravenous push',
        'iv_continuous_titratable': 'Intravenous continuous infusion titratable',
        'iv_continuous_non_titratable': 'Intravenous continuous infusion non-titratable',
    }
    
    # Order types in the order they should appear in the table
    ORDER_TYPES = [
        'po_solid',
        'po_liquid',
        'iv_intermit',
        'iv_push',
        'iv_continuous_titratable',
        'iv_continuous_non_titratable',
    ]
    
    # Get all models (use display names)
    models = []
    model_keys = []
    preferred = [
        'gpt-4o-mini',
        'azure-gpt-5-chat',
        'google_gemma-3-27b-it',
        'meta-llama_Llama-3.3-70B-Instruct',
        'Qwen_Qwen3-32B',
    ]
    for mk in preferred:
        if mk in aggregated_results:
            models.append(MODEL_NAME_MAPPING.get(mk, mk))
            model_keys.append(mk)
    for mk in sorted(aggregated_results.keys()):
        if mk not in model_keys:
            models.append(MODEL_NAME_MAPPING.get(mk, mk))
            model_keys.append(mk)
    
    # Collect data: order_type -> model -> accuracy
    data = {}
    for order_type in ORDER_TYPES:
        data[order_type] = {}
        for model_key in model_keys:
            model_data = aggregated_results[model_key]
            # Get perfect_ratio for this order_type across all runs
            accuracies = []
            has_data = False
            for run_data in model_data['runs']:
                metrics = run_data['metrics_by_dataset'].get(order_type, {})
                num_samples = metrics.get('num_samples', 0)
                if num_samples > 0:
                    has_data = True
                    perfect_ratio = metrics.get('perfect_ratio', 0.0)
                    accuracies.append(perfect_ratio)
            
            # Calculate average accuracy across runs (only if we have data)
            if has_data and accuracies:
                avg_accuracy = statistics.mean(accuracies)
            else:
                avg_accuracy = float('nan')  # No data available
            data[order_type][model_key] = avg_accuracy
    
    # Print table header
    print("\n" + "=" * 100)
    print("Table 4. LLM accuracy on MedMatch medication order standards")
    print("=" * 100)
    print("\nData reported as average accuracy across three runs.")
    print("Average Accuracy = (Run 1 percentage + Run 2 percentage + Run 3 percentage) / 3")
    print("where each run's percentage = (Number of entries where ALL fields match exactly) / (Total entries) × 100%")
    print()
    
    # Print column headers
    header = "Medication Order Type".ljust(50)
    for model in models:
        header += model.rjust(25)
    print(header)
    print("-" * len(header))
    
    # Print each row
    for order_type in ORDER_TYPES:
        display_name = ORDER_TYPE_MAPPING.get(order_type, order_type)
        
        # Get sample count from first model's first run (all should have same counts)
        sample_count = 0
        if model_keys and aggregated_results[model_keys[0]]['runs']:
            first_run = aggregated_results[model_keys[0]]['runs'][0]
            metrics = first_run['metrics_by_dataset'].get(order_type, {})
            sample_count = metrics.get('num_samples', 0)
        
        row = f"{display_name} (n={sample_count})".ljust(50)
        
        for model_key in model_keys:
            accuracy = data[order_type][model_key]
            if accuracy != accuracy:  # Check for NaN
                row += "N/A".rjust(25)
            else:
                row += f"{accuracy:.1%}".rjust(25)
        
        print(row)
    
    print()


# ============================================================================
# SECTION 8: CSV GENERATION FUNCTIONS
# ============================================================================
# Generate CSV files with entity-level accuracy in "n (%)" format
# ============================================================================

# Entity display name mappings: CSV column name -> Display name
ENTITY_DISPLAY_NAMES = {
    # PO Solid
    'Drug name': 'Drug Name',
    'Numerical dose': 'Dose',
    'unit': 'Dose Unit',
    'amount': 'Amount',
    'formulation': 'Formulation',
    'frequency': 'Frequency',
    
    # PO Liquid
    'volume': 'Volume Amount',
    'volume unit of measure': 'Volume Unit',
    'concentration': 'Concentration',
    'formulation unit of measure': 'Concentration Unit',
    
    # IV Intermittent
    'drug': 'Drug Name',
    'dose': 'Dose',
    'amount of diluent volume': 'Diluent Amount',
    'diluent': 'Compatible Diluent',
    'infusion time': 'Infusion Time',
    
    # IV Push
    'volume': 'Diluent Amount',  # IV push uses 'volume' for diluent amount
    'volume unit': 'Volume Unit',
    'concentration': 'Concentration',
    'concentration unit': 'Concentration Unit',
    
    # IV Continuous
    'unit (dose)': 'Dose Unit',
    'diluent volume': 'Diluent Amount',
    'volume unit': 'Volume Unit',
    'starting rate': 'Starting Rate',
    'unit (rate)': 'Starting Rate Unit',
    'titration dose': 'Titration Dose',
    'titration unit': 'Titration Dose Unit',
    'titration frequency': 'Titration Frequency',
    'titration goal': 'Titration Goal',
}

def get_entity_display_name(csv_col_name: str, json_field_name: str, dataset_type: str) -> str:
    """
    Get display name for an entity.
    
    Args:
        csv_col_name: CSV column name
        json_field_name: JSON field name
        dataset_type: Dataset type
        
    Returns:
        Display name for the entity
    """
    # Dataset-specific handling
    if dataset_type == 'iv_push' and csv_col_name == 'volume':
        return 'Diluent Amount'
    elif dataset_type == 'po_liquid' and csv_col_name == 'volume':
        return 'Volume Amount'
    
    # First try CSV column name mapping
    if csv_col_name in ENTITY_DISPLAY_NAMES:
        return ENTITY_DISPLAY_NAMES[csv_col_name]
    
    # Special handling for common fields based on JSON field name
    if json_field_name == 'drug_name':
        return 'Drug Name'
    elif json_field_name == 'numerical_dose':
        return 'Dose'
    elif json_field_name == 'abbreviated_unit_strength_of_dose':
        return 'Dose Unit'
    elif json_field_name == 'frequency':
        return 'Frequency'
    elif json_field_name == 'formulation':
        return 'Formulation'
    elif json_field_name == 'amount_of_diluent_volume' or json_field_name == 'amount_of_volume':
        return 'Diluent Amount'
    elif json_field_name == 'volume_unit_of_measure':
        return 'Volume Unit'
    elif json_field_name == 'compatible_diluent_type':
        return 'Compatible Diluent'
    elif json_field_name == 'infusion_time':
        return 'Infusion Time'
    elif json_field_name == 'concentration_of_solution' or json_field_name == 'concentration_of_formulation':
        return 'Concentration'
    elif json_field_name == 'concentration_unit_of_measure':
        return 'Concentration Unit'
    elif json_field_name == 'starting_rate':
        return 'Starting Rate'
    elif json_field_name == 'unit_of_measure':
        return 'Starting Rate Unit'
    elif json_field_name == 'titration_dose':
        return 'Titration Dose'
    elif json_field_name == 'titration_unit_of_measure':
        return 'Titration Dose Unit'
    elif json_field_name == 'titration_frequency':
        return 'Titration Frequency'
    elif json_field_name == 'titration_goal_based_on_physiologic_response_laboratory_result_or_assessment_score':
        return 'Titration Goal'
    elif json_field_name == 'amount':
        return 'Amount'
    elif json_field_name == 'numerical_volume':
        return 'Volume Amount'
    
    # Fallback: capitalize JSON field name
    return json_field_name.replace('_', ' ').title()


def calculate_entity_accuracy_per_model(
    aggregated_results: Dict[str, Any],
    dataset_type: str,
    entity_field: str
) -> Tuple[int, float]:
    """
    Calculate accuracy for a specific entity across all runs for all models.
    
    Returns the aggregated accuracy: (n_correct, percentage)
    where n_correct is the total number of correct predictions across all runs,
    and percentage is the accuracy percentage.
    
    Args:
        aggregated_results: Results from evaluate_all_files()
        dataset_type: Dataset type (e.g., 'po_solid')
        entity_field: JSON field name (e.g., 'drug_name')
        
    Returns:
        Tuple of (n_correct, percentage) where n_correct is aggregated across all runs
    """
    total_correct = 0
    total_predictions = 0
    
    for model_name, model_data in aggregated_results.items():
        for run_data in model_data['runs']:
            metrics = run_data['metrics_by_dataset'].get(dataset_type, {})
            per_field_acc = metrics.get('per_field_accuracy', {})
            num_samples = metrics.get('num_samples', 0)
            
            if entity_field in per_field_acc and num_samples > 0:
                accuracy = per_field_acc[entity_field]
                n_correct = int(round(accuracy * num_samples))
                total_correct += n_correct
                total_predictions += num_samples
    
    if total_predictions == 0:
        return (0, 0.0)
    
    percentage = (total_correct / total_predictions) * 100.0
    return (total_correct, percentage)


def calculate_entity_accuracy_per_model_separate(
    aggregated_results: Dict[str, Any],
    dataset_type: str,
    entity_field: str,
    model_name: str
) -> Tuple[int, float]:
    """
    Calculate accuracy for a specific entity for a specific model across all runs.
    
    Accuracy is calculated as: average of per-run accuracies, where each run's accuracy
    is the percentage of order sentences (samples) that had this entity correct.
    
    Args:
        aggregated_results: Results from evaluate_all_files()
        dataset_type: Dataset type (e.g., 'po_solid')
        entity_field: JSON field name (e.g., 'drug_name')
        model_name: Model name (internal key)
        
    Returns:
        Tuple of (n_correct, percentage) where:
        - n_correct: average number of correct order sentences across runs
        - percentage: average accuracy percentage across runs
    """
    if model_name not in aggregated_results:
        return (0, 0.0)
    
    model_data = aggregated_results[model_name]
    run_accuracies = []
    run_correct_counts = []
    
    for run_data in model_data['runs']:
        metrics = run_data['metrics_by_dataset'].get(dataset_type, {})
        per_field_acc = metrics.get('per_field_accuracy', {})
        num_samples = metrics.get('num_samples', 0)
        
        if entity_field in per_field_acc and num_samples > 0:
            accuracy = per_field_acc[entity_field]  # This is already a ratio (0.0 to 1.0)
            percentage = accuracy * 100.0  # Convert to percentage
            n_correct = int(round(accuracy * num_samples))
            run_accuracies.append(percentage)
            run_correct_counts.append(n_correct)
    
    if not run_accuracies:
        return (0, 0.0)
    
    # Average accuracy across runs
    avg_percentage = statistics.mean(run_accuracies)
    avg_n_correct = int(round(statistics.mean(run_correct_counts)))
    
    return (avg_n_correct, avg_percentage)


def generate_entity_accuracy_csv(
    aggregated_results: Dict[str, Any],
    output_path: str
) -> None:
    """
    Generate CSV file with entity-level accuracy for all datasets.
    
    Includes: Oral solid, Oral liquid, Intravenous intermittent, IV push,
    IV continuous titratable, IV continuous non-titratable
    
    Columns: Entity, GPT-4o-mini, GPT-5 Chat, Gemma3, LLaMA3, Qwen3
    
    Args:
        aggregated_results: Results from evaluate_all_files()
        output_path: Path to save CSV file
    """
    # Model name mapping: internal name -> display name
    MODEL_NAME_MAPPING = {
        'gpt-4o-mini': 'GPT-4o-mini',
        'azure-gpt-5-chat': 'GPT-5 Chat',
        'google_gemma-3-27b-it': 'Gemma3',
        'meta-llama_Llama-3.3-70B-Instruct': 'LLaMA3',
        'Qwen_Qwen3-32B': 'Qwen3',
    }
    
    # Model order for CSV columns
    MODEL_ORDER = ['GPT-4o-mini', 'GPT-5 Chat', 'Gemma3', 'LLaMA3', 'Qwen3']
    
    # Dataset order and display names (all datasets)
    DATASETS = [
        ('po_solid', 'Oral solid'),
        ('po_liquid', 'Oral liquid'),
        ('iv_intermit', 'Intravenous intermittent'),
        ('iv_push', 'Intravenous push'),
        ('iv_continuous_titratable', 'Intravenous continuous infusion titratable'),
        ('iv_continuous_non_titratable', 'Intravenous continuous infusion non-titratable'),
    ]
    
    # Get sample counts
    def get_sample_count(dataset_type: str) -> int:
        for model_name in aggregated_results.keys():
            model_data = aggregated_results[model_name]
            if model_data['runs']:
                metrics = model_data['runs'][0]['metrics_by_dataset'].get(dataset_type, {})
                return metrics.get('num_samples', 0)
        return 0
    
    # Write CSV file
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header row
        header = ['Entity']
        for model in MODEL_ORDER:
            header.append(model)
        writer.writerow(header)
        
        # Process each dataset
        for dataset_type, dataset_display in DATASETS:
            # Write dataset header row
            sample_count = get_sample_count(dataset_type)
            dataset_row = [f"{dataset_display} (n={sample_count})"]
            dataset_row.extend([''] * len(MODEL_ORDER))
            writer.writerow(dataset_row)
            
            # Get entity mappings for this dataset
            base_dataset_type = _get_base_dataset_type(dataset_type)
            field_mapping = FIELD_MAPPINGS.get(base_dataset_type, {})
            
            # Process each entity
            for csv_col_name, json_field in field_mapping.items():
                # Filter out titration fields for non-titratable IV continuous
                if dataset_type == 'iv_continuous_non_titratable':
                    if json_field in ['titration_dose', 'titration_unit_of_measure', 
                                     'titration_frequency', 
                                     'titration_goal_based_on_physiologic_response_laboratory_result_or_assessment_score']:
                        continue
                
                entity_display = get_entity_display_name(csv_col_name, json_field, dataset_type)
                
                row = [entity_display]
                
                # Calculate accuracy for each model
                for model_display in MODEL_ORDER:
                    # Find internal model name
                    model_internal = None
                    for internal_name, display_name in MODEL_NAME_MAPPING.items():
                        if display_name == model_display:
                            model_internal = internal_name
                            break
                    
                    if model_internal:
                        n_correct, percentage = calculate_entity_accuracy_per_model_separate(
                            aggregated_results, dataset_type, json_field, model_internal
                        )
                        row.append(f"{percentage:.1f}%")
                    else:
                        row.append("")
                
                writer.writerow(row)
    
    print(f"CSV file saved to: {output_path}")


# ============================================================================
# SECTION 9: MAIN ENTRY POINT
# ============================================================================

def main():
    """
    Main function to run the complete evaluation pipeline.
    
    Steps:
    1. Set up paths (results directory, data directory, output file)
    2. Load ground truth from CSV files
    3. Evaluate all JSONL files
    4. Calculate per-run and cross-run metrics
    5. Print report and save to JSON
    """
    # ------------------------------------
    # Define paths
    # ------------------------------------
    results_root = Path(current_results_root())
    results_dir = results_root / 'one-shot'
    data_dir = Path(default_data_dir())
    output_path = results_root / 'evaluation_results.json'
    metadata_path = results_root / 'evaluation_results.metadata.json'
    
    # ------------------------------------
    # Verify directories exist
    # ------------------------------------
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    # ------------------------------------
    # Run evaluation
    # ------------------------------------
    print(f"Results directory: {results_dir}")
    print(f"Data directory: {data_dir}")
    print("-" * 60)
    
    aggregated_results = evaluate_all_files(str(results_dir), str(data_dir))
    
    # ------------------------------------
    # Print report and save results
    # ------------------------------------
    print_evaluation_report(aggregated_results)
    print_overall_results_table(aggregated_results)
    save_results_to_json(aggregated_results, str(output_path))
    save_metadata_to_json(results_dir, data_dir, metadata_path)
    
    # ------------------------------------
    # Generate CSV file with entity-level accuracy
    # ------------------------------------
    csv_path = results_root / 'entity_accuracy_table.csv'
    
    generate_entity_accuracy_csv(aggregated_results, str(csv_path))


if __name__ == '__main__':
    main()
