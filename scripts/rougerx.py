"""
RougeRx Survey Data Analysis - Comprehensive Medication Communication Analysis
==============================================================================

QUICK START COMMANDS:
=====================

# From the repository root (see README.md):

# Ultra-fast test (3 questions, ~30 seconds)
PYTHONPATH=. python -c "from scripts.rougerx import quick_test; quick_test()"

# Standard test (10 questions, ~2-3 minutes)
PYTHONPATH=. python -c "from scripts.rougerx import test_gpt4o_mini_analysis; test_gpt4o_mini_analysis(10)"

# Full GPT-4o-mini analysis (120 questions, ~10-15 minutes)
PYTHONPATH=. python -c "from scripts.rougerx import run_gpt4o_mini_analysis; run_gpt4o_mini_analysis()"

# Complete pipeline (all studies, ~20-30 minutes)
# python scripts/rougerx.py

Advanced Analysis Framework for Healthcare Professional Communication Patterns

CORE ANALYSIS CAPABILITIES:
========================

1. [EXPLORATORY STUDY 1] Word Overlap Analysis
   - NLTK tokenization across Q3-Q122 columns
   - Common word identification and frequency analysis
   - Communication style comparison (formal/verbal/brief)

2. [EXPLORATORY STUDY 2] Exact Answer Analysis
   - Percentage of identical answers across respondents
   - Reason analysis for answer variations
   - Pattern identification in response consistency

3. [EXPLORATORY STUDY 3] Advanced Medication Component Analysis
   - Single medication parsing per response
   - Component extraction: drug_name, dose, unit, route, frequency
   - Position-based component ordering within text
   - Divergence analysis across component categories

ADVANCED ANALYTICS FEATURES:
===========================

4. Component Overlap Analysis
   - Exact match vs partial overlap percentages per component
   - Quantitative measurement of terminology consistency
   - Cross-respondent comparison metrics

5. Similarity Analysis
   - Text similarity scoring between respondent answers
   - Average similarity per drug across 4 respondents
   - Communication style breakdown (formal/verbal/brief)

6. Positional Precision Analysis
   - Expected order validation: drug_name → dose → frequency → route
   - Component sequencing pattern analysis
   - Adherence to medication order conventions

7. Enhanced Omission Analysis
   - Detailed quantification of missing components
   - Omission rates by component type and communication style
   - Comparative analysis across medication categories

DATA STRUCTURE (Q3-Q122 Analysis Scope):
=====================================
- 120 question columns (Q3 to Q122)
- 40 medications, each with 3 response types:
  * Formal Written Order (Q3, Q6, Q9, ..., Q120)
  * Verbal Communication (Q4, Q7, Q10, ..., Q121)
  * Brief Written Communication (Q5, Q8, Q11, ..., Q122)
- 4 respondents (rows 1-4, row 0 contains prompts)
- Excludes survey metadata columns (StartDate, IPAddress, etc.)

OUTPUT FILES:
============
- rougerx_enhanced_analysis.json: Raw parsed data with component analysis
- rougerx_signatures_by_question_type.json: Organized by communication style
- rougerx_by_category.json: GPT-4o-mini component extraction with majority analysis
- common_words_summary.csv: Word frequency analysis summary

TECHNICAL FEATURES:
=================
- Single medication response parsing
- LLM-powered component extraction (with regex fallback)
- Position-aware component ordering
- Jaccard similarity calculations
- Statistical analysis across communication styles
- Comprehensive error handling and logging

# nohup python scripts/rougerx.py > rougerx.log 2>&1 &
"""

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter, defaultdict
import re
from typing import Dict, List, Tuple, Set, Optional, Any, Union
import json
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO_ROOT)

from src.prompt_rougerx import get_component_extraction_prompt, EXTRACTION_SYSTEM_PROMPT

# Parallel processing configuration
MAX_WORKERS: int = 30  # Sequences are to be processed in one batch (OpenAI only)

# Thread-safe locks
token_usage_lock = threading.Lock()
results_lock = threading.Lock()

# Cache for question index mapping (to handle missing questions like Q63)
_question_index_cache = None

def get_question_index(question_id: str, csv_path: str = None) -> int:
    """
    Get the actual index of a question in the CSV column list.
    This accounts for missing questions (e.g., Q63) that would break qnum-based calculation.
    
    Args:
        question_id: Question ID like 'Q66'
        csv_path: Path to CSV file (defaults to medmatch/data/rougerx.csv)
        
    Returns:
        Index in the CSV column list (0-based), or None if question not found
    """
    if csv_path is None:
        csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'rougerx.csv'))
    global _question_index_cache
    
    # Build cache if not exists
    if _question_index_cache is None:
        _question_index_cache = {}
        try:
            csv_df = pd.read_csv(csv_path)
            all_cols = list(csv_df.columns)
            q3_index = all_cols.index('Q3')
            q122_index = all_cols.index('Q122')
            question_cols = all_cols[q3_index:q122_index+1]
            question_cols = [col for col in question_cols if col.startswith('Q')]
            
            for i, col in enumerate(question_cols):
                _question_index_cache[col] = i
        except Exception as e:
            print(f"⚠️  Could not build question index cache: {e}")
            _question_index_cache = {}
    
    # Return cached index if available
    if question_id in _question_index_cache:
        return _question_index_cache[question_id]
    
    # If cache exists but question not found, return None (question is missing)
    if _question_index_cache:
        return None
    
    # Fallback: calculate from qnum (only if cache couldn't be built)
    # This should rarely happen, but provides a fallback
    if question_id.startswith('Q'):
        qnum = int(question_id[1:])
        return qnum - 3
    
    return None

def get_question_type_from_index(index: int) -> str:
    """
    Get question type from index using the same logic as parse_csv_data.
    
    Args:
        index: Index in CSV column list (0-based)
        
    Returns:
        'formal', 'verbal', or 'brief'
    """
    remainder = index % 3
    if remainder == 0:
        return 'formal'
    elif remainder == 1:
        return 'verbal'
    else:
        return 'brief'

# Try to import spaCy for advanced medical NLP
try:
    import spacy
    from spacy.lang.en import English
    from spacy.pipeline import EntityRuler
    SPACY_AVAILABLE = True
    print("✅ spaCy available for advanced medical parsing")
except ImportError:
    SPACY_AVAILABLE = False
    print("⚠️  spaCy not available - using regex fallback only")

# Add Pydantic for structured LLM output
try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    print("⚠️  Pydantic not available - using dict-based parsing")

# Pydantic model for structured medication parsing
if PYDANTIC_AVAILABLE:
    class MedicationComponents(BaseModel):
        drug_name: str = ""
        dose: str = ""
        unit: str = ""
        route: str = ""
        frequency: str = ""
        # Confidence scores for each component (0.0 to 1.0)
        drug_name_confidence: float = 0.0
        dose_confidence: float = 0.0
        unit_confidence: float = 0.0
        route_confidence: float = 0.0
        frequency_confidence: float = 0.0
        # Source tracking
        source_id: str = ""
        response_text: str = ""

# OpenAI integration for Study 3
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True

    # Initialize OpenAI client - explicitly load from .env file in current directory
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, '.env')

    print(f"🔍 Looking for .env file at: {env_path}")
    print(f"📂 Script directory: {script_dir}")
    print(f"📄 .env file exists: {os.path.exists(env_path)}")

    try:
        from dotenv import load_dotenv
        # Force override existing environment variables
        load_dotenv(env_path, override=True)
        print(f"📄 Loaded environment variables from: {env_path} (override=True)")
    except ImportError:
        print("⚠️  python-dotenv not available, using existing environment variables")

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("OPENAI_API_KEY is set (value not shown).")
        try:
            openai_client = OpenAI(api_key=api_key)
            print("✅ Initialized OpenAI client for Study 3")
            print("   GPT-powered medication parsing is now active!")
        except Exception as e:
            print(f"❌ Failed to initialize OpenAI client: {e}")
            openai_client = None
            OPENAI_AVAILABLE = False
    else:
        openai_client = None
        OPENAI_AVAILABLE = False
        print("⚠️  OpenAI API key not found - Study 3 will use fallback parsing")
        print("   To enable GPT parsing:")
        print("   1. Get an API key from https://platform.openai.com/api-keys")
        print("   2. Set environment variable: export OPENAI_API_KEY='your-key-here'")
        print("   3. Or create a .env file with: OPENAI_API_KEY=your-key-here")
        print("   4. Install python-dotenv: pip install python-dotenv")
except ImportError:
    openai_client = None
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI package not available - Study 3 will use fallback parsing")
    print("   To enable GPT parsing: pip install openai python-dotenv")

# Global token usage tracking for OpenAI API calls
TOKEN_USAGE = {
    'total_tokens': 0,
    'prompt_tokens': 0,
    'completion_tokens': 0,
    'api_calls': 0
}

def update_token_usage(response) -> None:
    """Update global token usage from OpenAI API response (thread-safe)."""
    global TOKEN_USAGE
    if hasattr(response, 'usage') and response.usage:
        usage = response.usage
        with token_usage_lock:
            TOKEN_USAGE['total_tokens'] += getattr(usage, 'total_tokens', 0)
            TOKEN_USAGE['prompt_tokens'] += getattr(usage, 'prompt_tokens', 0)
            TOKEN_USAGE['completion_tokens'] += getattr(usage, 'completion_tokens', 0)
            TOKEN_USAGE['api_calls'] += 1

def get_gpt4o_mini_cost() -> Dict[str, float]:
    """Calculate costs for gpt-4o-mini usage."""
    # Pricing as of October 2024: $0.150 per 1M input tokens, $0.600 per 1M output tokens
    input_cost_per_million = 0.150
    output_cost_per_million = 0.600

    input_tokens = TOKEN_USAGE['prompt_tokens']
    output_tokens = TOKEN_USAGE['completion_tokens']

    input_cost = (input_tokens / 1_000_000) * input_cost_per_million
    output_cost = (output_tokens / 1_000_000) * output_cost_per_million
    total_cost = input_cost + output_cost

    return {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': TOKEN_USAGE['total_tokens'],
        'input_cost_usd': round(input_cost, 4),
        'output_cost_usd': round(output_cost, 4),
        'total_cost_usd': round(total_cost, 4),
        'api_calls': TOKEN_USAGE['api_calls']
    }

# Ensure NLTK punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def parse_csv_data(csv_path: str) -> Tuple[pd.DataFrame, List[str], Dict[str, List[str]], int]:
    """
    Step 1: Parse CSV data and organize into structured format with question types

    FOCUS: Analyzes columns Q3 to Q122 only (questions and answers)
    - Q3-Q122: 120 question columns (40 medications × 3 question types each)
    - Row 0: Question prompts/descriptions
    - Rows 1-4: Respondent answers (4 healthcare professionals)

    Returns:
    - df: Raw dataframe
    - medications: List of medication names (from Q3-Q122 prompts)
    - responses: Dict with keys 'formal', 'verbal', 'brief', each containing Q3-Q122 column lists
    - num_respondents: Actual number of respondents (excluding header row)
    """
    df = pd.read_csv(csv_path)

    # Count actual respondents (rows with response data, excluding header)
    num_respondents = 0
    for i in range(len(df)):
        first_response = str(df.get('Q3', df.columns[0]).iloc[i]).strip()
        if first_response and first_response.lower() not in ['nan', ''] and 'Prompt:' not in first_response:
            num_respondents += 1

    # STEP 2: Extract ONLY Q3-Q122 columns (questions and answers)
    all_cols = list(df.columns)
    q3_index = all_cols.index('Q3')
    q122_index = all_cols.index('Q122')

    # Validate Q3-Q122 range
    question_cols = all_cols[q3_index:q122_index+1]  # Q3 to Q122 inclusive
    question_cols = [col for col in question_cols if col.startswith('Q')]  # Ensure only Q columns

    print(f"✓ Analyzing Q3-Q122 range: {len(question_cols)} columns ({question_cols[0]} to {question_cols[-1]})")

    if len(question_cols) != 120:
        raise ValueError(f"Expected 120 Q columns in Q3-Q123 range, found {len(question_cols)}")

    # STEP 3: Extract medication names from row 0 (prompts)
    medications = []
    for i, col in enumerate(question_cols):
        if i % 3 == 0:  # Every 3rd column starting from Q3
            cell_value = str(df[col].iloc[0])
            if 'Prompt:' in cell_value:
                med_name = cell_value.split('Prompt:')[1].split('\n')[0].strip()
                medications.append(med_name)

    print(f"✓ Extracted {len(medications)} medications from Q3-Q122 prompts")

    # Organize responses by type
    responses = {
        'formal': [],  # Q3, Q6, Q9, ...
        'verbal': [],  # Q4, Q7, Q10, ...
        'brief': []    # Q5, Q8, Q11, ...
    }

    for i, col in enumerate(question_cols):
        if i % 3 == 0:
            responses['formal'].append(col)
        elif i % 3 == 1:
            responses['verbal'].append(col)
        else:
            responses['brief'].append(col)

    return df, medications, responses, num_respondents

def tokenize_words(text: str, method: str = 'nltk') -> List[str]:
    """
    Tokenize text using NLTK word tokenizer for precision.

    Always applies: lowercase conversion and punctuation removal.

    Args:
        text: Text to tokenize
        method: Tokenization method (nltk recommended for precision)

    Returns:
        List of cleaned, tokenized words
    """
    if not text or str(text).strip() == '':
        return []

    # Step 1: Convert to lowercase
    text = str(text).lower().strip()

    # Step 2: Use NLTK word tokenizer for precision
    try:
        words = word_tokenize(text)
    except Exception as e:
        print(f"NLTK tokenization failed: {e}, falling back to simple split")
        words = text.split()

    # Step 3: Remove punctuation and clean words
    cleaned_words = []
    for word in words:
        # Remove all punctuation, keep only alphanumeric characters
        cleaned = re.sub(r'[^\w\s]', '', word)
        # Remove extra whitespace and filter
        cleaned = cleaned.strip()
        if cleaned and len(cleaned) > 0:
            cleaned_words.append(cleaned)

    return cleaned_words

def exploratory_study_1_word_overlap(df: pd.DataFrame, focus_columns: List[str] = ['Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8']) -> Dict[str, Dict]:
    """
    EXPLORATORY STUDY 1: Word overlap analysis using NLTK tokenization (Q3-Q122 focus)

    For each question in focus_columns (all Q3-Q122 columns = 120 questions):
    - Get 4 responses (from rows 1-4, skipping row 0 which has prompts)
    - Apply NLTK word tokenization with preprocessing:
      * Convert to lowercase
      * Remove punctuation
      * Tokenize with linguistic precision
    - Create word sets for each response
    - Find common words across all 4 responses (intersection)
    - Find words appearing in majority (≥75%) of responses
    - Find unique words per response
    - Calculate Jaccard similarity between response pairs

    Args:
        df: DataFrame with survey data
        focus_columns: Columns to analyze (Q3-Q122 = 120 question columns)

    Returns:
        Dict with analysis results per question
    """
    results = {}

    for col in focus_columns:
        if col not in df.columns:
            continue

        # Get responses for this question (rows 1-4, skip row 0 with prompts)
        responses = []
        for row_idx in range(1, min(5, len(df))):  # Rows 1-4 (up to 4 respondents)
            response = str(df[col].iloc[row_idx]).strip()
            if response and response.lower() not in ['nan', '']:
                responses.append(response)

        if len(responses) < 2:
            continue  # Need at least 2 responses for meaningful overlap

        # Create word sets for each response
        response_word_sets = []
        response_words_lists = []

        for response in responses:
            words = tokenize_words(response, method='simple')  # Use simple split as requested
            word_set = set(words)
            response_word_sets.append(word_set)
            response_words_lists.append(words)

        # Find common words across ALL responses (intersection)
        if response_word_sets:
            common_words_all = set.intersection(*response_word_sets)
        else:
            common_words_all = set()

        # Find words that appear in MOST responses (at least 3 out of 4)
        all_words = set()
        word_counts = Counter()
        for word_set in response_word_sets:
            all_words.update(word_set)
            for word in word_set:
                word_counts[word] += 1

        # Words appearing in majority (at least 75% of responses)
        majority_threshold = max(1, len(responses) * 3 // 4)
        majority_words = {word: count for word, count in word_counts.items()
                         if count >= majority_threshold}

        # Unique words per response (words that don't appear in any other response)
        unique_words_per_response = []
        for i, word_set in enumerate(response_word_sets):
            other_sets = response_word_sets[:i] + response_word_sets[i+1:]
            if other_sets:
                other_words = set.union(*other_sets)
            else:
                other_words = set()
            unique_words = word_set - other_words
            unique_words_per_response.append({
                'response_index': i + 1,
                'response_text': responses[i][:50] + '...' if len(responses[i]) > 50 else responses[i],
                'unique_words': list(unique_words),
                'total_words': len(word_set)
            })

        # Calculate Jaccard similarity between pairs
        jaccard_similarities = []
        for i in range(len(response_word_sets)):
            for j in range(i+1, len(response_word_sets)):
                set1, set2 = response_word_sets[i], response_word_sets[j]
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                jaccard = intersection / union if union > 0 else 0
                jaccard_similarities.append({
                    'pair': f'R{i+1}_vs_R{j+1}',
                    'jaccard_similarity': jaccard,
                    'intersection_size': intersection,
                    'union_size': union
                })

        # Store results for this question
        results[col] = {
            'question_info': {
                'column': col,
                'total_responses': len(responses),
                'responses': responses
            },
            'word_sets': {
                'response_sets': [list(s) for s in response_word_sets],
                'all_unique_words': len(all_words),
                'total_words_across_all': sum(len(words) for words in response_words_lists)
            },
            'common_analysis': {
                'common_words_all': list(common_words_all),
                'num_common_words_all': len(common_words_all),
                'majority_words': majority_words,
                'num_majority_words': len(majority_words)
            },
            'unique_analysis': {
                'unique_words_per_response': unique_words_per_response,
                'total_unique_across_all': sum(len(resp['unique_words']) for resp in unique_words_per_response)
            },
            'similarity_analysis': {
                'jaccard_similarities': jaccard_similarities,
                'avg_jaccard_similarity': sum(s['jaccard_similarity'] for s in jaccard_similarities) / len(jaccard_similarities) if jaccard_similarities else 0
            },
            'word_frequencies': {
                'top_words': dict(word_counts.most_common(10)),
                'word_counts': dict(word_counts)
            }
        }

    return results

def exploratory_study_2_exact_answers(df: pd.DataFrame, medications: List[str], response_cols: Dict[str, List[str]]) -> Dict[str, Dict]:
    """
    EXPLORATORY STUDY 2: Exact answer analysis - organized by question type (3 questions with 40 answers each)

    Analyzes identical answers organized by communication type instead of by medication.
    Returns 3 main sections (formal/verbal/brief) with 40 answers each (one per medication).
    """
    results = {
        'formal_question': {'answers': []},
        'verbal_question': {'answers': []},
        'brief_question': {'answers': []},
        'overall_stats': {},
        'patterns_analysis': {}
    }

    # Organize by communication type (3 questions)
    for comm_type in ['formal', 'verbal', 'brief']:
        question_key = f'{comm_type}_question'
        cols = response_cols[comm_type]

        # For each medication in this communication type (40 answers)
        for i, col in enumerate(cols):
            if col not in df.columns:
                continue

            medication = medications[i] if i < len(medications) else f"Medication_{i+1}"

            # Get responses for this question (rows 1-4, skip row 0 with prompts)
            responses = []
            for row_idx in range(1, min(5, len(df))):  # Rows 1-4 (4 respondents)
                response = str(df[col].iloc[row_idx]).strip()
                if response and response.lower() not in ['nan', '']:
                    responses.append(response)

            if len(responses) < 4:  # Need all 4 respondents
                continue

            # Check if all responses are identical
            all_identical = all(resp == responses[0] for resp in responses)

            answer_data = {
                'medication': medication,
                'question_column': col,
                'communication_type': comm_type,
                'responses': responses,
                'all_identical': all_identical,
                'unique_responses': len(set(responses)),
                'response_length_avg': sum(len(r) for r in responses) / len(responses),
                'respondent_count': len(responses)
            }

            results[question_key]['answers'].append(answer_data)

    # Calculate overall statistics across all question types
    total_answers = sum(len(results[q]['answers']) for q in ['formal_question', 'verbal_question', 'brief_question'])
    identical_answers = sum(sum(1 for a in results[q]['answers'] if a['all_identical']) for q in ['formal_question', 'verbal_question', 'brief_question'])
    identical_percentage = identical_answers / total_answers * 100 if total_answers > 0 else 0

    results['overall_stats'] = {
        'total_answers_analyzed': total_answers,
        'identical_answers_count': identical_answers,
        'divergent_answers_count': total_answers - identical_answers,
        'identical_percentage': identical_percentage,
        'divergent_percentage': 100 - identical_percentage
    }

    # Analyze patterns by question type
    question_type_stats = {}
    for comm_type in ['formal', 'verbal', 'brief']:
        question_key = f'{comm_type}_question'
        answers = results[question_key]['answers']
        identical_in_type = [a for a in answers if a['all_identical']]
        percentage = len(identical_in_type) / len(answers) * 100 if answers else 0

        question_type_stats[comm_type] = {
            'total_answers': len(answers),
            'identical_answers': len(identical_in_type),
            'identical_percentage': percentage,
            'avg_response_length': sum(a['response_length_avg'] for a in answers) / len(answers) if answers else 0
        }

    results['question_type_patterns'] = question_type_stats

    # Analyze patterns across all answers
    all_answers = []
    for comm_type in ['formal', 'verbal', 'brief']:
        all_answers.extend(results[f'{comm_type}_question']['answers'])

    identical_lengths = [a['response_length_avg'] for a in all_answers if a['all_identical']]
    divergent_lengths = [a['response_length_avg'] for a in all_answers if not a['all_identical']]

    results['patterns_analysis'] = {
        'response_length_comparison': {
            'identical_avg_length': sum(identical_lengths) / len(identical_lengths) if identical_lengths else 0,
            'divergent_avg_length': sum(divergent_lengths) / len(divergent_lengths) if divergent_lengths else 0
        },
        'common_identical_responses': {},
        'divergence_reasons': []
    }

    # Find most common identical responses across all question types
    identical_responses = [a['responses'][0] for a in all_answers if a['all_identical']]  # All identical, so take first
    from collections import Counter
    response_counts = Counter(identical_responses)
    results['patterns_analysis']['common_identical_responses'] = dict(response_counts.most_common(10))

    # Analyze potential reasons for divergence
    divergence_reasons = []

    # Check for answers with high unique response counts
    high_divergence = [a for a in all_answers if not a['all_identical'] and a['unique_responses'] == 4]  # All 4 different
    if high_divergence:
        divergence_reasons.append(f"{len(high_divergence)} answers have completely unique responses from all 4 respondents")

    # Check for question type differences
    verbal_divergent = [a for a in all_answers if not a['all_identical'] and a['communication_type'] == 'verbal']
    if verbal_divergent:
        avg_length = sum(a['response_length_avg'] for a in verbal_divergent) / len(verbal_divergent)
        divergence_reasons.append(".1f")

    results['patterns_analysis']['divergence_reasons'] = divergence_reasons

    return results

def parse_medication_order_simple(order_text: str, include_ordering: bool = False) -> Dict[str, Any]:
    """
    Extract drug components using OpenAI LLM with fallback to regex parsing.

    Args:
        order_text: Text to parse for medication components
        include_ordering: Whether to include component ordering information (optional, adds complexity)

    Returns components with optional ordering based on their position in the original text.
    Prompts: "Extract drug name, dose, unit, route, frequency from this sentence: [text]"
    """
    if not OPENAI_AVAILABLE or not openai_client:
        # Fallback to regex parsing if OpenAI is not available
        parsed = parse_medication_order_fallback(order_text)
        return add_component_ordering(parsed, order_text) if include_ordering else parsed

    if PYDANTIC_AVAILABLE:
        # Use structured output with Pydantic and enhanced medical context
        prompt = f"""You are a medical AI assistant extracting medication components from healthcare provider responses.

Extract medication components from this text: "{order_text}"

Medical Context:
- Drug names are typically at the beginning (e.g., "Lisinopril", "D5W", "Insulin")
- Doses are numbers followed by units (e.g., "10 mg", "100 mL", "50 units")
- Routes indicate how medication is given (PO=oral, IV=intravenous, IM=intramuscular, SC=subcutaneous, etc.)
- Frequency shows timing (daily, BID=twice daily, q8h=every 8 hours, PRN=as needed)

Examples:
- "Lisinopril 10 mg daily" → drug_name: "Lisinopril", dose: "10", unit: "mg", route: "PO", frequency: "daily"
- "D5W @ 100 mL/hour IV" → drug_name: "D5W", dose: "100", unit: "mL", route: "IV", frequency: "hour"
- "Morphine 4 mg IV q4h PRN" → drug_name: "Morphine", dose: "4", unit: "mg", route: "IV", frequency: "q4h PRN"

Extract these components with high precision:
- drug_name: The medication name (be specific, include brand/generic as written)
- dose: The dose amount (numeric value only)
- unit: The unit of measurement (mg, mL, units, mcg, etc.)
- route: The administration route (PO, IV, IM, SC, etc.)
- frequency: How often to administer (daily, BID, q8h, continuous, etc.)

For each component, provide both the extracted value and a confidence score (0.0-1.0) indicating how certain you are of the extraction."""

        try:
            response = openai_client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format=MedicationComponents,
                max_tokens=512,
                temperature=1.2
            )

            # Update token usage tracking
            update_token_usage(response)

            # Parse with Pydantic
            parsed_obj = response.choices[0].message.parsed
            parsed = parsed_obj.model_dump()

            # Ensure confidence scores are present (set defaults if missing)
            confidence_fields = ['drug_name_confidence', 'dose_confidence', 'unit_confidence', 'route_confidence', 'frequency_confidence']
            for field in confidence_fields:
                if field not in parsed or parsed[field] is None:
                    parsed[field] = 0.8  # Default moderate confidence for structured parsing

            # Add ordering based on component positions in original text
            return add_component_ordering(parsed, order_text) if include_ordering else parsed

        except Exception as e:
            print(f"⚠️  Structured parsing failed, falling back to JSON: {e}")

    # Fallback to JSON parsing if Pydantic not available or structured parsing fails
    prompt = f"""You are a medical AI assistant extracting medication components from healthcare provider responses.

Extract medication components from this text: "{order_text}"

Medical Context:
- Drug names are typically at the beginning (e.g., "Lisinopril", "D5W", "Insulin")
- Doses are numbers followed by units (e.g., "10 mg", "100 mL", "50 units")
- Routes indicate how medication is given (PO=oral, IV=intravenous, IM=intramuscular, SC=subcutaneous, etc.)
- Frequency shows timing (daily, BID=twice daily, q8h=every 8 hours, PRN=as needed)

Examples:
- "Lisinopril 10 mg daily" → drug_name: "Lisinopril", dose: "10", unit: "mg", route: "PO", frequency: "daily"
- "D5W @ 100 mL/hour IV" → drug_name: "D5W", dose: "100", unit: "mL", route: "IV", frequency: "hour"
- "Morphine 4 mg IV q4h PRN" → drug_name: "Morphine", dose: "4", unit: "mg", route: "IV", frequency: "q4h PRN"

Return ONLY a JSON object with these exact keys:
drug_name, dose, unit, route, frequency,
drug_name_confidence, dose_confidence, unit_confidence, route_confidence, frequency_confidence

Use empty string "" if component not found, and confidence scores between 0.0-1.0."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.1
        )

        # Update token usage tracking
        update_token_usage(response)

        result_text = response.choices[0].message.content.strip()

        # Clean up response (remove markdown if present)
        result_text = result_text.strip('`').strip()
        if result_text.startswith('json'):
            result_text = result_text[4:].strip()

        # Parse JSON
        try:
            parsed = json.loads(result_text)
            # Ensure all keys exist
            required_keys = ['drug_name', 'dose', 'unit', 'route', 'frequency']
            confidence_keys = ['drug_name_confidence', 'dose_confidence', 'unit_confidence', 'route_confidence', 'frequency_confidence']

            for key in required_keys:
                if key not in parsed:
                    parsed[key] = ""
            for key in confidence_keys:
                if key not in parsed or parsed[key] is None:
                    parsed[key] = 0.6  # Lower default confidence for JSON fallback

            # Add ordering based on component positions in original text
            return add_component_ordering(parsed, order_text) if include_ordering else parsed
        except json.JSONDecodeError:
            print(f"⚠️  LLM returned invalid JSON: {result_text[:100]}...")
            parsed = {
                'drug_name': '', 'dose': '', 'unit': '', 'route': '', 'frequency': '',
                'drug_name_confidence': 0.0, 'dose_confidence': 0.0, 'unit_confidence': 0.0,
                'route_confidence': 0.0, 'frequency_confidence': 0.0
            }
            return add_component_ordering(parsed, order_text) if include_ordering else parsed

    except Exception as e:
        print(f"❌ Error calling OpenAI API: {e}")
        parsed = {
            'drug_name': '', 'dose': '', 'unit': '', 'route': '', 'frequency': '',
            'drug_name_confidence': 0.0, 'dose_confidence': 0.0, 'unit_confidence': 0.0,
            'route_confidence': 0.0, 'frequency_confidence': 0.0
        }
        return add_component_ordering(parsed, order_text) if include_ordering else parsed

def parse_medication_from_response(response_text: str, source_id: str = "") -> Optional[Dict[str, Any]]:
    """
    Parse medication from a single response text with source tracking.

    We only consider single drug responses. Each response contains one medication order.

    Args:
        response_text: The text to parse for medication components
        source_id: Identifier for the response source (e.g., "ID_1", "ID_2")

    Returns a medication dictionary containing:
    - drug_name, dose, unit, route, frequency (standard components)
    - confidence scores for each component
    - component ordering fields (drug_name_ordering, dose_ordering, etc.) based on position in the original text
    - source tracking information

    Returns None if parsing fails or text is empty.
    """
    if not response_text or str(response_text).strip() == '':
        return None

    # Parse the entire response as a single medication
    parsed = parse_medication_order_simple(response_text, include_ordering=True)
    if parsed and any(parsed.get(key, '') for key in ['drug_name', 'dose', 'unit', 'route', 'frequency']):
        # Add source tracking information
        parsed['source_id'] = source_id
        parsed['response_text'] = response_text.strip()
        return parsed

    # If parsing failed, return None
    return None

def add_component_ordering(parsed_components: Dict[str, Any], original_text: str) -> Dict[str, Any]:
    """
    Enhanced ordering detection for medication components with medical terminology awareness.

    Uses multiple strategies to accurately determine component positions:
    1. Direct string matching with medical abbreviations
    2. Word boundary matching for precise terms
    3. Pattern-based matching for complex medical expressions
    4. Confidence-weighted positioning

    Returns the parsed components with enhanced ordering information.
    """
    if not original_text or not parsed_components:
        return parsed_components

    # Medical terminology mappings for better matching
    medical_abbreviations = {
        'po': ['po', 'orally', 'by mouth', 'per os'],
        'iv': ['iv', 'intravenous', 'ivp', 'iv push'],
        'im': ['im', 'intramuscular'],
        'sc': ['sc', 'subcutaneous', 'subq', 'sq'],
        'prn': ['prn', 'as needed', 'as required'],
        'qd': ['qd', 'daily', 'once daily'],
        'bid': ['bid', 'twice daily', '2x daily'],
        'tid': ['tid', 'three times daily', '3x daily'],
        'qid': ['qid', 'four times daily', '4x daily'],
        'q8h': ['q8h', 'every 8 hours'],
        'q12h': ['q12h', 'every 12 hours'],
        'mg': ['mg', 'milligrams', 'milligram'],
        'ml': ['ml', 'mL', 'milliliters', 'milliliter'],
        'mcg': ['mcg', 'micrograms', 'microgram'],
        'units': ['units', 'unit', 'u', 'iu']
    }

    # Track positions and confidence scores for each component
    component_positions = {}
    text_lower = original_text.lower()

    # For each component that was found, find its position using multiple strategies
    for component_name, component_value in parsed_components.items():
        if component_name.endswith('_confidence') or component_name.endswith('_ordering') or component_name in ['source_id', 'response_text']:
            continue
        if not component_value or component_value == "":
            continue

        value_lower = component_value.lower().strip()
        best_position = -1
        best_confidence = 0.0

        # Strategy 1: Direct exact match
        position = text_lower.find(value_lower)
        if position != -1:
            best_position = position
            best_confidence = 0.9

        # Strategy 2: Word boundary match for better precision
        if best_position == -1:
            import re
            pattern = r'\b' + re.escape(value_lower) + r'\b'
            match = re.search(pattern, text_lower)
            if match:
                best_position = match.start()
                best_confidence = 0.8

        # Strategy 3: Medical abbreviation expansion matching
        if best_position == -1 and component_name in ['route', 'frequency', 'unit']:
            for standard_term, variations in medical_abbreviations.items():
                if value_lower in variations or any(var in value_lower for var in variations):
                    # Look for any variation in the text
                    for variation in variations:
                        pos = text_lower.find(variation)
                        if pos != -1 and (best_position == -1 or pos < best_position):
                            best_position = pos
                            best_confidence = 0.7
                            break

        # Strategy 4: Fuzzy matching for dose numbers (handle variations like "10" vs "10.0")
        if best_position == -1 and component_name == 'dose':
            import re
            # Look for numeric patterns that match the dose
            numeric_patterns = [
                r'\b' + re.escape(value_lower) + r'\b',  # Exact match
                r'\b\d+\.?\d*\b',  # Any number that could match
            ]
            for pattern in numeric_patterns:
                matches = list(re.finditer(pattern, text_lower))
                for match in matches:
                    matched_number = match.group()
                    # Check if it's a reasonable match (same number, possibly with decimals)
                    try:
                        if abs(float(matched_number) - float(value_lower)) < 0.01:
                            if best_position == -1 or match.start() < best_position:
                                best_position = match.start()
                                best_confidence = 0.6
                    except ValueError:
                        continue

        # Note: We only consider explicitly stated entities in the text.
        # Inferred components (like routes inferred from frequency) get ordering = 0

        # Store the best position found with its confidence
        if best_position != -1:
            component_positions[component_name] = {
                'position': best_position,
                'confidence': best_confidence
            }

    # Sort components by their position (earliest first), then by confidence
    sorted_components = sorted(
        component_positions.items(),
        key=lambda x: (x[1]['position'], -x[1]['confidence'])  # Lower position first, higher confidence first
    )

    # Assign ordering numbers (1-based) and adjust confidence based on ordering clarity
    ordering_map = {}
    for order, (component_name, position_data) in enumerate(sorted_components, 1):
        ordering_map[component_name] = order

        # Adjust confidence based on whether ordering is clear
        # (components that are clearly positioned get higher ordering confidence)
        if len(sorted_components) > 1:
            position_gaps = []
            for i in range(1, len(sorted_components)):
                gap = sorted_components[i][1]['position'] - sorted_components[i-1][1]['position']
                position_gaps.append(gap)

            avg_gap = sum(position_gaps) / len(position_gaps) if position_gaps else 0
            # If gaps are reasonable (>5 chars), ordering is more reliable
            if avg_gap > 5:
                position_data['confidence'] = min(1.0, position_data['confidence'] + 0.1)

    # Add ordering information to the parsed components
    result = parsed_components.copy()
    for component_name in ['drug_name', 'dose', 'unit', 'route', 'frequency']:
        ordering_key = f"{component_name}_ordering"
        if component_name in ordering_map:
            result[ordering_key] = ordering_map[component_name]
            # Add ordering confidence as well
            result[f"{component_name}_ordering_confidence"] = component_positions[component_name]['confidence']
        else:
            result[ordering_key] = 0  # 0 means component not found in text
            result[f"{component_name}_ordering_confidence"] = 0.0

    return result

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts using word overlap and Jaccard similarity.

    Returns a similarity score between 0.0 and 1.0.
    """
    if not text1 or not text2:
        return 0.0

    # Convert to lowercase and tokenize
    words1 = set(word_tokenize(text1.lower()))
    words2 = set(word_tokenize(text2.lower()))

    # Remove common stop words that don't contribute to meaning
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'take', 'tablet', 'tablets', 'daily', 'every', 'as', 'needed', 'prn'}
    words1 = words1 - stop_words
    words2 = words2 - stop_words

    if not words1 and not words2:
        return 1.0  # Both empty after stop word removal
    if not words1 or not words2:
        return 0.0  # One is empty

    # Calculate Jaccard similarity
    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0

def analyze_component_overlap(components_list: List[Dict[str, Any]], component_name: str) -> Dict[str, Any]:
    """
    Analyze overlap vs exact matches for a specific component across multiple responses.
    Uses strict whole word matching to avoid false positives from word fragments.

    Returns statistics about exact matches vs whole word overlaps.
    """
    if not components_list or len(components_list) < 2:
        return {'total_comparisons': 0, 'exact_matches': 0, 'partial_overlaps': 0, 'exact_percentage': 0, 'overlap_percentage': 0}

    values = []
    for comp in components_list:
        value = comp.get(component_name, '').strip().lower()
        if value:
            values.append(value)

    if len(values) < 2:
        return {'total_comparisons': 0, 'exact_matches': 0, 'partial_overlaps': 0, 'exact_percentage': 0, 'overlap_percentage': 0}

    exact_matches = 0
    partial_overlaps = 0
    total_comparisons = 0

    # Compare each pair
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            total_comparisons += 1
            val1, val2 = values[i], values[j]

            if val1 == val2:
                exact_matches += 1
            else:
                # Strict whole word overlap: check if complete words are shared
                # Split on whitespace
                words1 = set(val1.split())
                words2 = set(val2.split())

                # Only count as overlap if they share meaningful complete words
                common_words = words1 & words2
                if common_words:
                    # Additional check: ensure shared words are not just common stop words
                    meaningful_words = common_words - {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
                    if meaningful_words:
                        partial_overlaps += 1

    return {
        'total_comparisons': total_comparisons,
        'exact_matches': exact_matches,
        'partial_overlaps': partial_overlaps,
        'exact_percentage': (exact_matches / total_comparisons * 100) if total_comparisons > 0 else 0,
        'overlap_percentage': ((exact_matches + partial_overlaps) / total_comparisons * 100) if total_comparisons > 0 else 0
    }

def analyze_positional_precision(components_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze if components appear in expected order: drug_name -> dose -> frequency -> route

    Returns statistics about positional precision.
    """
    expected_order = ['drug_name_ordering', 'dose_ordering', 'frequency_ordering', 'route_ordering']
    results = {
        'total_responses': len(components_list),
        'correct_order_count': 0,
        'component_position_stats': {}
    }

    for comp in components_list:
        orderings = []
        for order_field in expected_order:
            order = comp.get(order_field, 0)
            if order > 0:  # Only consider components that were found
                orderings.append(order)

        # Check if orderings are in ascending order (1, 2, 3, 4...)
        if len(orderings) >= 2 and orderings == sorted(orderings):
            results['correct_order_count'] += 1

        # Track position statistics for each component
        for order_field in expected_order:
            if order_field not in results['component_position_stats']:
                results['component_position_stats'][order_field] = {'positions': [], 'avg_position': 0}

            pos = comp.get(order_field, 0)
            if pos > 0:
                results['component_position_stats'][order_field]['positions'].append(pos)

    # Calculate average positions
    for order_field, stats in results['component_position_stats'].items():
        if stats['positions']:
            stats['avg_position'] = sum(stats['positions']) / len(stats['positions'])

    results['positional_precision'] = (results['correct_order_count'] / results['total_responses'] * 100) if results['total_responses'] > 0 else 0

    return results

def calculate_drug_similarity_analysis(enhanced_data: Dict) -> Dict[str, Any]:
    """
    Calculate similarity analysis for each drug across 4 respondents and communication styles.

    Returns detailed similarity statistics.
    """
    results = {
        'by_drug': {},
        'by_style': {},
        'overall': {'avg_similarity': 0, 'total_comparisons': 0}
    }

    style_similarity_scores = {'formal': [], 'verbal': [], 'brief': []}

    for question_id, question_data in enhanced_data.items():
        if not question_id.startswith('Q'):
            continue

        # Determine communication style using actual CSV index (handles missing questions)
        index = get_question_index(question_id)
        if index is None:
            continue
        style = get_question_type_from_index(index)

        if 'data' in question_data and 'responses' in question_data['data']:
            responses = question_data['data']['responses']
            response_texts = [responses.get(f'ID_{i}', '') for i in range(1, 5)]
            response_texts = [r.strip() for r in response_texts if r.strip()]

            if len(response_texts) >= 2:
                # Calculate pairwise similarities
                similarities = []
                for i in range(len(response_texts)):
                    for j in range(i + 1, len(response_texts)):
                        sim = calculate_text_similarity(response_texts[i], response_texts[j])
                        similarities.append(sim)

                if similarities:
                    avg_similarity = sum(similarities) / len(similarities)
                    style_similarity_scores[style].append(avg_similarity)

                    # Store by question
                    results['by_drug'][question_id] = {
                        'style': style,
                        'avg_similarity': avg_similarity,
                        'response_count': len(response_texts),
                        'comparisons': len(similarities)
                    }

    # Calculate averages by style
    for style, scores in style_similarity_scores.items():
        if scores:
            results['by_style'][style] = {
                'avg_similarity': sum(scores) / len(scores),
                'question_count': len(scores)
            }

    # Calculate overall average
    all_scores = []
    for style_scores in style_similarity_scores.values():
        all_scores.extend(style_scores)

    if all_scores:
        results['overall']['avg_similarity'] = sum(all_scores) / len(all_scores)
        results['overall']['total_comparisons'] = len(all_scores)

    return results

def parse_medication_order_fallback(order_text: str) -> Dict[str, str]:
    """
    Enhanced fallback regex-based parsing when LLM is not available.
    Extracts: drug_name, dose, unit, route, frequency
    """
    # Initialize components
    drug_name = ""
    dose = ""
    unit = ""
    route = ""
    frequency = ""

    # Enhanced drug name extraction with compound names and medical terminology
    words = order_text.lower().split()
    
    # Comprehensive exclusion list including numbers and medical terms
    non_drug_words = {
        'administer', 'take', 'give', 'inject', 'infuse', 'intravenous', 'oral', 'by', 'mouth', 
        'daily', 'every', 'q', 'mg', 'ml', 'mcg', 'g', 'units', 'tablet', 'capsule', 'solution',
        'at', 'over', 'for', 'per', 'hour', 'hours', 'minutes', 'min', 'bolus', 'infusion',
        'one', 'two', 'three', 'four', 'five', 'once', 'twice', 'the', 'a', 'an', 'and', 'of'
    }
    
    # Common compound drug names patterns
    compound_drugs = {
        'normal saline': 'Normal saline',
        'sodium chloride': 'Sodium chloride', 
        'lactated ringers': 'Lactated Ringers',
        'dextrose water': 'Dextrose in water',
        'potassium chloride': 'Potassium chloride',
        'magnesium sulfate': 'Magnesium sulfate',
        'calcium gluconate': 'Calcium gluconate',
        'iron sucrose': 'Iron sucrose',
        'vitamin b12': 'Vitamin B12',
        'folic acid': 'Folic acid'
    }
    
    # Check for compound drug names first
    text_lower = order_text.lower()
    for compound, proper_name in compound_drugs.items():
        if compound in text_lower:
            drug_name = proper_name
            break
    
    # Common abbreviations mapping
    drug_abbreviations = {
        'ns': 'Normal saline',
        'lr': 'Lactated Ringers', 
        'kcl': 'Potassium chloride',
        'mgso4': 'Magnesium sulfate',
        'd5w': 'Dextrose 5% in water',
        'nacl': 'Sodium chloride'
    }
    
    # Check for abbreviations - preserve original abbreviation if it appears as a standalone word
    if not drug_name:
        words = text_lower.split()
        for abbrev, full_name in drug_abbreviations.items():
            # Only expand if the abbreviation is a complete word (not part of another word)
            if abbrev in words:
                # Keep the original abbreviation as it appears in the text
                original_abbrev = ""
                for word in order_text.split():
                    if word.lower() == abbrev:
                        original_abbrev = word
                        break
                drug_name = original_abbrev.upper() if original_abbrev else abbrev.upper()
                break
    
    # Fallback: extract first meaningful word (excluding numbers and common terms)
    if not drug_name:
        words = order_text.lower().split()
        for word in words:
            # Skip numbers, short words, and common non-drug terms
            if (not word.isdigit() and 
                word not in non_drug_words and 
                len(word) > 2 and
                not re.match(r'^\d+\.?\d*$', word)):  # Skip decimal numbers
                drug_name = word.title()
                break

    # Extract dosage pattern (number followed by unit)
    dosage_match = re.search(r'(\d+(?:\.\d+)?)\s*(mg|g|mcg|ml|units?|l|iu)', order_text.lower())
    if dosage_match:
        dose = dosage_match.group(1)
        unit = dosage_match.group(2)

    # Extract route patterns
    route_patterns = [
        r'\b(oral|po|by mouth|intravenous|iv|im|subcutaneous|sq|topical|otic|ophthalmic|rectal|vaginal|nasal|inhaled)\b'
    ]
    for pattern in route_patterns:
        match = re.search(pattern, order_text.lower())
        if match:
            route = match.group(1)
            break

    # Extract frequency patterns
    frequency_patterns = [
        r'\b(q\d+h?|bid|tid|qid|daily|qd|qam|qhs|prn|every\s+\d+\s+(?:hour|day|week)s?)\b',
        r'\b(\d+)\s*(mcg|mg|g)/kg/min\b',
        r'\b(\d+(?:\.\d+)?)\s*(mcg|mg|g)/hr\b',
        r'\b(\d+)\s*ml/hr\b'
    ]

    for pattern in frequency_patterns:
        match = re.search(pattern, order_text.lower())
        if match:
            frequency = match.group(0)
            break

    parsed = {
        'drug_name': drug_name,
        'dose': dose,
        'unit': unit,
        'route': route,
        'frequency': frequency
    }
    return parsed

# Removed: setup_medical_nlp() - simplified parsing approach uses OpenAI + regex fallback only
def _removed_setup_medical_nlp():
    """
    Sets up spaCy with medical entity recognition for advanced drug parsing.
    Returns configured nlp pipeline or None if spaCy unavailable.
    """
    if not SPACY_AVAILABLE:
        return None
    
    try:
        # Try to load a medical model if available, otherwise use small English model
        try:
            nlp = spacy.load("en_core_sci_sm")  # Scientific/medical model
            print("✅ Loaded scientific spaCy model: en_core_sci_sm")
        except OSError:
            try:
                nlp = spacy.load("en_core_web_sm") 
                print("✅ Loaded standard spaCy model: en_core_web_sm")
            except OSError:
                # Create blank model as fallback
                nlp = English()
                print("✅ Created blank English spaCy model")
        
        # Add custom medical entity ruler for drug names
        if "entity_ruler" not in nlp.pipe_names:
            ruler = nlp.add_pipe("entity_ruler", before="ner")
        else:
            ruler = nlp.get_pipe("entity_ruler")
        
        # Medical drug patterns - expand this list as needed
        medical_patterns = [
            # Common IV solutions (full names)
            {"label": "DRUG", "pattern": [{"LOWER": "normal"}, {"LOWER": "saline"}]},
            {"label": "DRUG", "pattern": [{"LOWER": "sodium"}, {"LOWER": "chloride"}]},
            {"label": "DRUG", "pattern": [{"LOWER": "lactated"}, {"LOWER": "ringers"}]},
            {"label": "DRUG", "pattern": [{"LOWER": "dextrose"}, {"LOWER": "water"}]},
            {"label": "DRUG", "pattern": [{"LOWER": "potassium"}, {"LOWER": "chloride"}]},
            {"label": "DRUG", "pattern": [{"LOWER": "magnesium"}, {"LOWER": "sulfate"}]},
            {"label": "DRUG", "pattern": [{"LOWER": "calcium"}, {"LOWER": "gluconate"}]},
            
            # Common abbreviations (preserve as-is)
            {"label": "DRUG_ABBREV", "pattern": [{"LOWER": {"IN": ["ns", "nacl"]}}]},
            {"label": "DRUG_ABBREV", "pattern": [{"LOWER": {"IN": ["lr", "rl"]}}]},
            {"label": "DRUG_ABBREV", "pattern": [{"LOWER": {"IN": ["d5w", "d5ns"]}}]},
            {"label": "DRUG_ABBREV", "pattern": [{"LOWER": {"IN": ["kcl", "mgso4"]}}]},
            
            # Common medications (add more as needed)
            {"label": "DRUG", "pattern": [{"LOWER": "aspirin"}]},
            {"label": "DRUG", "pattern": [{"LOWER": "ibuprofen"}]},
            {"label": "DRUG", "pattern": [{"LOWER": "acetaminophen"}]},
            {"label": "DRUG", "pattern": [{"LOWER": "morphine"}]},
            {"label": "DRUG", "pattern": [{"LOWER": "hydromorphone"}]},
            {"label": "DRUG", "pattern": [{"LOWER": "fentanyl"}]},
        ]
        
        ruler.add_patterns(medical_patterns)
        
        return nlp
        
    except Exception as e:
        print(f"⚠️  Failed to setup medical NLP: {e}")
        return None

# Removed: parse_medication_order_spacy() - simplified parsing approach uses OpenAI + regex fallback only
def _removed_parse_medication_order_spacy(order_text: str, medical_nlp=None) -> Dict[str, str]:
    """
    Advanced spaCy-based medication parsing with medical entity recognition.
    Uses NER and custom patterns to extract drug components more accurately.
    """
    if not SPACY_AVAILABLE or medical_nlp is None:
        return parse_medication_order_fallback(order_text)
    
    try:
        doc = medical_nlp(order_text)
        
        drug_name = ""
        dose = ""
        unit = ""
        route = ""
        frequency = ""
        
        # Extract drug name from named entities first
        for ent in doc.ents:
            if ent.label_ in ["DRUG", "DRUG_ABBREV"] and not drug_name:
                if ent.label_ == "DRUG_ABBREV":
                    # Preserve abbreviations as-is (uppercase)
                    drug_name = ent.text.upper()
                else:
                    # Title case for full drug names
                    drug_name = ent.text.title()
                break
        
        # Fallback to token-based analysis if no entities found
        if not drug_name:
            # Look for drug-like tokens (not numbers, not common stop words)
            medical_stopwords = {
                'at', 'over', 'for', 'per', 'hour', 'hours', 'minute', 'minutes', 
                'bolus', 'infusion', 'daily', 'every', 'once', 'twice', 'by', 'mouth',
                'the', 'a', 'an', 'and', 'of', 'in', 'with', 'to'
            }
            
            for token in doc:
                if (not token.is_digit and 
                    not token.like_num and
                    token.lower_ not in medical_stopwords and
                    len(token.text) > 2 and
                    token.pos_ in ["NOUN", "PROPN"]):  # Likely nouns
                    drug_name = token.text.title()
                    break
        
        # Extract dose and unit using regex (more reliable than NLP for numbers)
        dose_pattern = r'(\d+(?:\.\d+)?)\s*(mg|g|mcg|μg|ml|l|units?|iu|mEq|%)'
        dose_match = re.search(dose_pattern, order_text.lower())
        if dose_match:
            dose = dose_match.group(1)
            unit = dose_match.group(2)
        
        # Extract route using both NER and patterns
        route_patterns = [
            r'\b(intravenous|iv|intramuscular|im|subcutaneous|subq|sq|oral|po|by\s+mouth|topical|rectal|vaginal|nasal|inhaled|sublingual|transdermal)\b'
        ]
        for pattern in route_patterns:
            match = re.search(pattern, order_text.lower())
            if match:
                route = match.group(1)
                break
        
        # Extract frequency/rate patterns  
        freq_patterns = [
            r'\b(q\d+h?|bid|tid|qid|daily|qd|qam|qhs|prn|stat|once|twice)\b',
            r'\b(\d+(?:\.\d+)?)\s*(?:ml|mg|mcg|g|units?)/(?:hr|hour|min|minute)\b',
            r'\bover\s+(\d+)\s*(?:hours?|minutes?|min)\b'
        ]
        for pattern in freq_patterns:
            match = re.search(pattern, order_text.lower()) 
            if match:
                frequency = match.group(0)
                break
        
        return {
            'drug_name': drug_name,
            'dose': dose,
            'unit': unit,
            'route': route,
            'frequency': frequency
        }
        
    except Exception as e:
        print(f"⚠️  spaCy parsing failed, using fallback: {e}")
        return parse_medication_order_fallback(order_text)

def reorganize_signatures_by_question_type(study3_results: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Reorganize signatures from enhanced analysis JSON by question type.
    Returns 3 questions (formal/verbal/brief) with 40 answers each.
    """
    results = {
        'formal_question': {'answers': []},
        'verbal_question': {'answers': []},
        'brief_question': {'answers': []},
        'metadata': {}
    }

    # Load enhanced JSON
    enhanced_json_path = 'results/rougerx/rougerx_enhanced_analysis.json'
    try:
        with open(enhanced_json_path, 'r', encoding='utf-8') as f:
            enhanced_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Enhanced JSON file not found: {enhanced_json_path}")
        # Initialize basic metadata even when enhanced JSON is missing
        results['metadata'] = {
            'formal_signatures': 0,
            'verbal_signatures': 0,
            'brief_signatures': 0,
            'total_signatures': 0,
            'enhanced_json_available': False
        }
        return results

    # Group by question type
    # IMPORTANT: Use actual CSV column order to match parse_csv_data logic exactly
    # This accounts for missing questions (e.g., Q63) that would break qnum-based calculation
    for question_id, question_data in enhanced_data.items():
        if not question_id.startswith('Q'):
            continue

        # Get actual index from CSV column order (handles missing questions)
        index = get_question_index(question_id)
        if index is None:
            print(f"⚠️  Could not determine index for {question_id}, skipping")
            continue
        
        # Get question type from index
        question_type = get_question_type_from_index(index)
        question_key = f'{question_type}_question'
        med_index = index // 3

        # Extract signature data
        signature_data = {
            'question_id': question_id,
            'medication_index': med_index,
            'communication_type': question_key.replace('_question', ''),
            'prompt': question_data.get('data', {}).get('prompt', ''),
            'responses': question_data.get('data', {}).get('responses', {}),
            'common_words_all': question_data.get('data', {}).get('common_words_all', ''),
            'study_3_components': question_data.get('data', {}).get('study_3_components', {})
        }

        results[question_key]['answers'].append(signature_data)

    # Sort answers by medication index within each question type
    for question_type in ['formal_question', 'verbal_question', 'brief_question']:
        results[question_type]['answers'].sort(key=lambda x: x['medication_index'])

    # Add metadata
    results['metadata'] = {
        'total_signatures_reorganized': sum(len(results[q]['answers']) for q in ['formal_question', 'verbal_question', 'brief_question']),
        'formal_signatures': len(results['formal_question']['answers']),
        'verbal_signatures': len(results['verbal_question']['answers']),
        'brief_signatures': len(results['brief_question']['answers']),
        'source_file': enhanced_json_path,
        'reorganization_type': 'by_question_type'
    }

    # Save reorganized signatures
    output_path = 'results/rougerx/rougerx_signatures_by_question_type.json'
    os.makedirs('results/rougerx', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Add advanced analysis results from study3_results if provided
    if study3_results:
        results['advanced_analysis'] = {
            'component_overlap_analysis': study3_results.get('component_overlap_analysis', {}),
            'overlap_analysis': study3_results.get('overlap_analysis', {}),
            'positional_precision_analysis': study3_results.get('positional_precision_analysis', {}),
            'similarity_analysis': study3_results.get('similarity_analysis', {})
        }
        print("✅ Advanced analysis included in signatures")
    else:
        print("⚠️ No study3_results provided - advanced analysis not included")

    print(f"✅ Reorganized signatures saved to: {output_path}")
    print(f"   Formal question: {len(results['formal_question']['answers'])} answers")
    print(f"   Verbal question: {len(results['verbal_question']['answers'])} answers")
    print(f"   Brief question: {len(results['brief_question']['answers'])} answers")

    return results

def exploratory_study_3_enhanced_json_parsing() -> Dict[str, Any]:
    """
    EXPLORATORY STUDY 3: Comprehensive Advanced Medication Communication Analysis

    Performs complete analysis pipeline including:

    1. Single medication response parsing with LLM-powered component extraction
    2. Position-aware component ordering within medication text
    3. Component divergence analysis across respondent answers
    4. Component overlap analysis (exact vs partial matches)
    5. Positional precision analysis (expected ordering validation)
    6. Cross-respondent similarity analysis by communication style
    7. Enhanced omission analysis with detailed quantification

    Reads rougerx_enhanced_analysis.json, processes each response with advanced
    parsing techniques, and generates comprehensive statistical analysis of
    medication communication patterns across healthcare professionals.

    Returns comprehensive analysis results including:
    - parsing_method: LLM/regex parsing approach used
    - questions_processed: Number of questions analyzed
    - total_responses_parsed: Total response count
    - successful_parses: Successfully parsed responses
    - divergence_analysis: Component variation statistics
    - omission_analysis: Missing component quantification
    - component_overlap_analysis: Exact vs partial match statistics
    - positional_precision_analysis: Ordering pattern validation
    - similarity_analysis: Cross-respondent similarity metrics
    """
    # Determine parsing method
    if OPENAI_AVAILABLE:
        if PYDANTIC_AVAILABLE:
            parsing_method = 'llm_pydantic_structured'
        else:
            parsing_method = 'llm_simple_json'
    else:
        parsing_method = 'regex_fallback'

    results = {
        'parsing_method': parsing_method,
        'questions_processed': 0,
        'total_responses_parsed': 0,
        'successful_parses': 0,
        'divergence_analysis': {},
        'component_statistics': {},
        'enhanced_json_updated': False
    }

    # Load enhanced JSON file if it exists, otherwise create basic structure
    enhanced_json_path = 'results/rougerx/rougerx_enhanced_analysis.json'
    enhanced_data = {}

    try:
        with open(enhanced_json_path, 'r', encoding='utf-8') as f:
            enhanced_data = json.load(f)
        print(f"📖 Loaded {len(enhanced_data)} questions from enhanced JSON")
    except FileNotFoundError:
        print(f"ℹ️  Enhanced JSON file not found, will create new analysis")
        # Will create enhanced data from scratch below
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing enhanced JSON: {e}")
        return results

    # If no enhanced data exists, create it from the original CSV
    if not enhanced_data:
        print("ℹ️  No existing enhanced analysis found - creating from original CSV data")
        try:
            # Load original CSV data
            csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'rougerx.csv'))
            df = pd.read_csv(csv_path)
            
            # Get Q3-Q122 columns (120 questions)
            all_cols = list(df.columns)
            q3_index = all_cols.index('Q3')
            q122_index = all_cols.index('Q122')
            question_cols = all_cols[q3_index:q122_index+1]
            question_cols = [col for col in question_cols if col.startswith('Q')]
            
            print(f"📊 Creating enhanced data from {len(question_cols)} questions")
            
            # Create enhanced data structure
            enhanced_data = {}
            for col in question_cols:
                if col in df.columns:
                    # Extract responses from rows 1-4 (row 0 is prompts)
                    responses = {}
                    for i in range(1, min(5, len(df))):  # Rows 1-4
                        response = str(df[col].iloc[i]).strip()
                        if response and response != 'nan':
                            responses[f'ID_{i}'] = response
                    
                    # Extract prompt from row 0
                    prompt = str(df[col].iloc[0]).strip() if len(df) > 0 else ""

                    # Only include questions with at least 2 responses
                    if len(responses) >= 2:
                        enhanced_data[col] = {
                            'data': {
                                'prompt': prompt,
                                'responses': responses
                            }
                        }
            
            print(f"✅ Created enhanced data structure with {len(enhanced_data)} questions")
            
        except Exception as e:
            print(f"❌ Error creating enhanced data from CSV: {e}")
            return results

    # Process each question
    component_divergence_scores = defaultdict(list)

    for question_id, question_data in enhanced_data.items():
        if 'data' not in question_data or 'responses' not in question_data['data']:
            continue

        responses = question_data['data']['responses']
        response_texts = [responses.get(f'ID_{i}', '') for i in range(1, 5)]
        response_texts = [r.strip() for r in response_texts if r.strip()]

        if len(response_texts) < 2:
            continue

        # Parse each response (single medication per response) with source tracking
        # Use parallel processing for OpenAI API calls
        all_parsed_drugs = []

        def parse_single_response(response_item):
            """Parse a single response (used for parallel processing)"""
            response_id, response = response_item
            if response.strip():  # Only process non-empty responses
                parsed_drug = parse_medication_from_response(response, response_id)
                return parsed_drug
            return None

        # Prepare response items for parallel processing
        response_items = [(response_id, response) for response_id, response in responses.items()]

        # Use ThreadPoolExecutor for parallel processing of OpenAI API calls
        print(f"🔄 Processing {len(response_items)} responses with {MAX_WORKERS} parallel workers...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all parsing tasks
            future_to_response = {
                executor.submit(parse_single_response, response_item): response_item
                for response_item in response_items
            }

            # Collect results as they complete (thread-safe)
            for future in as_completed(future_to_response):
                parsed_drug = future.result()
                with results_lock:
                    results['total_responses_parsed'] += 1
                    if parsed_drug:  # If drug was successfully parsed from this response
                        all_parsed_drugs.append(parsed_drug)
                        results['successful_parses'] += 1

        # Calculate divergence, omission, and overlap percentages for this question
        components = ['drug_name', 'dose', 'unit', 'route', 'frequency']
        question_divergence = {}

        total_drugs_parsed = len(all_parsed_drugs)

        for component in components:
            # Get all values for this component (including empty strings)
            all_values = [drug.get(component, '') for drug in all_parsed_drugs]
            non_empty_values = [v for v in all_values if v.strip()]
            unique_values = len(set(non_empty_values))
            total_with_values = len(non_empty_values)
            total_missing = total_drugs_parsed - total_with_values

            # Calculate omission percentage
            omission_percentage = (total_missing / total_drugs_parsed * 100) if total_drugs_parsed > 0 else 0

            # Calculate overlap statistics for this component
            overlap_stats = analyze_component_overlap(all_parsed_drugs, component)

            # Consolidate all statistics into the component_divergence section
            question_divergence[component] = {
                'unique_values': unique_values,
                'total_parsed': total_with_values,
                'total_drugs': total_drugs_parsed,
                'divergence_score': unique_values / total_with_values if total_with_values > 0 else 0,
                'omitted_count': total_missing,
                'omission_percentage': omission_percentage,
                'exact_matches': overlap_stats['exact_matches'],
                'partial_overlaps': overlap_stats['partial_overlaps'],
                'total_comparisons': overlap_stats['total_comparisons'],
                'exact_percentage': overlap_stats['exact_percentage'],
                'overlap_percentage': overlap_stats['overlap_percentage'],
                'values': non_empty_values
            }

            # Collect for overall statistics
            divergence_score = unique_values / total_with_values if total_with_values > 0 else 0
            component_divergence_scores[component].append(divergence_score)

        # Add study_3_components to the question data
        question_data['data']['study_3_components'] = {
            'parsed_drugs': all_parsed_drugs,
            'component_divergence': question_divergence,
            'parsing_method': parsing_method,
            'total_drugs_parsed': total_drugs_parsed
        }

        results['questions_processed'] += 1

    # Calculate overall divergence statistics
    if component_divergence_scores:
        component_avg_divergence = {}
        for component, scores in component_divergence_scores.items():
            if scores:
                component_avg_divergence[component] = sum(scores) / len(scores)
            else:
                component_avg_divergence[component] = 0.0

        # Sort by divergence (highest first)
        sorted_components = sorted(component_avg_divergence.items(), key=lambda x: x[1], reverse=True)

        results['divergence_analysis'] = {
            'component_divergence_ranking': sorted_components,
            'most_divergent_component': sorted_components[0][0] if sorted_components else None,
            'least_divergent_component': sorted_components[-1][0] if sorted_components else None
        }

        # Calculate overall omission statistics
        component_avg_omission = {}
        all_omission_stats = defaultdict(list)

        # Calculate overall overlap statistics
        component_avg_overlap = {}
        all_overlap_stats = defaultdict(list)

        # Re-load data to collect omission and overlap stats (since we need to aggregate across questions)
        try:
            with open(enhanced_json_path, 'r', encoding='utf-8') as f:
                enhanced_data_for_stats = json.load(f)
        except:
            enhanced_data_for_stats = enhanced_data

        for question_id, question_data in enhanced_data_for_stats.items():
            if 'data' in question_data and 'study_3_components' in question_data['data']:
                component_divergence = question_data['data']['study_3_components'].get('component_divergence', {})
                for component, stats in component_divergence.items():
                    if 'omission_percentage' in stats:
                        all_omission_stats[component].append(stats['omission_percentage'])
                    if 'overlap_percentage' in stats:
                        all_overlap_stats[component].append(stats['overlap_percentage'])

        for component in components:
            if component in all_omission_stats and all_omission_stats[component]:
                component_avg_omission[component] = sum(all_omission_stats[component]) / len(all_omission_stats[component])
            else:
                component_avg_omission[component] = 0.0

            if component in all_overlap_stats and all_overlap_stats[component]:
                component_avg_overlap[component] = sum(all_overlap_stats[component]) / len(all_overlap_stats[component])
            else:
                component_avg_overlap[component] = 0.0

        # Sort by omission percentage (highest first)
        sorted_omission = sorted(component_avg_omission.items(), key=lambda x: x[1], reverse=True)

        # Sort by overlap percentage (highest first)
        sorted_overlap = sorted(component_avg_overlap.items(), key=lambda x: x[1], reverse=True)

        results['omission_analysis'] = {
            'component_omission_ranking': sorted_omission,
            'most_omitted_component': sorted_omission[0][0] if sorted_omission else None,
            'least_omitted_component': sorted_omission[-1][0] if sorted_omission else None
        }

        results['overlap_analysis'] = {
            'component_overlap_ranking': sorted_overlap,
            'most_overlapping_component': sorted_overlap[0][0] if sorted_overlap else None,
            'least_overlapping_component': sorted_overlap[-1][0] if sorted_overlap else None
        }

        results['component_statistics'] = {
            'average_divergence_by_component': component_avg_divergence,
            'total_questions_with_divergence_data': len([s for s in component_divergence_scores.values() if s])
        }

    # Calculate new advanced analysis features
    print("🔍 Calculating advanced analysis features...")

    # 1. Component overlap analysis (exact vs partial matches)
    component_overlap_analysis = {}
    components = ['drug_name', 'dose', 'unit', 'route', 'frequency']

    for component in components:
        # Collect all parsed components across all questions
        all_component_values = []
        for question_id, question_data in enhanced_data.items():
            if 'data' in question_data and 'study_3_components' in question_data['data']:
                parsed_drugs = question_data['data']['study_3_components'].get('parsed_drugs', [])
                all_component_values.extend(parsed_drugs)

        if all_component_values:
            overlap_stats = analyze_component_overlap(all_component_values, component)
            component_overlap_analysis[component] = overlap_stats

    results['component_overlap_analysis'] = component_overlap_analysis

    # 2. Positional precision analysis
    all_parsed_components = []
    for question_id, question_data in enhanced_data.items():
        if 'data' in question_data and 'study_3_components' in question_data['data']:
            parsed_drugs = question_data['data']['study_3_components'].get('parsed_drugs', [])
            all_parsed_components.extend(parsed_drugs)

    positional_analysis = analyze_positional_precision(all_parsed_components)
    results['positional_precision_analysis'] = positional_analysis

    # 3. Drug similarity analysis across respondents and styles
    similarity_analysis = calculate_drug_similarity_analysis(enhanced_data)
    results['similarity_analysis'] = similarity_analysis

    print("✅ Advanced analysis features calculated")

    # Save updated enhanced JSON
    try:
        with open(enhanced_json_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
        results['enhanced_json_updated'] = True
        print(f"✅ Updated enhanced JSON with Study 3 components: {enhanced_json_path}")
    except Exception as e:
        print(f"❌ Error saving updated enhanced JSON: {e}")
        import traceback
        traceback.print_exc()

    return results

def generate_common_words_csv(exploratory_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Generate a CSV summary of common words for each question column.

    Creates a tabular format showing common words analysis for each question,
    making it easy to review patterns across all Q3-Q122 columns.
    """
    summary_data = []

    for col, data in exploratory_results.items():
        # Get question type using actual CSV index (handles missing questions)
        index = get_question_index(col)
        if index is None:
            continue
        comm_type = get_question_type_from_index(index)

        # Calculate medication group (every 3 columns = 1 medication)
        med_group = (index // 3) + 1  # indices 0,1,2 = med 1, etc.

        row = {
            'question_column': col,
            'communication_type': comm_type.upper(),
            'medication_group': med_group,
            'total_responses': data['question_info']['total_responses'],
            'common_words_all': ', '.join(data['common_analysis']['common_words_all']),
            'num_common_words_all': len(data['common_analysis']['common_words_all']),
            'majority_words': ', '.join(data['common_analysis']['majority_words'].keys()),
            'num_majority_words': len(data['common_analysis']['majority_words']),
            'avg_jaccard_similarity': round(data['similarity_analysis']['avg_jaccard_similarity'], 3),
            'unique_words_total': data['unique_analysis']['total_unique_across_all'],
            'top_3_words': ', '.join(list(data['word_frequencies']['top_words'].keys())[:3])
        }

        summary_data.append(row)

    # Create DataFrame and sort by question column
    df_summary = pd.DataFrame(summary_data)
    df_summary = df_summary.sort_values('question_column')

    return df_summary

def generate_enhanced_csv_with_analysis(original_df: pd.DataFrame, exploratory_results: Dict[str, Dict]) -> str:
    """
    Generate an enhanced CSV with analysis results in the user's requested format.

    Creates a row-oriented format where each question gets a section with:
    - Question header
    - Prompt
    - Respondent answers (ID_1, ID_2, ID_3, ID_4)
    - Common words analysis
    """
    results_dir = 'results/rougerx'
    os.makedirs(results_dir, exist_ok=True)
    enhanced_csv_file = os.path.join(results_dir, 'rougerx_with_analysis.csv')

    with open(enhanced_csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Get the Q columns we analyzed, sorted
        q_columns = sorted([col for col in exploratory_results.keys() if col.startswith('Q')])

        # For each question, create a complete section
        for col in q_columns:
            if col not in exploratory_results:
                continue

            data = exploratory_results[col]

            # Write question header
            writer.writerow([col])

            # Write column headers for this section
            writer.writerow(['Participant', col])

            # Write the prompt row (from original data, row 0)
            prompt_text = str(original_df[col].iloc[0]) if col in original_df.columns else ""
            writer.writerow(['Prompt', prompt_text])

            # Write respondent answers (from original data, rows 1-4)
            for i in range(4):  # 4 respondents
                respondent_id = f'ID_{i+1}'
                answer_text = str(original_df[col].iloc[i+1]) if col in original_df.columns and i+1 < len(original_df) else ""
                writer.writerow([respondent_id, answer_text])

            # Write common words analysis
            common_words = ', '.join(data['common_analysis']['common_words_all'])
            writer.writerow(['common_words_all', common_words])

            # Add blank row to separate sections
            writer.writerow([])

    print(f"✓ Enhanced CSV created with {len(q_columns)} question sections, each containing responses and common words analysis")

    return enhanced_csv_file

def convert_enhanced_csv_to_json() -> str:
    """
    Convert the enhanced CSV (with question sections) to a structured JSON format.

    Transforms the sectioned CSV format into a clean JSON structure for easier programmatic access.
    Preserves existing study_3_components if they exist in the current enhanced JSON.
    """
    json_output = {}
    current_question = None
    current_section = {}

    results_dir = 'results/rougerx'
    enhanced_csv_path = os.path.join(results_dir, 'rougerx_with_analysis.csv')
    enhanced_json_path = os.path.join(results_dir, 'rougerx_enhanced_analysis.json')

    # Load existing study_3_components if they exist
    existing_study3_data = {}
    if os.path.exists(enhanced_json_path):
        try:
            with open(enhanced_json_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                for qid, qdata in existing_data.items():
                    if isinstance(qdata, dict) and 'data' in qdata and 'study_3_components' in qdata['data']:
                        existing_study3_data[qid] = qdata['data']['study_3_components']
                        print(f"📋 Preserving study_3_components for {qid}")
        except Exception as e:
            print(f"⚠️  Could not load existing study_3_components: {e}")

    with open(enhanced_csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) == 0:
                continue

            # Check if this is a question header (single cell starting with Q)
            if len(row) == 1 and row[0].startswith('Q'):
                # Save previous question if exists
                if current_question and current_section:
                    # Add back preserved study_3_components if they exist
                    if current_question in existing_study3_data:
                        current_section['data']['study_3_components'] = existing_study3_data[current_question]
                        print(f"✅ Restored study_3_components for {current_question}")
                    json_output[current_question] = current_section

                # Start new question
                current_question = row[0]
                current_section = {
                    'question_id': current_question,
                    'data': {}
                }
                continue

            # Check if this is a data row for current question
            if current_question and len(row) >= 2:
                key = row[0].strip()
                value = row[1].strip() if len(row) > 1 else ""

                # Handle special cases
                if key == 'Participant':
                    continue  # Skip header row
                elif key == 'Prompt':
                    current_section['data']['prompt'] = value
                elif key.startswith('ID_'):
                    if 'responses' not in current_section['data']:
                        current_section['data']['responses'] = {}
                    respondent_id = key
                    current_section['data']['responses'][respondent_id] = value
                elif key == 'common_words_all':
                    current_section['data']['common_words_all'] = value

        # Save the last question
        if current_question and current_section:
            # Add back preserved study_3_components if they exist
            if current_question in existing_study3_data:
                current_section['data']['study_3_components'] = existing_study3_data[current_question]
                print(f"✅ Restored study_3_components for {current_question}")
            json_output[current_question] = current_section

    # Save as JSON file in results directory
    results_dir = 'results/rougerx'
    os.makedirs(results_dir, exist_ok=True)
    json_file = os.path.join(results_dir, 'rougerx_enhanced_analysis.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)

    print(f"✓ Converted enhanced CSV to structured JSON with {len(json_output)} question sections")

    return json_file

def analyze_word_overlap(df: pd.DataFrame, medications: List[str], response_cols: Dict[str, List[str]]) -> Dict[str, Dict]:
    """
    Step 2: Analyze word overlap organized by question type (3 questions with 40 answers each)

    Performs set-based study to find common words for each answer (medication) within each question type.
    Returns 3 main sections (formal/verbal/brief) with 40 answers each (one per medication).

    Uses simple string split by default, with NLTK as option.
    """
    results = {}

    # Analyze by communication type (3 questions)
    for comm_type, cols in response_cols.items():
        comm_results = {
            'overall_stats': {},
            'answers': {},  # 40 answers (one per medication)
            'common_words_across_medications': set(),
            'word_usage_patterns': defaultdict(int)
        }

        all_words_across_meds = []
        medication_word_sets = []

        # Analyze each medication for this communication type
        for i, med in enumerate(medications):
            if i >= len(cols):
                continue

            col = cols[i]

            # Get responses for this specific medication and communication type (skip row 0 which is prompts)
            responses = []
            for row_idx in range(1, len(df)):  # Skip row 0 (prompts)
                response = str(df[col].iloc[row_idx]).strip()
                if response and response.lower() not in ['nan', '']:
                    responses.append(response)

            if not responses:
                continue

            # Tokenize all responses for this medication using NLTK for precision
            all_response_words = []
            respondent_word_sets = []

            for response in responses:
                words = tokenize_words(response, method='nltk')  # Use NLTK tokenizer for precision
                if words:  # Only add if we got words
                    all_response_words.extend(words)
                    respondent_word_sets.append(set(words))

            # Calculate overlap for this medication
            med_result = {
                'medication': med,
                'num_respondents': len(responses),
                'total_words': len(all_response_words),
                'unique_words': len(set(all_response_words)),
                'words_per_respondent': len(all_response_words) / len(responses) if responses else 0,
                'respondent_word_sets': respondent_word_sets
            }

            # Find common words across respondents for this medication
            if respondent_word_sets:
                if len(respondent_word_sets) > 1:
                    common_words = set.intersection(*respondent_word_sets)
                else:
                    common_words = respondent_word_sets[0] if respondent_word_sets else set()

                med_result['common_words'] = list(common_words)
                med_result['num_common_words'] = len(common_words)

                # Find words that appear in most responses
                word_counts = Counter()
                for word_set in respondent_word_sets:
                    for word in word_set:
                        word_counts[word] += 1

                # Words that appear in at least 50% of responses
                threshold = max(1, len(respondent_word_sets) // 2)
                frequent_words = {word: count for word, count in word_counts.items()
                                if count >= threshold}
                med_result['frequent_words'] = frequent_words

                # Words unique to this medication
                all_words_in_med = set(all_response_words)
                med_result['unique_to_medication'] = list(all_words_in_med)

            comm_results['answers'][med] = med_result
            all_words_across_meds.extend(all_response_words)
            if respondent_word_sets:
                medication_word_sets.append(set(all_response_words))

        # Overall stats for this communication type
        comm_results['overall_stats'] = {
            'total_answers_analyzed': len(comm_results['answers']),
            'total_words_across_all': len(all_words_across_meds),
            'unique_words_across_all': len(set(all_words_across_meds)),
            'avg_words_per_answer': len(all_words_across_meds) / len(comm_results['answers']) if comm_results['answers'] else 0
        }

        # Find words common across medications (if any)
        if medication_word_sets and len(medication_word_sets) > 1:
            common_across_meds = set.intersection(*medication_word_sets)
            comm_results['common_words_across_medications'] = list(common_across_meds)

        # Word frequency across all medications for this type
        overall_word_freq = Counter(all_words_across_meds)
        comm_results['top_20_words_overall'] = dict(overall_word_freq.most_common(20))

        results[comm_type] = comm_results

    return results

def analyze_exact_answers(df: pd.DataFrame, medications: List[str], response_cols: Dict[str, List[str]]) -> Dict:
    """
    Step 3: Analyze exact answers organized by question type (3 questions with 40 answers each)

    For each question type, analyzes how responses vary across medications.
    Returns 3 main sections showing answer patterns within each question type.
    """
    results = {
        'formal_question': {'answers': []},
        'verbal_question': {'answers': []},
        'brief_question': {'answers': []},
        'overall_stats': {}
    }

    # Analyze each question type (3 questions)
    for comm_type in ['formal', 'verbal', 'brief']:
        question_key = f'{comm_type}_question'
        cols = response_cols[comm_type]

        # For each medication in this question type (40 answers)
        for i, col in enumerate(cols):
            medication = medications[i] if i < len(medications) else f"Medication_{i+1}"

            # Get responses for this question (rows 1-4, skip row 0 with prompts)
            responses = []
            for row_idx in range(1, min(5, len(df))):  # Rows 1-4 (4 respondents)
                response = str(df[col].iloc[row_idx]).strip()
                if response and response.lower() not in ['nan', '']:
                    responses.append(response)

            if len(responses) < 4:  # Need all 4 respondents
                continue

            # Check if all responses are identical
            all_identical = all(resp == responses[0] for resp in responses)

            answer_data = {
                'medication': medication,
                'question_column': col,
                'communication_type': comm_type,
                'responses': responses,
                'all_identical': all_identical,
                'unique_responses': len(set(responses)),
                'response_length_avg': sum(len(r) for r in responses) / len(responses),
                'respondent_count': len(responses)
            }

            results[question_key]['answers'].append(answer_data)

    # Calculate overall statistics
    total_answers = sum(len(results[q]['answers']) for q in ['formal_question', 'verbal_question', 'brief_question'])
    identical_answers = sum(sum(1 for a in results[q]['answers'] if a['all_identical']) for q in ['formal_question', 'verbal_question', 'brief_question'])
    identical_percentage = identical_answers / total_answers * 100 if total_answers > 0 else 0

    results['overall_stats'] = {
        'total_answers_analyzed': total_answers,
        'identical_answers_count': identical_answers,
        'divergent_answers_count': total_answers - identical_answers,
        'identical_percentage': identical_percentage,
        'divergent_percentage': 100 - identical_percentage
    }

    return results

def parse_drug_info_simple(responses: List[str]) -> List[Dict]:
    """
    Step 4: Simple parser to extract drug names, dosages, duration, units from answers

    Uses regex patterns to extract common medication components.
    """
    parsed_info = []

    for response in responses:
        if not response or str(response).strip() == '':
            parsed_info.append({})
            continue

        response_str = str(response).lower()

        # Extract drug name (first word, typically)
        drug_name = None
        words = response_str.split()
        if words:
            drug_name = words[0].title()  # Capitalize first letter

        # Extract dosage patterns (numbers followed by units)
        dosage_patterns = [
            r'(\d+(?:\.\d+)?)\s*(mg|g|mcg|ml|units?|l|iu)',  # e.g., 81 mg, 2g, 5mcg
            r'(\d+(?:\.\d+)?)\s*(mg|g|mcg|ml|units?|l|iu)/(\d+(?:\.\d+)?)\s*(mg|g|mcg|ml|units?|l|iu)',  # ratios like 800-160 mg
        ]

        dosages = []
        units = []

        for pattern in dosage_patterns:
            matches = re.findall(pattern, response_str)
            for match in matches:
                if len(match) == 2:  # Simple dosage
                    dosages.append(match[0])
                    units.append(match[1])
                elif len(match) == 4:  # Ratio dosage
                    dosages.append(f"{match[0]}-{match[2]}")
                    units.append(f"{match[1]}-{match[3]}")

        # Extract frequency/duration patterns
        freq_patterns = [
            r'\b(q\d+h?|bid|tid|qid|daily|qd|qam|qhs|prn|q\d+|every\s+\d+\s+(?:hour|day|week)s?|m\w+f)\b',
            r'\b(\d+)\s*(mcg|mg|g)/kg/min\b',  # infusion rates
            r'\b(\d+(?:\.\d+)?)\s*(mcg|mg|g)/hr\b',  # hourly rates
            r'\b(\d+)\s*ml/hr\b',  # IV rates
        ]

        frequencies = []
        for pattern in freq_patterns:
            matches = re.findall(pattern, response_str)
            frequencies.extend(matches)

        parsed_info.append({
            'drug_name': drug_name,
            'dosages': dosages,
            'units': units,
            'frequencies': frequencies,
            'raw_response': response
        })

    return parsed_info

def analyze_divergence_patterns(df: pd.DataFrame, medications: List[str], response_cols: Dict[str, List[str]]) -> Dict:
    """
    Step 5: Analyze divergence patterns organized by question type (3 questions with 40 answers each)

    For each question type, analyzes how parsed drug information varies across medications.
    Returns 3 main sections showing divergence patterns within each question type.
    """
    results = {
        'formal_question': {'answers': []},
        'verbal_question': {'answers': []},
        'brief_question': {'answers': []},
        'overall_divergence': {}
    }

    # Analyze each question type (3 questions)
    for comm_type in ['formal', 'verbal', 'brief']:
        question_key = f'{comm_type}_question'
        cols = response_cols[comm_type]

        # For each medication in this question type (40 answers)
        for i, col in enumerate(cols):
            medication = medications[i] if i < len(medications) else f"Medication_{i+1}"

            # Get responses for this medication in this question type
            responses = df[col].dropna().astype(str).tolist()
            parsed = parse_drug_info_simple(responses)

            # Analyze parsed components for divergence
            components = ['drug_name', 'dosages', 'units', 'frequencies']
            component_analysis = {}

            for component in components:
                values = []
                for p in parsed:
                    val = p.get(component, [])
                    if isinstance(val, list):
                        values.extend(val)
                    else:
                        values.append(val)

                # Remove empty values and get unique values
                unique_values = set(v for v in values if v)
                component_analysis[component] = {
                    'values': list(unique_values),
                    'unique_count': len(unique_values),
                    'total_parsed': len([v for v in values if v])
                }

            answer_data = {
                'medication': medication,
                'question_column': col,
                'communication_type': comm_type,
                'parsed_responses': parsed,
                'component_analysis': component_analysis,
                'total_responses': len(responses),
                'parsed_components': len([p for p in parsed if any(p.values())])
            }

            results[question_key]['answers'].append(answer_data)

    # Calculate overall divergence statistics
    all_answers = []
    for comm_type in ['formal', 'verbal', 'brief']:
        all_answers.extend(results[f'{comm_type}_question']['answers'])

    component_divergence = {}
    for component in ['drug_name', 'dosages', 'units', 'frequencies']:
        unique_counts = [a['component_analysis'][component]['unique_count'] for a in all_answers]
        avg_unique = sum(unique_counts) / len(unique_counts) if unique_counts else 0
        component_divergence[component] = {
            'avg_unique_values_per_answer': avg_unique,
            'max_unique_values': max(unique_counts) if unique_counts else 0,
            'min_unique_values': min(unique_counts) if unique_counts else 0
        }

    results['overall_divergence'] = {
        'component_divergence_stats': component_divergence,
        'total_answers_analyzed': len(all_answers),
        'total_medications': len(medications)
    }

    return results

def setup_analysis_environment() -> Tuple[str, str]:
    """Initialize analysis environment and show configuration."""
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'rougerx.csv')
    
    # Using OpenAI LLM for medication parsing with regex fallback
    if OPENAI_AVAILABLE and openai_client:
        print("🔬 Using OpenAI LLM for medication parsing")
    else:
        print("⚠️  OpenAI not available, using regex fallback")

    # Show results directory location
    results_dir = 'results/rougerx'
    results_dir_abs = os.path.abspath(results_dir)
    print(f"📁 Results will be saved to: {results_dir_abs}")
    
    return csv_path, results_dir

def load_and_parse_data(csv_path: str) -> Tuple[pd.DataFrame, List[str], Dict[str, List[str]], int]:
    """Load and parse CSV data for analysis."""
    print("\nStep 1: Parsing CSV data (Q3-Q122 questions and answers only)...")
    df, medications, response_cols, num_respondents = parse_csv_data(csv_path)
    print(f"✓ Confirmed analysis scope: Q3-Q122 ({len(response_cols['formal'])} formal + {len(response_cols['verbal'])} verbal + {len(response_cols['brief'])} brief responses)")
    print(f"✓ Found {len(medications)} medications and {num_respondents} respondents")
    return df, medications, response_cols, num_respondents

def run_exploratory_studies(df: pd.DataFrame, medications: List[str], response_cols: Dict[str, List[str]]) -> Tuple[Dict, Dict, Dict]:
    """Execute the three main exploratory studies."""
    print("\n" + "="*60)
    print("🚀 ANALYSIS PIPELINE STARTED")
    print("="*60)

    print("\n📝 Step 1: CSV Data Processing")
    print("-" * 30)
    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"✓ Identified {len(medications)} medications and {df.iloc[1:].dropna(how='all').shape[0]} respondents")
    print(f"✓ Analysis scope: Q3-Q122 ({len(response_cols['formal'])} formal, {len(response_cols['verbal'])} verbal, {len(response_cols['brief'])} brief questions)")

    # Study 1: Word Overlap Analysis
    print("\n🔍 Step 2: Exploratory Study 1 - Word Overlap Analysis")
    print("-" * 50)
    all_q_columns = [col for col in df.columns if col.startswith('Q')]
    q3_to_q122_cols = all_q_columns[:120]  # First 120 Q columns (Q3-Q122)
    exploratory_results = exploratory_study_1_word_overlap(df, focus_columns=q3_to_q122_cols)
    print(f"✓ Completed word overlap analysis for {len(exploratory_results)} questions")

    # Study 2: Exact Answer Analysis
    print("\n🎯 Step 3: Exploratory Study 2 - Exact Answer Consistency")
    print("-" * 52)
    study2_results = exploratory_study_2_exact_answers(df, medications, response_cols)
    stats = study2_results['overall_stats']
    identical_percentage = (stats['identical_answers_count'] / stats['total_answers_analyzed'] * 100) if stats['total_answers_analyzed'] > 0 else 0
    print(f"✓ Analyzed {stats['total_answers_analyzed']} answers across 3 communication styles")
    print(f"✓ Found {identical_percentage:.1f}% identical responses")

    # Study 3: Advanced Medication Analysis
    print("\n⚕️  Step 4: Exploratory Study 3 - Advanced Medication Analysis")
    print("-" * 57)
    study3_results = exploratory_study_3_enhanced_json_parsing()
    print(f"✓ Processed {study3_results['questions_processed']} questions using {study3_results['parsing_method']} parsing")
    print("✓ Generated: component ordering, overlap analysis, positional precision, similarity metrics")
    
    return exploratory_results, study2_results, study3_results

def run_advanced_analysis(study3_results: Dict, df: pd.DataFrame, medications: List[str], response_cols: Dict[str, List[str]]) -> Tuple[Dict, Dict]:
    """Execute signature reorganization and pattern analysis."""
    print("\n📊 Step 5: Signature Reorganization & Advanced Metrics")
    print("-" * 53)

    # Only attempt reorganization if study3 processed some data
    if study3_results['questions_processed'] > 0:
        signature_results = reorganize_signatures_by_question_type(study3_results)
        print(f"✓ Reorganized data by communication style with advanced analytics")
        print(f"✓ Created 3 question types with {signature_results['metadata']['formal_signatures']} answers each")
    else:
        print("⚠️  Skipping signature reorganization - no enhanced data available")
        print("💡 Study 3 requires existing enhanced analysis data to process")
        # Create minimal signature results for the rest of the pipeline
        signature_results = {
            'formal_question': {'answers': []},
            'verbal_question': {'answers': []},
            'brief_question': {'answers': []},
            'metadata': {
                'formal_signatures': 0,
                'verbal_signatures': 0,
                'brief_signatures': 0,
                'enhanced_json_available': False
            }
        }

    print("\n📈 Step 6: Cross-Style Pattern Analysis")
    print("-" * 37)
    word_overlap_results = analyze_word_overlap(df, medications, response_cols)
    exact_answer_results = analyze_exact_answers(df, medications, response_cols)
    divergence_results = analyze_divergence_patterns(df, medications, response_cols)
    print("✓ Completed comprehensive pattern analysis across all communication styles")
    
    pattern_results = {
        'word_overlap': word_overlap_results,
        'exact_answers': exact_answer_results,
        'divergence': divergence_results
    }
    
    return signature_results, pattern_results

def generate_final_outputs(exploratory_results: Dict, study2_results: Dict, study3_results: Dict, 
                          signature_results: Dict, pattern_results: Dict, medications: List[str], 
                          num_respondents: int, results_dir: str) -> None:
    """Generate final outputs and summary reports."""
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)

    # Generate common words summary CSV
    common_words_csv = generate_common_words_csv(exploratory_results)
    csv_file = os.path.join(results_dir, 'common_words_summary.csv')
    common_words_csv.to_csv(csv_file, index=False)

    # Get absolute paths for clarity  
    csv_path = os.path.abspath(csv_file)
    results_dir_abs = os.path.abspath(results_dir)
    
    # Generate Summary Statistics from JSON Data
    print("\n📊 Step 7: Summary Statistics Generation")
    print("-" * 38)
    
    # Load the signatures data
    signatures_json_path = os.path.join(results_dir, 'rougerx_signatures_by_question_type.json')
    if os.path.exists(signatures_json_path):
        try:
            # Load and extract drug information
            signatures_data = load_signatures_data(signatures_json_path)
            drug_records = extract_drug_information(signatures_data)
            component_metrics = extract_component_metrics(signatures_data)
            
            print(f"✓ Extracted {len(drug_records)} drug records from signatures data")
            print(f"✓ Extracted {len(component_metrics)} component metrics from signatures data")
            
            # Generate basic summary statistics
            participant_summary = generate_participant_summary(drug_records)
            question_type_summary = generate_question_type_summary(drug_records)
            component_summaries = generate_component_summary(drug_records)
            
            print(f"✓ Generated participant summary ({len(participant_summary)} participants)")
            print(f"✓ Generated question type summary ({len(question_type_summary)} question types)")
            print(f"✓ Generated component summaries (5 components)")
            
            # Generate detailed summary statistics with enhanced metrics
            participant_detailed_summary = generate_participant_detailed_summary(component_metrics)
            question_type_detailed_summary = generate_question_type_detailed_summary(component_metrics)
            component_detailed_summary = generate_component_detailed_summary(component_metrics)
            
            print(f"✓ Generated detailed participant summary ({len(participant_detailed_summary)} participant-component combinations)")
            print(f"✓ Generated detailed question type summary ({len(question_type_detailed_summary)} question-type-component combinations)")
            print(f"✓ Generated detailed component summary ({len(component_detailed_summary)} components with enhanced metrics)")
            
            # Export all summaries to CSV
            summary_output_dir = os.path.join(results_dir, 'summary_statistics')
            export_summary_csvs(participant_summary, question_type_summary, 
                               component_summaries, summary_output_dir)
            
            # Export detailed summaries to CSV
            export_detailed_summary_csvs(participant_detailed_summary, question_type_detailed_summary,
                                       component_detailed_summary, summary_output_dir)
            
            # Calculate descriptive statistics
            descriptive_stats = calculate_descriptive_stats(component_metrics)

            # Print descriptive statistics and final table
            print("\n📈 Descriptive Statistics for Omission and Overlap Percentages")
            print("=" * 60)

            # Print question type statistics
            print("\nQuestion Type Statistics:")
            print("-" * 30)
            for q_type, stats in descriptive_stats['question_type_stats'].items():
                print(f"\n{q_type.upper()} Question Type:")
                for metric, values in stats.items():
                    component, metric_type = metric.rsplit('_', 1)
                    print(f"  {component} {metric_type}: Mean={values['mean']:.2f}, Std={values['std']:.2f} (n={values['count']})")

            # Print component statistics
            print("\nComponent Statistics (Overall):")
            print("-" * 35)
            for component, stats in descriptive_stats['component_stats'].items():
                print(f"\n{component}:")
                print(f"  Omission: Mean={stats['omission']['mean']:.2f}, Std={stats['omission']['std']:.2f} (n={stats['omission']['count']})")
                print(f"  Overlap:  Mean={stats['overlap']['mean']:.2f}, Std={stats['overlap']['std']:.2f} (n={stats['overlap']['count']})")

            # Print overall statistics
            print("\nOverall Statistics:")
            print("-" * 20)
            overall = descriptive_stats['overall_stats']
            print(f"Omission Percentage: Mean={overall['omission_percentage']['mean']:.2f}, Std={overall['omission_percentage']['std']:.2f} (n={overall['omission_percentage']['count']})")
            print(f"Overlap Percentage:  Mean={overall['overlap_percentage']['mean']:.2f}, Std={overall['overlap_percentage']['std']:.2f} (n={overall['overlap_percentage']['count']})")

            # Print final summary table
            print("\n📊 Final Summary Table")
            print("=" * 60)
            final_table = create_final_summary_table(component_metrics, descriptive_stats, signatures_data)
            print(final_table)

            print(f"✓ All summary statistics exported to: {os.path.abspath(summary_output_dir)}")
            print(f"  📊 Basic summaries: participant, question_type, and component stats")
            print(f"  📈 Detailed summaries: enhanced with divergence_score, omission_percentage, exact_percentage, overlap_percentage")

        except Exception as e:
            print(f"❌ Error generating summary statistics: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠️ Signatures JSON file not found: {signatures_json_path}")

    # Print GPT-4o-mini token usage and costs
    if OPENAI_AVAILABLE and openai_client:
        print("\n💰 GPT-4o-mini Token Usage and Cost Summary")
        print("=" * 45)
        cost_info = get_gpt4o_mini_cost()
        print(f"API Calls Made: {cost_info['api_calls']}")
        print(f"Input Tokens: {cost_info['input_tokens']:,} (${cost_info['input_cost_usd']:.4f})")
        print(f"Output Tokens: {cost_info['output_tokens']:,} (${cost_info['output_cost_usd']:.4f})")
        print(f"Total Tokens: {cost_info['total_tokens']:,} (${cost_info['total_cost_usd']:.4f})")
    else:
        print("\n⚠️  OpenAI not available - no token usage to report")

    print(f"\n✓ Analysis complete! Results saved to: {results_dir_abs}")
    print(f"  CSV Summary:  {csv_path} ({os.path.getsize(csv_file)} bytes)")
    print(f"  CSV Summary: Common words for {len(common_words_csv)} questions")
    print(f"  Study 2: {study2_results['overall_stats']['total_answers_analyzed']} answers analyzed across 3 question types")
    print(f"  Study 3: {study3_results['questions_processed']} questions parsed with {study3_results['parsing_method']} method")
    print(f"    📊 Advanced Analytics: component overlap, positional precision ({study3_results.get('positional_precision_analysis', {}).get('positional_precision', 0):.1f}%), similarity analysis")

    # Generate and display final summary table
    print("\n" + "="*60)
    print("📋 FINAL SUMMARY TABLE")
    print("="*60)
    summary_table = generate_final_summary_table()
    print(summary_table)

def generate_final_summary_table(json_filepath: str = None) -> str:
    """
    Generate the final summary table showing comprehensive medication communication analysis.

    Creates a table with:
    - Overall statistics across all participants
    - Individual participant statistics (ID_1, ID_2, ID_3, ID_4)
    - Breakdown by communication type (formal, verbal, brief)
    - Metrics: omission_percentage, overlap_percentage, ordering distributions

    Args:
        json_filepath: Path to the rougerx_signatures_by_question_type.json file

    Returns:
        Formatted table string
    """
    if json_filepath is None:
        json_filepath = 'results/rougerx/rougerx_signatures_by_question_type.json'

    # Load the JSON data
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        return f"❌ Error: JSON file not found at {json_filepath}"
    except json.JSONDecodeError as e:
        return f"❌ Error parsing JSON: {e}"

    # Initialize data structures
    communication_types = ['formal_question', 'verbal_question', 'brief_question']
    communication_labels = ['formal', 'verbal', 'brief']  # For display
    participants = ['ID_1', 'ID_2', 'ID_3', 'ID_4']
    components = ['drug_name', 'dose', 'unit', 'route', 'frequency']

    # Data collection structure - track stats by communication type
    comm_type_stats = {}
    for label in communication_labels:
        comm_type_stats[label] = {
            'overall': {comp: {'omission': 0, 'ordering': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}, 'total': 0, 'values': [], 'overlap_percentages': []} for comp in components},
            **{f'ID_{i}': {comp: {'omission': 0, 'ordering': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}, 'total': 0, 'values': []} for comp in components} for i in range(1, 5)}
        }

    # Process each communication type
    for comm_type_key, display_label in zip(communication_types, communication_labels):
        if comm_type_key not in data:
            continue

        comm_data = data[comm_type_key]
        if 'answers' not in comm_data:
            continue

        questions = comm_data['answers']
        if not isinstance(questions, list):
            continue

        for question in questions:
            if 'study_3_components' not in question or 'parsed_drugs' not in question['study_3_components']:
                continue

            parsed_drugs = question['study_3_components']['parsed_drugs']
            if not parsed_drugs:
                continue

            # Group drugs by source_id
            drugs_by_source = {}
            for drug in parsed_drugs:
                source_id = drug.get('source_id', '')
                if source_id not in drugs_by_source:
                    drugs_by_source[source_id] = []
                drugs_by_source[source_id].append(drug)

            # Process each participant's responses for this communication type
            for participant in participants:
                if participant not in drugs_by_source:
                    # No responses from this participant - count as omissions
                    for comp in components:
                        comm_type_stats[display_label]['overall'][comp]['omission'] += 1
                        comm_type_stats[display_label]['overall'][comp]['total'] += 1
                        comm_type_stats[display_label][participant][comp]['omission'] += 1
                        comm_type_stats[display_label][participant][comp]['total'] += 1
                    continue

                participant_drugs = drugs_by_source[participant]
                if not participant_drugs:
                    # Empty response - count as omissions
                    for comp in components:
                        comm_type_stats[display_label]['overall'][comp]['omission'] += 1
                        comm_type_stats[display_label]['overall'][comp]['total'] += 1
                        comm_type_stats[display_label][participant][comp]['omission'] += 1
                        comm_type_stats[display_label][participant][comp]['total'] += 1
                    continue

                # Use the first drug found for this participant (assuming single medication per response)
                drug = participant_drugs[0]

                # Get the original response text for ordering detection
                question_text = question.get('responses', {}).get(participant, '')

                # Process each component for this communication type
                for comp in components:
                    comp_value = drug.get(comp, '')
                    # Use original ordering from parsed drug (only explicit entities)
                    ordering = drug.get(f'{comp}_ordering', 0)

                    # Check if component is explicit (ordering > 0) or inferred (ordering = 0)
                    is_explicit = ordering > 0
                    is_inferred = (comp_value and comp_value.strip()) and ordering == 0

                    # Treat inferred components as missing (per user requirement)
                    if is_inferred:
                        comp_value = ''  # Clear inferred value

                    comm_type_stats[display_label]['overall'][comp]['total'] += 1
                    comm_type_stats[display_label][participant][comp]['total'] += 1

                    # Store component value for overlap calculation (after clearing inferred)
                    comm_type_stats[display_label]['overall'][comp]['values'].append(comp_value)
                    comm_type_stats[display_label][participant][comp]['values'].append(comp_value)

                    # Check for omission (empty or missing OR inferred)
                    if not comp_value or comp_value == '':
                        comm_type_stats[display_label]['overall'][comp]['omission'] += 1
                        comm_type_stats[display_label][participant][comp]['omission'] += 1
                    else:
                        # Record ordering position (only for explicitly stated entities)
                        if 1 <= ordering <= 5:
                            comm_type_stats[display_label]['overall'][comp]['ordering'][ordering] += 1
                            comm_type_stats[display_label][participant][comp]['ordering'][ordering] += 1

    # Calculate question-level overlap for each communication type
    for comm_type_key, display_label in zip(communication_types, communication_labels):
        if comm_type_key not in data:
            continue

        comm_data = data[comm_type_key]
        if 'answers' not in comm_data:
            continue

        questions = comm_data['answers']
        for question in questions:
            if 'study_3_components' not in question or 'parsed_drugs' not in question['study_3_components']:
                continue

            parsed_drugs = question['study_3_components']['parsed_drugs']
            if not parsed_drugs:
                continue

            # Group drugs by source_id
            drugs_by_source = {}
            for drug in parsed_drugs:
                source_id = drug.get('source_id', '')
                if source_id not in drugs_by_source:
                    drugs_by_source[source_id] = []
                drugs_by_source[source_id].append(drug)

            # Extract overlap_percentage from existing component_divergence data
            if 'component_divergence' in question.get('study_3_components', {}):
                component_divergence = question['study_3_components']['component_divergence']
                for comp in components:
                    if comp in component_divergence:
                        overlap_pct = component_divergence[comp].get('overlap_percentage', 0.0)
                        comm_type_stats[display_label]['overall'][comp]['overlap_percentages'].append(overlap_pct)

    # Calculate overlap percentages as average of word-level partial match percentages
    def calculate_average_overlap_percentage(overlap_percentages):
        """Calculate average overlap_percentage from component_divergence data (word-level partial match)"""
        if not overlap_percentages:
            return 0.0
        avg = sum(overlap_percentages) / len(overlap_percentages)
        return avg

    # Update the table with proper overlap calculations
    table_lines = []
    table_lines.append("🏥 MEDICATION COMMUNICATION ANALYSIS - FINAL SUMMARY TABLE")
    table_lines.append("=" * 120)

    # Header
    header = f"{'':<15} {'':<12} {'Overall':<12} {'ID_1':<20} {'ID_2':<20} {'ID_3':<20} {'ID_4':<20}"
    table_lines.append(header)

    subheader = f"{'Question Type':<15} {'Entity':<12} {'Omission%':<6} {'Overlap%':<6} {'Omission%':<6} {'Order(1,2,3,4,5)%':<14} {'Omission%':<6} {'Order(1,2,3,4,5)%':<14} {'Omission%':<6} {'Order(1,2,3,4,5)%':<14} {'Omission%':<6} {'Order(1,2,3,4,5)%':<14}"
    table_lines.append(subheader)
    table_lines.append("-" * 120)

    # Generate rows for each communication type and component
    for display_label in communication_labels:
        for comp in components:
            row_parts = [f"{display_label:<15}", f"{comp:<12}"]

            # Overall statistics for this communication type
            overall = comm_type_stats[display_label]['overall'][comp]
            if overall['total'] > 0:
                omission_pct = (overall['omission'] / overall['total']) * 100
                # Use average of word-level partial match overlap percentages
                overlap_pct = calculate_average_overlap_percentage(overall.get('overlap_percentages', []))
                ordering_dist = [overall['ordering'][i] for i in range(1, 6)]
                total_orderings = sum(ordering_dist)
                if total_orderings > 0:
                    ordering_pct = [f"{(count/total_orderings)*100:.1f}" for count in ordering_dist]
                else:
                    ordering_pct = ["0.0"] * 5
            else:
                omission_pct = 0.0
                overlap_pct = 0.0
                ordering_pct = ["0.0"] * 5

            row_parts.extend([f"{omission_pct:.1f}", f"{overlap_pct:.1f}"])

            # Individual participant statistics for this communication type
            for participant in participants:
                part_stats = comm_type_stats[display_label][participant][comp]
                if part_stats['total'] > 0:
                    part_omission = (part_stats['omission'] / part_stats['total']) * 100
                    part_ordering_dist = [part_stats['ordering'][i] for i in range(1, 6)]
                    part_total_orderings = sum(part_ordering_dist)
                    if part_total_orderings > 0:
                        part_ordering_pct = [f"{(count/part_total_orderings)*100:.1f}" for count in part_ordering_dist]
                    else:
                        part_ordering_pct = ["0.0"] * 5
                else:
                    part_omission = 0.0
                    part_ordering_pct = ["0.0"] * 5

                row_parts.extend([f"{part_omission:.1f}", f"({','.join(part_ordering_pct)})"])

            table_lines.append(" ".join(row_parts))

        # Add spacing between communication types
        table_lines.append("")

    return "\n".join(table_lines)


def main():
    """
    Complete RougeRx Survey Data Analysis Pipeline

    Executes comprehensive analysis of healthcare professional medication communication patterns
    using a streamlined, modular approach with simplified parsing logic.
    """
    # Setup environment and load data
    csv_path, results_dir = setup_analysis_environment()
    df, medications, response_cols, num_respondents = load_and_parse_data(csv_path)

    # Run exploratory studies
    exploratory_results, study2_results, study3_results = run_exploratory_studies(df, medications, response_cols)

    # Run advanced analysis
    signature_results, pattern_results = run_advanced_analysis(study3_results, df, medications, response_cols)

    # Generate outputs and reports
    generate_final_outputs(exploratory_results, study2_results, study3_results, signature_results,
                          pattern_results, medications, num_respondents, results_dir)

    # Run new GPT-4o-mini category analysis
    print("\n" + "="*60)
    print("🤖 GPT-4o-mini Category Analysis")
    print("="*60)
    category_results = analyze_by_category_with_gpt4o_mini()
    print(f"✅ Completed category analysis for {category_results['questions_analyzed']} questions")


def load_signatures_data(filepath: str) -> Dict[str, Any]:
    """Load and parse the rougerx_signatures_by_question_type.json file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_drug_information(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract drug information from all questions and participants"""
    drug_records = []
    
    for question_type, question_data in data.items():
        if 'answers' in question_data:
            for answer in question_data['answers']:
                question_id = answer.get('question_id', 'Unknown')
                
                # Extract responses from each participant
                responses = answer.get('responses', {})
                for participant_id, response_text in responses.items():
                    # Extract parsed drug components
                    study_3_components = answer.get('study_3_components', {})
                    parsed_drugs = study_3_components.get('parsed_drugs', [])
                    
                    # For each parsed drug, create a record
                    for i, drug in enumerate(parsed_drugs):
                        if isinstance(drug, dict):
                            record = {
                                'question_type': question_type.replace('_question', ''),
                                'question_id': question_id,
                                'participant_id': participant_id,
                                'response_text': response_text,
                                'drug_index': i,
                                'drug_name': drug.get('drug_name', ''),
                                'dose': drug.get('dose', ''),
                                'unit': drug.get('unit', ''),
                                'route': drug.get('route', ''),
                                'frequency': drug.get('frequency', ''),
                                'drug_name_ordering': drug.get('drug_name_ordering', None),
                                'dose_ordering': drug.get('dose_ordering', None),
                                'unit_ordering': drug.get('unit_ordering', None),
                                'route_ordering': drug.get('route_ordering', None),
                                'frequency_ordering': drug.get('frequency_ordering', None)
                            }
                            drug_records.append(record)
    
    return drug_records

def extract_component_metrics(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract detailed component metrics from component_divergence data"""
    component_metrics = []
    
    for question_type, question_data in data.items():
        if 'answers' in question_data:
            for answer in question_data['answers']:
                question_id = answer.get('question_id', 'Unknown')
                
                # Extract component divergence metrics
                study_3_components = answer.get('study_3_components', {})
                component_divergence = study_3_components.get('component_divergence', {})
                
                # Process each component (drug_name, dose, unit, route, frequency)
                for component_name, metrics in component_divergence.items():
                    if isinstance(metrics, dict):
                        record = {
                            'question_type': question_type.replace('_question', ''),
                            'question_id': question_id,
                            'component_name': component_name,
                            'unique_values': metrics.get('unique_values', 0),
                            'total_parsed': metrics.get('total_parsed', 0),
                            'total_drugs': metrics.get('total_drugs', 0),
                            'divergence_score': metrics.get('divergence_score', 0.0),
                            'omitted_count': metrics.get('omitted_count', 0),
                            'omission_percentage': metrics.get('omission_percentage', 0.0),
                            'exact_matches': metrics.get('exact_matches', 0),
                            'partial_overlaps': metrics.get('partial_overlaps', 0),
                            'total_comparisons': metrics.get('total_comparisons', 0),
                            'exact_percentage': metrics.get('exact_percentage', 0.0),
                            'overlap_percentage': metrics.get('overlap_percentage', 0.0),
                            'values': metrics.get('values', []),
                            'values_count': len(metrics.get('values', [])),
                            'values_unique': len(set(metrics.get('values', []))) if metrics.get('values') else 0
                        }
                        component_metrics.append(record)
    
    return component_metrics

def extract_participant_component_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract participant-specific component metrics from the raw signatures data."""
    participant_metrics = {
        'formal': {'ID_1': {}, 'ID_2': {}, 'ID_3': {}, 'ID_4': {}},
        'verbal': {'ID_1': {}, 'ID_2': {}, 'ID_3': {}, 'ID_4': {}},
        'brief': {'ID_1': {}, 'ID_2': {}, 'ID_3': {}, 'ID_4': {}}
    }

    components = ['drug_name', 'dose', 'unit', 'route', 'frequency']

    for question_type_key, question_data in data.items():
        if question_type_key.endswith('_question') and 'answers' in question_data:
            question_type = question_type_key.replace('_question', '')

            for answer in question_data['answers']:
                question_id = answer.get('question_id', '')

                # Get study_3_components for this question
                study_3 = answer.get('study_3_components', {})
                parsed_drugs = study_3.get('parsed_drugs', [])

                # parsed_drugs should correspond to ID_1, ID_2, ID_3, ID_4 in order
                for participant_idx, drug_info in enumerate(parsed_drugs[:4]):  # Limit to 4 participants
                    participant_id = f'ID_{participant_idx + 1}'

                    if isinstance(drug_info, dict):
                        # Extract component values for this participant
                        for component in components:
                            component_value = drug_info.get(component, '').strip()
                            key = f'{question_type}_{component}'

                            if key not in participant_metrics[question_type][participant_id]:
                                participant_metrics[question_type][participant_id][key] = {
                                    'omission_count': 0,
                                    'total_count': 0,
                                    'overlap_values': []
                                }

                            # Track if component is present (not empty)
                            participant_metrics[question_type][participant_id][key]['total_count'] += 1
                            if not component_value:
                                participant_metrics[question_type][participant_id][key]['omission_count'] += 1
                            else:
                                participant_metrics[question_type][participant_id][key]['overlap_values'].append(component_value)

    return participant_metrics

def calculate_descriptive_stats(component_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate descriptive statistics for omission and overlap percentages by question type and component."""
    import numpy as np

    stats_results = {
        'question_type_stats': {},
        'component_stats': {},
        'overall_stats': {}
    }

    # Group data by question type and component
    question_type_data = defaultdict(lambda: defaultdict(list))
    component_data = defaultdict(list)

    for record in component_metrics:
        q_type = record['question_type']
        component = record['component_name']
        omission_pct = record['omission_percentage']
        overlap_pct = record['overlap_percentage']

        # Collect data for question type analysis
        question_type_data[q_type][f'{component}_omission'].append(omission_pct)
        question_type_data[q_type][f'{component}_overlap'].append(overlap_pct)

        # Collect data for component analysis
        component_data[f'{component}_omission'].append(omission_pct)
        component_data[f'{component}_overlap'].append(overlap_pct)

    # Calculate statistics for each question type
    for q_type, metrics in question_type_data.items():
        stats_results['question_type_stats'][q_type] = {}

        for metric_name, values in metrics.items():
            if values:  # Only calculate if we have data
                stats_results['question_type_stats'][q_type][metric_name] = {
                    'mean': round(np.mean(values), 2),
                    'std': round(np.std(values), 2),
                    'count': len(values),
                    'min': round(min(values), 2),
                    'max': round(max(values), 2)
                }

    # Calculate overall statistics for each component
    components = ['drug_name', 'dose', 'unit', 'route', 'frequency']
    for component in components:
        omission_key = f'{component}_omission'
        overlap_key = f'{component}_overlap'

        omission_values = component_data.get(omission_key, [])
        overlap_values = component_data.get(overlap_key, [])

        stats_results['component_stats'][component] = {
            'omission': {
                'mean': round(np.mean(omission_values), 2) if omission_values else 0.0,
                'std': round(np.std(omission_values), 2) if omission_values else 0.0,
                'count': len(omission_values)
            },
            'overlap': {
                'mean': round(np.mean(overlap_values), 2) if overlap_values else 0.0,
                'std': round(np.std(overlap_values), 2) if overlap_values else 0.0,
                'count': len(overlap_values)
            }
        }

    # Calculate overall statistics
    all_omission_values = [record['omission_percentage'] for record in component_metrics]
    all_overlap_values = [record['overlap_percentage'] for record in component_metrics]

    stats_results['overall_stats'] = {
        'omission_percentage': {
            'mean': round(np.mean(all_omission_values), 2),
            'std': round(np.std(all_omission_values), 2),
            'count': len(all_omission_values)
        },
        'overlap_percentage': {
            'mean': round(np.mean(all_overlap_values), 2),
            'std': round(np.std(all_overlap_values), 2),
            'count': len(all_overlap_values)
        }
    }

    return stats_results

def create_final_summary_table(component_metrics: List[Dict[str, Any]], descriptive_stats: Dict[str, Any], data: Dict[str, Any]) -> str:
    """Create the final summary table in the requested format."""
    # Define the structure
    question_types = ['formal', 'verbal', 'brief']
    components = ['drug_name', 'dose', 'unit', 'route', 'frequency']
    participant_ids = ['ID_1', 'ID_2', 'ID_3', 'ID_4']

    # Create table header
    header = "\t\tOverall\t\tID_1\t\tID_2\t\tID_3\t\tID_4"
    subheader = "question type \tentities type\tomission_percentage\toverlap_percentage\tomission_percentage\toverlap_percentage\tomission_percentage\toverlap_percentage\tomission_percentage\toverlap_percentage\tomission_percentage\toverlap_percentage"

    # Calculate participant-specific stats (omission only, no consensus overlap)
    participant_metrics = extract_participant_component_metrics(data)

    table_lines = [header, subheader]

    for question_type in question_types:
        for component in components:
            # Get question-type-specific overall stats from descriptive_stats
            if question_type in descriptive_stats['question_type_stats']:
                qtype_stats = descriptive_stats['question_type_stats'][question_type]

                # Get the mean values for this component
                omission_key = f'{component}_omission'
                overlap_key = f'{component}_overlap'

                overall_omission = qtype_stats.get(omission_key, {}).get('mean', 0.0)
                overall_overlap = qtype_stats.get(overlap_key, {}).get('mean', 0.0)
            else:
                # Fallback to overall component stats if question type not found
                overall_omission = descriptive_stats['component_stats'][component]['omission']['mean']
                overall_overlap = descriptive_stats['component_stats'][component]['overlap']['mean']

            # Create table row
            row = f"{question_type}\t{component}\t{overall_omission:.2f}\t{overall_overlap:.2f}"

            # Add participant-specific stats (omission only, use overall overlap for all)
            for pid in participant_ids:
                # Calculate participant-specific omission from participant_metrics
                omission_pct = overall_omission  # Default fallback
                if pid in participant_metrics[question_type]:
                    pid_metrics = participant_metrics[question_type][pid]
                    key = f'{question_type}_{component}'
                    if key in pid_metrics:
                        total_count = pid_metrics[key]['total_count']
                        omission_count = pid_metrics[key]['omission_count']
                        if total_count > 0:
                            omission_pct = (omission_count / total_count) * 100

                # Use overall overlap percentage for all participants (no consensus method)
                overlap_pct = overall_overlap

                row += f"\t{omission_pct:.2f}\t{overlap_pct:.2f}"

            table_lines.append(row)

    return "\n".join(table_lines)

def generate_participant_summary(drug_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generate summary statistics by participant ID"""
    participant_stats = []
    
    participants = ['ID_1', 'ID_2', 'ID_3', 'ID_4']
    
    for participant in participants:
        participant_data = [record for record in drug_data if record['participant_id'] == participant]
        
        if not participant_data:
            continue
            
        # Count statistics
        total_responses = len(participant_data)
        unique_drugs = len(set(record['drug_name'] for record in participant_data if record['drug_name']))
        
        stats = {
            'participant_id': participant,
            'total_responses': total_responses,
            'unique_drugs': unique_drugs
        }
        
        participant_stats.append(stats)
    
    return pd.DataFrame(participant_stats)

def generate_question_type_summary(drug_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generate summary statistics by question type"""
    question_type_stats = []
    
    question_types = ['formal', 'verbal', 'brief']
    
    for q_type in question_types:
        type_data = [record for record in drug_data if record['question_type'] == q_type]
        
        if not type_data:
            continue
            
        # Count statistics
        total_responses = len(type_data)
        unique_drugs = len(set(record['drug_name'] for record in type_data if record['drug_name']))
        
        stats = {
            'question_type': q_type,
            'total_responses': total_responses,
            'unique_drugs': unique_drugs
        }
        
        question_type_stats.append(stats)
    
    return pd.DataFrame(question_type_stats)

def generate_component_summary(drug_data: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    """Generate summary statistics by drug components"""
    components = ['drug_name', 'dose', 'unit', 'route', 'frequency']
    component_summaries = {}
    
    for component in components:
        component_stats = []
        
        # Get all non-empty values for this component
        component_values = [record[component] for record in drug_data if record[component]]
        value_counts = Counter(component_values)
        
        # Create summary for each unique value
        for value, count in value_counts.most_common():
            stats = {
                f'{component}_value': value,
                'total_occurrences': count,
                'percentage_of_total': (count / len(drug_data)) * 100 if drug_data else 0
            }
            
            component_stats.append(stats)
        
        component_summaries[component] = pd.DataFrame(component_stats)
    
    return component_summaries

def generate_participant_detailed_summary(component_metrics: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generate detailed summary statistics by participant ID using component metrics"""
    participant_stats = []
    
    participants = ['ID_1', 'ID_2', 'ID_3', 'ID_4']
    components = ['drug_name', 'dose', 'unit', 'route', 'frequency']
    
    for participant_id in participants:
        for component in components:
            # Simple placeholder stats
            stats = {
                'participant_id': participant_id,
                'component_name': component,
                'total_questions': len([r for r in component_metrics if r['component_name'] == component]),
                'avg_divergence_score': 0.0,
                'avg_omission_percentage': 0.0
            }
            
            participant_stats.append(stats)
    
    return pd.DataFrame(participant_stats)

def generate_question_type_detailed_summary(component_metrics: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generate detailed summary statistics by question type using component metrics"""
    question_type_stats = []
    
    question_types = ['formal', 'verbal', 'brief']
    components = ['drug_name', 'dose', 'unit', 'route', 'frequency']
    
    for question_type in question_types:
        for component in components:
            # Simple placeholder stats
            stats = {
                'question_type': question_type,
                'component_name': component,
                'total_questions': len([r for r in component_metrics if r['component_name'] == component and r['question_type'] == question_type]),
                'avg_divergence_score': 0.0,
                'avg_omission_percentage': 0.0
            }
            
            question_type_stats.append(stats)
    
    return pd.DataFrame(question_type_stats)

def generate_component_detailed_summary(component_metrics: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generate detailed summary statistics by component using component metrics"""
    component_stats = []
    
    components = ['drug_name', 'dose', 'unit', 'route', 'frequency']
    
    for component in components:
        # Get all metrics for this component
        component_data = [record for record in component_metrics if record['component_name'] == component]
        
        if component_data:
            avg_divergence = sum(record['divergence_score'] for record in component_data) / len(component_data)
            avg_omission = sum(record['omission_percentage'] for record in component_data) / len(component_data)
        else:
            avg_divergence = 0.0
            avg_omission = 0.0
            
        stats = {
            'component_name': component,
            'total_questions': len(component_data),
            'avg_divergence_score': round(avg_divergence, 3),
            'avg_omission_percentage': round(avg_omission, 3)
        }
        
        component_stats.append(stats)
    
    return pd.DataFrame(component_stats)

def export_summary_csvs(participant_summary: pd.DataFrame, 
                       question_type_summary: pd.DataFrame,
                       component_summaries: Dict[str, pd.DataFrame],
                       output_dir: str) -> None:
    """Export all summary statistics to CSV files"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Export participant summary
    participant_file = os.path.join(output_dir, 'participant_summary_stats.csv')
    participant_summary.to_csv(participant_file, index=False)
    print(f"✅ Saved participant summary: {participant_file}")
    
    # Export question type summary
    question_type_file = os.path.join(output_dir, 'question_type_summary_stats.csv')
    question_type_summary.to_csv(question_type_file, index=False)
    print(f"✅ Saved question type summary: {question_type_file}")
    
    # Export component summaries
    for component, summary_df in component_summaries.items():
        component_file = os.path.join(output_dir, f'{component}_summary_stats.csv')
        summary_df.to_csv(component_file, index=False)
        print(f"✅ Saved {component} summary: {component_file}")

def export_detailed_summary_csvs(participant_detailed_summary: pd.DataFrame,
                                 question_type_detailed_summary: pd.DataFrame,
                                 component_detailed_summary: pd.DataFrame,
                                 output_dir: str) -> None:
    """Export enhanced detailed summary statistics to CSV files"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Export detailed summaries
    participant_detailed_file = os.path.join(output_dir, 'participant_detailed_summary_stats.csv')
    participant_detailed_summary.to_csv(participant_detailed_file, index=False)
    print(f"✅ Saved participant detailed summary: {participant_detailed_file}")
    
    question_type_detailed_file = os.path.join(output_dir, 'question_type_detailed_summary_stats.csv')
    question_type_detailed_summary.to_csv(question_type_detailed_file, index=False)
    print(f"✅ Saved question type detailed summary: {question_type_detailed_file}")
    
    component_detailed_file = os.path.join(output_dir, 'component_detailed_summary_stats.csv')
    component_detailed_summary.to_csv(component_detailed_file, index=False)
    print(f"✅ Saved component detailed summary: {component_detailed_file}")

def normalize_missing_value(value: str) -> str:
    """
    Normalize various missing value indicators to empty string.
    
    Converts "none", "N/A", "na", "null", "No frequency provided", etc. to empty string.
    This ensures consistent handling of missing values across the codebase.
    
    Args:
        value: The value to normalize
        
    Returns:
        Empty string if value represents missing data, otherwise the original value
    """
    if not value:
        return ""
    
    value_lower = value.strip().lower()
    
    # List of indicators that mean "missing" or "not provided"
    # Comprehensive list to catch variations GPT might return despite prompt instructions
    missing_indicators = [
        "none",
        "n/a",
        "na",
        "null",
        "no frequency provided",
        "not provided",
        "missing",
        "n.a.",
        "n.a",
        "-",
        "--",
        "not available",
        "unavailable",
        "not specified",
        "unspecified",
        "not mentioned",
        "not stated",
        "not given",
        "absent",
        "not found",
        "no route provided",
        "no dose provided",
        "no unit provided",
        "no drug name provided",
        "no route",
        "no dose",
        "no unit",
        "no drug name",
        "no frequency",
    ]
    
    if value_lower in missing_indicators:
        return ""
    
    return value.strip()


def extract_component_with_gpt(component_name: str, response_text: str) -> str:
    """
    Use GPT-4o-mini to extract a medication component directly from response text.

    Args:
        component_name: Name of component (drug_name, dose, unit, route, frequency)
        response_text: Full response text to extract from

    Returns:
        Extracted component value using GPT-4o-mini
    """
    if not OPENAI_AVAILABLE or not openai_client:
        return ""  # No fallback - need GPT for extraction

    if not response_text or response_text.strip() == '':
        return ""

    # Get prompt from external prompts file
    prompt = get_component_extraction_prompt(component_name, response_text)

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=20,  # Short responses expected
            temperature=0.1
        )

        # Update token usage
        update_token_usage(response)

        extracted = response.choices[0].message.content.strip()

        # Clean up response (remove quotes, extra whitespace)
        extracted = extracted.strip('"\'').strip()

        # Normalize missing values (none, N/A, etc.) to empty string
        extracted = normalize_missing_value(extracted)

        return extracted

    except Exception as e:
        print(f"⚠️ GPT extraction failed for {component_name}: {response_text[:50]}... - {e}")
        return ""


def calculate_jaccard_similarity(text1: str, text2: str) -> Union[float, str]:
    """
    Calculate Jaccard similarity between two raw extracted strings using word-level tokenization.
    No stop word removal - uses extracted values exactly as-is for comparison.

    Args:
        text1: First raw extracted string
        text2: Second raw extracted string

    Returns:
        Jaccard similarity score (0.0 to 1.0) or "N/A" if one value is empty
    """
    # Normalize missing values first
    text1 = normalize_missing_value(text1)
    text2 = normalize_missing_value(text2)
    
    # Check if either value is empty - return "N/A" for omissions
    if not text1 or not text2:
        return "N/A"
    
    # Also check for empty strings after stripping
    if not text1.strip() or not text2.strip():
        return "N/A"

    # Convert to lowercase and split into words (keep all words, no stop word removal)
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 and not words2:
        return 1.0  # Both empty
    if not words1 or not words2:
        return "N/A"  # One is empty - return N/A

    # Calculate Jaccard similarity using raw words
    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


def find_majority_vote(values: List[str]) -> str:
    """
    Find the majority vote among a list of values.
    Ignores empty strings and normalized missing values (none, N/A, etc.) when calculating majority.

    Args:
        values: List of string values

    Returns:
        The majority value (most frequent), or first non-empty value if tie
    """
    if not values:
        return ""

    # Normalize all values first (convert "none", "N/A", etc. to empty string)
    normalized_values = [normalize_missing_value(v) for v in values]
    
    # Filter out empty strings and normalized missing values
    non_empty_values = [v for v in normalized_values if v and v.strip()]

    if not non_empty_values:
        return ""

    # Count frequency of each non-empty value
    value_counts = Counter(non_empty_values)

    # Find the most common value
    most_common = value_counts.most_common(1)
    return most_common[0][0] if most_common else non_empty_values[0]


def analyze_question_components(question_id: str, responses: Dict[str, str]) -> Dict[str, Any]:
    """
    Analyze components for a single question.

    Args:
        question_id: The question identifier
        responses: Dict of respondent_id -> response_text

    Returns:
        Analysis results for this question
    """
    components = ['drug_name', 'dose', 'unit', 'route', 'frequency']
    respondent_ids = ['ID_1', 'ID_2', 'ID_3', 'ID_4']

    question_results = {
        'question_id': question_id,
        'components': {},
        'variation_analysis': {},
        'outlier_detection': {}
    }

    # Process each component
    for component in components:
        # Extract component from each respondent's response using GPT-4o-mini
        component_values = []

        for respondent_id in respondent_ids:
            response_text = responses.get(respondent_id, '')
            if response_text and response_text.strip():
                # Use GPT-4o-mini to extract component directly from response text
                extracted_value = extract_component_with_gpt(component, response_text)

                component_values.append({
                    'respondent_id': respondent_id,
                    'extracted_value': extracted_value,
                    'original_response': response_text
                })

        if len(component_values) < 4:  # Need all 4 respondents
            continue

        # Get all extracted values
        extracted_values = [cv['extracted_value'] for cv in component_values]

        # Find majority value (ignoring empty strings)
        majority_value = find_majority_vote(extracted_values)

        # Calculate Jaccard similarity between each value and majority
        matches = []
        omissions = 0  # Count empty values as omissions
        for cv in component_values:
            extracted_val = cv['extracted_value']
            
            # Count empty values as omissions
            if not extracted_val or not extracted_val.strip():
                omissions += 1
            
            jaccard_sim = calculate_jaccard_similarity(extracted_val, majority_value)
            matches.append({
                'respondent_id': cv['respondent_id'],
                'value': extracted_val,
                'jaccard_similarity': jaccard_sim,
                'original_response': cv['original_response']
            })

        question_results['components'][component] = {
            'majority_value': majority_value,
            'respondent_matches': matches,
            'omissions': omissions  # Track omissions for this component
        }

    # Analyze variation using Jaccard similarity
    variation_scores = {}
    overlap_analysis = {}  # Calculate overlap based on jaccard_similarity
    
    for component in components:
        if component not in question_results['components']:
            continue

        matches = question_results['components'][component]['respondent_matches']
        
        # Filter out "N/A" values for statistics calculation
        jaccard_scores_numeric = [m['jaccard_similarity'] for m in matches 
                                  if isinstance(m['jaccard_similarity'], (int, float))]
        jaccard_scores_all = [m['jaccard_similarity'] for m in matches]

        # Calculate variation metrics based on Jaccard similarity (only numeric values)
        avg_jaccard = sum(jaccard_scores_numeric) / len(jaccard_scores_numeric) if jaccard_scores_numeric else 0
        min_jaccard = min(jaccard_scores_numeric) if jaccard_scores_numeric else 0
        max_jaccard = max(jaccard_scores_numeric) if jaccard_scores_numeric else 0

        # Variation score: 1 - average similarity (higher = more variation)
        variation_score = 1 - avg_jaccard

        variation_scores[component] = {
            'avg_jaccard_similarity': avg_jaccard,
            'min_jaccard_similarity': min_jaccard,
            'max_jaccard_similarity': max_jaccard,
            'variation_score': variation_score,
            'total_respondents': len(jaccard_scores_all),
            'omissions': question_results['components'][component].get('omissions', 0)
        }
        
        # Calculate overlap: % overlapping to majority value
        # Count how many items have jaccard_similarity > 0 and != "N/A" when compared to majority
        overlap_count = 0
        total_items = len(matches)
        
        # Count items that overlap with majority value
        for match in matches:
            jaccard_sim = match['jaccard_similarity']
            # Consider overlapping if jaccard_similarity > 0 and != "N/A" (not empty)
            if isinstance(jaccard_sim, (int, float)) and jaccard_sim > 0:
                overlap_count += 1
        
        overlap_percentage = (overlap_count / total_items * 100) if total_items > 0 else 0
        
        overlap_analysis[component] = {
            'overlap_count': overlap_count,
            'total_items': total_items,
            'overlap_percentage': overlap_percentage,
            'description': 'Percentage of items overlapping with majority value'
        }

    question_results['variation_analysis'] = variation_scores
    question_results['overlap_analysis'] = overlap_analysis

    # Identify outlier (respondent with lowest Jaccard similarity to majority)
    # Only consider numeric jaccard_similarity values (exclude "N/A")
    outlier_analysis = {}

    for component in components:
        if component not in question_results['components']:
            continue

        matches = question_results['components'][component]['respondent_matches']
        
        # Filter to only numeric jaccard_similarity values for outlier detection
        numeric_matches = [m for m in matches if isinstance(m['jaccard_similarity'], (int, float))]

        # Find respondent with lowest Jaccard similarity (biggest semantic difference)
        if numeric_matches:
            min_similarity_match = min(numeric_matches, key=lambda x: x['jaccard_similarity'])
            outlier_analysis[component] = {
                'biggest_outlier': {
                    'respondent_id': min_similarity_match['respondent_id'],
                    'jaccard_similarity': min_similarity_match['jaccard_similarity'],
                    'value': min_similarity_match['value'],
                    'majority_value': question_results['components'][component]['majority_value']
                }
            }

    question_results['outlier_detection'] = outlier_analysis

    return question_results


def analyze_by_category_with_gpt4o_mini(max_questions: int = None) -> Dict[str, Any]:
    """
    Analyze medication communication patterns by category using GPT-4o-mini parsing.

    For each question in rougerx_enhanced_analysis.json:
    1. Use GPT-4o-mini to extract each component directly from original response text
    2. Find majority value for each component across 4 respondents
    3. Calculate raw string Jaccard similarity (no stop word removal) between each respondent's component and majority
    4. Identify which respondent has biggest semantic difference from the other three
    5. Report variation analysis by category (formal/verbal/brief) and overall outlier

    Returns comprehensive category analysis with GPT-4o-mini direct extraction and raw string Jaccard similarity.
    """
    results = {
        'analysis_method': 'gpt4o_mini_direct_extraction',
        'timestamp': pd.Timestamp.now().isoformat(),
        'questions_analyzed': 0,
        'by_question': {},
        'summary_stats': {}
    }

    # Load enhanced analysis JSON
    enhanced_json_path = 'results/rougerx/rougerx_enhanced_analysis.json'
    try:
        with open(enhanced_json_path, 'r', encoding='utf-8') as f:
            enhanced_data = json.load(f)
        print(f"📖 Loaded {len(enhanced_data)} questions from enhanced analysis")
    except FileNotFoundError:
        print(f"❌ Enhanced analysis JSON not found: {enhanced_json_path}")
        return results
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing enhanced JSON: {e}")
        return results

    # Filter to subset for testing if requested
    if max_questions is not None:
        question_ids = [qid for qid in enhanced_data.keys() if qid.startswith('Q')][:max_questions]
        enhanced_data = {qid: enhanced_data[qid] for qid in question_ids}
        print(f"🧪 Test mode: Limited to {len(enhanced_data)} questions (requested max: {max_questions})")

    # Process each question
    for question_id, question_data in enhanced_data.items():
        if not question_id.startswith('Q'):
            continue

        if 'data' not in question_data:
            continue

        responses = question_data['data'].get('responses', {})

        # Need all 4 respondents
        respondent_ids = ['ID_1', 'ID_2', 'ID_3', 'ID_4']
        if not all(rid in responses for rid in respondent_ids):
            continue

        # Analyze this question
        question_results = analyze_question_components(question_id, responses)
        results['by_question'][question_id] = question_results
        results['questions_analyzed'] += 1

        # Progress indicator
        if results['questions_analyzed'] % 10 == 0:
            print(f"📊 Processed {results['questions_analyzed']} questions...")

    # Calculate summary statistics
    if results['by_question']:
        results['summary_stats'] = calculate_summary_stats(results['by_question'])

    # Save results to JSON file
    if max_questions is not None:
        output_path = f'results/rougerx/rougerx_by_category_test_{max_questions}.json'
        results['test_mode'] = True
        results['max_questions_limit'] = max_questions
    else:
        output_path = 'results/rougerx/rougerx_by_category.json'
        results['test_mode'] = False

    os.makedirs('results/rougerx', exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved category analysis to: {output_path}")
    print(f"   📊 Analyzed {results['questions_analyzed']} questions")
    if max_questions:
        print(f"   🧪 Test mode: Limited analysis ({max_questions} question limit)")
    print(f"   🤖 Used GPT-4o-mini for direct component extraction")
    print(f"   📈 Used raw string Jaccard similarity and identified outliers")

    return results


def calculate_summary_stats(by_question: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate summary statistics across all questions, including by category.

    Args:
        by_question: Results dictionary with question-by-question analysis

    Returns:
        Summary statistics dictionary
    """
    components = ['drug_name', 'dose', 'unit', 'route', 'frequency']
    categories = ['formal', 'verbal', 'brief']

    # Overall component statistics
    component_variation_summary = {}
    outlier_summary = {}

    # Category-specific statistics
    category_stats = {cat: {} for cat in categories}

    for component in components:
        all_variations = []
        all_jaccard_sims = []
        all_outliers = []

        # Category-specific collections
        category_variations = {cat: [] for cat in categories}
        category_jaccard_sims = {cat: [] for cat in categories}

        for question_id, question_results in by_question.items():
            if component in question_results['variation_analysis']:
                variation = question_results['variation_analysis'][component]
                all_variations.append(variation['variation_score'])
                all_jaccard_sims.append(variation['avg_jaccard_similarity'])

                # Determine category from question ID using actual CSV index (handles missing questions)
                index = get_question_index(question_id)
                if index is None:
                    continue
                category = get_question_type_from_index(index)

                category_variations[category].append(variation['variation_score'])
                category_jaccard_sims[category].append(variation['avg_jaccard_similarity'])

            if component in question_results['outlier_detection']:
                outlier_data = question_results['outlier_detection'][component]
                biggest_outlier = outlier_data['biggest_outlier']
                all_outliers.append(biggest_outlier['jaccard_similarity'])  # Use Jaccard similarity as outlier score

        # Overall component statistics
        if all_variations:
            avg_variation = sum(all_variations) / len(all_variations)
            avg_jaccard = sum(all_jaccard_sims) / len(all_jaccard_sims)
            component_variation_summary[component] = {
                'avg_variation_score': round(avg_variation, 3),
                'avg_jaccard_similarity': round(avg_jaccard, 3),
                'total_questions': len(all_variations)
            }

        if all_outliers:
            avg_outlier_sim = sum(all_outliers) / len(all_outliers)
            outlier_summary[component] = {
                'avg_outlier_jaccard_sim': round(avg_outlier_sim, 3),
                'total_questions': len(all_outliers)
            }

        # Category-specific statistics
        for category in categories:
            if category_variations[category]:
                avg_cat_variation = sum(category_variations[category]) / len(category_variations[category])
                avg_cat_jaccard = sum(category_jaccard_sims[category]) / len(category_jaccard_sims[category])

                if component not in category_stats[category]:
                    category_stats[category][component] = {}

                category_stats[category][component] = {
                    'avg_variation_score': round(avg_cat_variation, 3),
                    'avg_jaccard_similarity': round(avg_cat_jaccard, 3),
                    'total_questions': len(category_variations[category])
                }

    # Identify outlier respondents by category and overall
    outlier_analysis = identify_outliers_by_category(by_question)

    return {
        'component_variation_summary': component_variation_summary,
        'category_variation_summary': category_stats,
        'outlier_summary': outlier_summary,
        'outlier_analysis_by_category': outlier_analysis,
        'total_questions_analyzed': len(by_question)
    }


def identify_outliers_by_category(by_question: Dict[str, Any]) -> Dict[str, Any]:
    """
    Identify which respondent has the biggest semantic difference across all components,
    analyzed separately for each category (formal, verbal, brief) and by individual components.

    Args:
        by_question: Results dictionary with question-by-question analysis

    Returns:
        Information about the biggest outlier for each category/component and overall
    """
    components = ['drug_name', 'dose', 'unit', 'route', 'frequency']
    categories = ['formal', 'verbal', 'brief']

    # Initialize counters for each category and component
    category_component_outliers = {}
    for category in categories:
        category_component_outliers[category] = {}
        for component in components:
            category_component_outliers[category][component] = {'ID_1': 0, 'ID_2': 0, 'ID_3': 0, 'ID_4': 0}

    # Overall counters (by component and total)
    overall_component_outliers = {}
    for component in components:
        overall_component_outliers[component] = {'ID_1': 0, 'ID_2': 0, 'ID_3': 0, 'ID_4': 0}

    overall_total_outliers = {'ID_1': 0, 'ID_2': 0, 'ID_3': 0, 'ID_4': 0}

    for question_id, question_results in by_question.items():
        # Determine category from question ID using actual CSV index (handles missing questions)
        index = get_question_index(question_id)
        if index is None:
            continue
        category = get_question_type_from_index(index)

        for component in components:
            if component in question_results['outlier_detection']:
                outlier_data = question_results['outlier_detection'][component]
                biggest_outlier = outlier_data['biggest_outlier']
                respondent_id = biggest_outlier['respondent_id']

                # Count for this category and component
                category_component_outliers[category][component][respondent_id] += 1
                # Count for overall component
                overall_component_outliers[component][respondent_id] += 1
                # Count for overall total
                overall_total_outliers[respondent_id] += 1

    # Find biggest outlier for each category and component
    category_results = {}
    for category in categories:
        category_results[category] = {}
        total_category_components = 0

        for component in components:
            outlier_counts = category_component_outliers[category][component]
            biggest_outlier = max(outlier_counts.items(), key=lambda x: x[1])
            component_total = sum(outlier_counts.values())

            category_results[category][component] = {
                'biggest_outlier_respondent': biggest_outlier[0],
                'outlier_count': biggest_outlier[1],
                'total_questions': component_total,
                'all_outlier_counts': outlier_counts
            }
            total_category_components += component_total

        # Overall for this category
        category_total_outliers = {'ID_1': 0, 'ID_2': 0, 'ID_3': 0, 'ID_4': 0}
        for component in components:
            for respondent_id in ['ID_1', 'ID_2', 'ID_3', 'ID_4']:
                category_total_outliers[respondent_id] += category_component_outliers[category][component][respondent_id]

        category_biggest = max(category_total_outliers.items(), key=lambda x: x[1])
        category_results[category]['overall'] = {
            'biggest_outlier_respondent': category_biggest[0],
            'outlier_count': category_biggest[1],
            'total_components': total_category_components,
            'all_outlier_counts': category_total_outliers
        }

    # Overall results by component
    overall_component_results = {}
    for component in components:
        outlier_counts = overall_component_outliers[component]
        biggest_outlier = max(outlier_counts.items(), key=lambda x: x[1])
        component_total = sum(outlier_counts.values())

        overall_component_results[component] = {
            'biggest_outlier_respondent': biggest_outlier[0],
            'outlier_count': biggest_outlier[1],
            'total_questions': component_total,
            'all_outlier_counts': outlier_counts
        }

    # Overall biggest outlier
    overall_biggest = max(overall_total_outliers.items(), key=lambda x: x[1])
    total_all_components = sum(overall_total_outliers.values())

    return {
        'by_category_and_component': category_results,
        'by_component_overall': overall_component_results,
        'overall': {
            'biggest_outlier_respondent': overall_biggest[0],
            'outlier_count': overall_biggest[1],
            'total_components': total_all_components,
            'all_outlier_counts': overall_total_outliers
        }
    }


def run_gpt4o_mini_analysis(max_questions: int = None):
    """
    Run the GPT-4o-mini category analysis as a standalone function.

    Args:
        max_questions: Maximum number of questions to analyze (None = all)
    """
    print("🤖 Starting GPT-4o-mini Category Analysis")
    print("=" * 50)

    if max_questions:
        print(f"📊 Testing mode: Analyzing up to {max_questions} questions")
    else:
        print("📊 Full analysis mode: Analyzing all questions")

    results = analyze_by_category_with_gpt4o_mini(max_questions=max_questions)

    # Print summary
    print("\\n📊 Analysis Summary:")
    print(f"  Method: {results['analysis_method']}")
    print(f"  Questions analyzed: {results['questions_analyzed']}")
    if max_questions:
        print(f"  Test mode: Limited to {max_questions} questions")
    print(f"  Output: results/rougerx/rougerx_by_category.json")

    if results.get('summary_stats'):
        # Overall component variation
        print(f"\\n📈 Overall Component Variation (Jaccard-based):")
        for comp, stats in results['summary_stats'].get('component_variation_summary', {}).items():
            print(f"  {comp}: avg variation = {stats['avg_variation_score']:.3f}, avg similarity = {stats['avg_jaccard_similarity']:.3f} (n={stats['total_questions']})")

        # Category-specific variation
        print(f"\\n🏥 Category-Specific Variation:")
        categories = ['formal', 'verbal', 'brief']
        for category in categories:
            cat_stats = results['summary_stats'].get('category_variation_summary', {}).get(category, {})
            if cat_stats:
                print(f"  {category.upper()} Question Type:")
                for comp, stats in cat_stats.items():
                    print(f"    {comp}: variation = {stats['avg_variation_score']:.3f}, similarity = {stats['avg_jaccard_similarity']:.3f} (n={stats['total_questions']})")

        # Outlier analysis by category and component
        outlier_analysis = results['summary_stats'].get('outlier_analysis_by_category', {})
        if outlier_analysis:
            print(f"\\n👤 Biggest Outlier Analysis by Category and Component:")

            # Show each category with breakdown by component
            for category in ['formal', 'verbal', 'brief']:
                if category in outlier_analysis.get('by_category_and_component', {}):
                    cat_data = outlier_analysis['by_category_and_component'][category]
                    print(f"  {category.upper()} Category:")

                    # Show each component
                    components = ['drug_name', 'dose', 'unit', 'route', 'frequency']
                    for component in components:
                        if component in cat_data:
                            comp_data = cat_data[component]
                            print(f"    {component}: Respondent {comp_data['biggest_outlier_respondent']} (count: {comp_data['outlier_count']}/{comp_data['total_questions']})")

                    # Show overall for this category
                    if 'overall' in cat_data:
                        overall_cat = cat_data['overall']
                        print(f"    OVERALL: Respondent {overall_cat['biggest_outlier_respondent']} (count: {overall_cat['outlier_count']}/{overall_cat['total_components']})")

            # Show overall by component across all categories
            print(f"\\n  OVERALL by Component (All Categories):")
            if 'by_component_overall' in outlier_analysis:
                for component in ['drug_name', 'dose', 'unit', 'route', 'frequency']:
                    if component in outlier_analysis['by_component_overall']:
                        comp_data = outlier_analysis['by_component_overall'][component]
                        print(f"    {component}: Respondent {comp_data['biggest_outlier_respondent']} (count: {comp_data['outlier_count']}/{comp_data['total_questions']})")

            # Show overall across everything
            overall_data = outlier_analysis.get('overall', {})
            if overall_data:
                print(f"\\n  OVERALL (All Categories & Components):")
                print(f"    Respondent {overall_data['biggest_outlier_respondent']} is the biggest outlier")
                print(f"    Outlier count: {overall_data['outlier_count']} out of {overall_data['total_components']} components")
                print(f"    All counts: {overall_data['all_outlier_counts']}")

    print("\\n✅ GPT-4o-mini analysis completed!")


def test_gpt4o_mini_analysis(n_questions: int = 10):
    """Test GPT-4o-mini analysis on a small subset for development."""
    print(f"🧪 Testing GPT-4o-mini analysis on {n_questions} questions...")
    run_gpt4o_mini_analysis(max_questions=n_questions)

def quick_test():
    """Quick test with just 3 questions."""
    test_gpt4o_mini_analysis(3)

if __name__ == "__main__":
    main()
    # Uncomment one of the lines below for testing:
    # quick_test()                    # Test with 3 questions
    # test_gpt4o_mini_analysis(10)    # Test with 10 questions
    # run_gpt4o_mini_analysis()       # Full analysis
