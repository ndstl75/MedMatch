"""
Survey2 LLM appropriateness evaluation and Percentage Appropriate (%) table.

Runs two prompts per medication route by default (LLM Prompt 1 without format, LLM
Prompt 2 with MedMatch preferred-format text) over the five survey2 CSVs, parses YES/NO
responses, and aggregates into the table: rows = Formal Written Order, Brief Written
Communication, Verbal Communication; columns = Oral, Intravenous, All; cells =
Clinician (optional), LLM %, LLM With MedMatch %.

Usage:

  # OpenAI
  python scripts/survey2gpt5.py --mode openai --model_name gpt-4o-mini

  # Azure OpenAI (set AZURE_OPENAI_ENDPOINT; e.g. GPT-5 Chat deployment)
  python scripts/survey2gpt5.py --mode azure --model_name azure-gpt-5-chat \\
    --max_workers 5 --output_dir results/survey2/gpt5_chat

  # Run both Prompt 1 (no format) and Prompt 2 (MedMatch format)
  python scripts/survey2gpt5.py --prompt_mode both

  # Run MedMatch format prompt only
  python scripts/survey2gpt5.py --mode azure --model_name azure-gpt-5-chat \\
   --max_workers 5 --output_dir results/survey2_v2/gpt5_chat --medmatch_only

  # vLLM
  CUDA_VISIBLE_DEVICES=0,1 python scripts/survey2gpt5.py \\
    --mode vllm --model_name meta-llama/Llama-3.3-70B-Instruct --batch_size 30

  # Skip LLM (recompute table from cache)
  python scripts/survey2gpt5.py --skip_llm

  # Dry run (random YES/NO for testing)
  python scripts/survey2gpt5.py --dry_run

Clinician column: Survey2 CSVs do not contain clinician ratings. Use --clinician_csv PATH
to load precomputed clinician percentages, or leave the column empty (default).
"""

import argparse
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from tqdm import tqdm

# Environment
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Optional providers
try:
    from openai import OpenAI, AzureOpenAI
    OPENAI_AVAILABLE = True
    AZURE_AVAILABLE = True
except ImportError:
    try:
        from openai import OpenAI
        OPENAI_AVAILABLE = True
        AZURE_AVAILABLE = False
    except ImportError:
        OPENAI_AVAILABLE = False
        AZURE_AVAILABLE = False

try:
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    AZURE_IDENTITY_AVAILABLE = True
except ImportError:
    AZURE_IDENTITY_AVAILABLE = False

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


#####################################################################
#                       Constants & configuration                    #
#####################################################################

SURVEY2_BASE = os.path.join(os.path.dirname(__file__), "..", "data", "survey2")
# Outputs (tables, raw results, figures, cache) default to results/survey2 under the repo
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "survey2")

# Single source of truth: each route key must have (1) a CSV in survey2, (2) a Prompt 2 body.
# Mapping: route_key → CSV filename (in survey2 dir) → Prompt 2 title in PROMPT2_BODY_BY_ROUTE
#   oral_solid                      → "(oral solid).csv"           → "Oral solid preferred..."
#   iv_intermittent                 → "(Intravenous intermittent).csv" → "Intravenous intermittent preferred..."
#   iv_push                         → "(Intravenous push).csv"     → "Intravenous push preferred..."
#   iv_continuous_titratable       → "(titratable CI).csv"        → "Intravenous continuous titratable preferred..."
#   iv_continuous_nontitratable     → "(non-titratable CI).csv"    → "Intravenous continuous nontitratable preferred..."
ROUTE_KEYS = (
    "oral_solid",
    "iv_intermittent",
    "iv_push",
    "iv_continuous_titratable",
    "iv_continuous_nontitratable",
)

# Route key → (CSV filename in survey2 dir, table category: Oral | Intravenous)
ROUTE_CONFIG: Dict[str, Tuple[str, str]] = {
    "oral_solid": ("computer-generated survey- for LLM Medmatch(oral solid).csv", "Oral"),
    "iv_intermittent": ("computer-generated survey- for LLM Medmatch(Intravenous intermittent).csv", "Intravenous"),
    "iv_push": ("computer-generated survey- for LLM Medmatch(Intravenous push).csv", "Intravenous"),
    "iv_continuous_titratable": ("computer-generated survey- for LLM Medmatch(titratable CI).csv", "Intravenous"),
    "iv_continuous_nontitratable": ("computer-generated survey- for LLM Medmatch(non-titratable CI).csv", "Intravenous"),
}

# Survey CSV column → normalized communication_type
ORDER_TYPE_MAP = {
    "verbal- green": "verbal",
    "verbal - green": "verbal",
    "formal written-blue": "formal_written",
    "formal written - blue": "formal_written",
    "brief written-orange": "brief_written",
    "brief written - orange": "brief_written",
}

# Table row labels
COMM_TYPE_LABEL = {
    "formal_written": "Formal Written Order",
    "brief_written": "Brief Written Communication",
    "verbal": "Verbal Communication",
}
TABLE_COLUMNS = ["Oral", "Intravenous", "All"]


#####################################################################
#                       Prompt definitions                           #
#####################################################################

# System prompt used for both Prompt 1 and Prompt 2.
SYSTEM_PROMPT = "You are a clinician in the hospital."

# <order> in user templates is replaced by the medication order text from the data CSV (column "Drug") for each row.
# Explicit JSON format so the model outputs parseable JSON.
PROMPT1_USER = """Is the following medication order sentence appropriate if it was a computer-generated order?

### Medication order:
<order>

Respond with only a JSON object. Use exactly: {"answer": "YES"} or {"answer": "NO"}. Do not include any other text, explanation, or formatting."""

# Route key → Prompt 2 user message body (system prompt added in build_prompt2). <order> = order text from data CSV (column "Drug").
PROMPT2_BODY_BY_ROUTE: Dict[str, str] = {
    "oral_solid": """Is the following medication order sentence appropriate if it was a computer-generated order? Clinicians prefer the following components to be included for computer-generated medication orders.

Oral solid preferred computer-generated order format
[drug name][numerical dose][abbreviated unit strength of dose][amount][formulation] by mouth [frequency]

[drug name]: The generic or brand name of the medication.
[numerical dose]: The numeric amount of drug administered per dose, representing the total drug amount for that administration (e.g., 5, 10, 500). For orders with multiple identical tablets or capsules, multiply the per-unit strength by the amount (e.g., 2 capsules of 300 mg each -> numerical dose 600, amount 2).
[abbreviated unit strength of dose]: The standardized abbreviated unit associated with the dose (e.g., mg, mcg, g).
[amount]: The number of dosage units taken per administration (e.g., 1, 2).
[formulation]: The oral solid dosage form (e.g., tablet, capsule, extended-release tablet).
by mouth: The route of administration, fixed as oral.
[frequency]: How often the medication is taken (e.g., once daily, twice daily, every 8 hours).
Respond with only a JSON object. Use exactly: {"answer": "YES"} or {"answer": "NO"}. Do not include any other text, explanation, or formatting.

### Medication order:
<order>""",
    "iv_intermittent": """Is the following medication order sentence appropriate if it was a computer-generated order? Clinicians prefer the following components to be included for computer-generated medication orders.

Intravenous intermittent preferred computer-generated order format
[drug name][numerical dose][abbreviated unit strength of dose][amount of diluent volume][volume unit of measure][compatible diluent type] intravenously infused over [infusion time] [frequency]

[drug name]: The generic or brand name of the medication to be administered intravenously.
[numerical dose]: The numeric value of the drug dose to be given per administration (e.g., 1, 500).
[abbreviated unit strength of dose]: The standardized abbreviated unit associated with the dose (e.g., mg, g, units).
[amount of diluent volume]: The numeric volume of diluent used to prepare the IV medication (e.g., 50, 100).
[volume unit of measure]: The standardized abbreviated unit for diluent volume (e.g., mL).
[compatible diluent type]: The IV fluid used for dilution that is compatible with the medication (e.g., 0.9% sodium chloride, D5W).
intravenous: The fixed route of administration.
infused over: Indicates the medication is administered as an infusion rather than IV push.
[infusion time]: The duration over which the medication is infused (e.g., 30 minutes, 1 hour).
[frequency]: How often the intermittent IV dose is administered (e.g., every 8 hours, once daily).
Respond with only a JSON object. Use exactly: {"answer": "YES"} or {"answer": "NO"}. Do not include any other text, explanation, or formatting.

### Medication order:
<order>""",
    "iv_push": """Is the following medication order sentence appropriate if it was a computer-generated order? Clinicians prefer the following components to be included for computer-generated medication orders.

Intravenous push preferred computer-generated order format
[drug name][numerical dose][abbreviated unit strength of dose][amount of volume][volume unit of measure] of the [concentration of solution][concentration unit of measure][formulation] intravenous push [frequency]

[drug name]: The generic or brand name of the medication administered by IV push.
[numerical dose]: The numeric value of the drug dose delivered per administration (e.g., 2, 10).
[abbreviated unit strength of dose]: The standardized abbreviated unit for the dose (e.g., mg, mcg).
[amount of volume]: The numeric volume administered with the IV push (e.g., 2, 5).
[volume unit of measure]: The standardized abbreviated unit for volume (e.g., mL).
[concentration of solution]: The strength of the drug within the solution (e.g., 2 mg/2 mL).
[concentration unit of measure]: The unit basis used to express the concentration (e.g., mg/mL).
[formulation]: The injectable dosage form (e.g., solution).
intravenous push: The fixed route and method of administration.
[frequency]: How often the IV push dose is administered (e.g., every 6 hours, once).

Respond with only a JSON object. Use exactly: {"answer": "YES"} or {"answer": "NO"}. Do not include any other text, explanation, or formatting.

### Medication order:
<order>""",
    "iv_continuous_titratable": """Is the following medication order sentence appropriate if it was a computer-generated order? Clinicians prefer the following components to be included for computer-generated medication orders.

Intravenous continuous titratable preferred computer-generated order format
[drug name][numerical dose][abbreviated unit strength of dose] "in" [diluent volume][volume unit of measure][compatible diluent type] "continuous intravenous infusion starting at" [starting rate][unit of measure] "titrated by" [titration dose][titration unit of measure] [titration frequency] to achieve a goal of [titration goal]

[drug name]: The generic or brand name of the medication administered as a continuous IV infusion.
[numerical dose]: The numeric amount of drug contained in the prepared infusion (e.g., 50, 250).
[abbreviated unit strength of dose]: The standardized abbreviated unit associated with the dose (e.g., mg, units).
[diluent volume]: The numeric volume of diluent used to prepare the infusion (e.g., 100, 250).
[volume unit of measure]: The standardized abbreviated unit for the diluent volume (e.g., mL).
[compatible diluent type]: The IV fluid used to dilute the medication (e.g., 0.9% sodium chloride, D5W).
continuous intravenous infusion: The fixed route and method of administration.
[starting rate]: The initial infusion rate at which the medication is started (e.g., 0.05, 5).
[unit of measure]: The unit associated with the infusion rate (e.g., mcg/kg/min, units/hr, mL/hr).
titrated by: Indicates dose or rate adjustments are permitted.
[titration dose]: The numeric amount by which the infusion rate is adjusted per titration step (e.g., 0.01, 2).
[titration unit of measure]: The unit associated with the titration increment (e.g., mcg/kg/min, units/hr).
[titration frequency]: The time interval between allowable titrations, expressed in minutes (e.g., 5, 15).
[titration goal based on physiologic response, laboratory result, or assessment score]: The clinical target guiding titration (e.g., MAP ≥ 65 mmHg, RASS score -1 to 1).
Respond with only a JSON object. Use exactly: {"answer": "YES"} or {"answer": "NO"}. Do not include any other text, explanation, or formatting.

### Medication order:
<order>""",
    "iv_continuous_nontitratable": """Is the following medication order sentence appropriate if it was a computer-generated order? Clinicians prefer the following components to be included for computer-generated medication orders.

Intravenous continuous nontitratable preferred computer-generated order format
[drug name][numerical dose][abbreviated unit strength of dose][diluent volume][volume unit of measure]"in"[compatible diluent type] "continuous intravenous infusion at" [rate][unit of measure]

[drug name]: The generic or brand name of the medication administered as a continuous IV infusion.
[numerical dose]: The numeric amount of drug contained in the prepared infusion (e.g., 50, 250).
[abbreviated unit strength of dose]: The standardized abbreviated unit associated with the dose (e.g., mg, units).
[diluent volume]: The numeric volume of diluent used to prepare the infusion (e.g., 100, 250).
[volume unit of measure]: The standardized abbreviated unit for the diluent volume (e.g., mL).
[compatible diluent type]: The IV fluid used to dilute the medication (e.g., 0.9% sodium chloride, D5W).
continuous intravenous infusion: The fixed route and method of administration.
[starting rate]: The initial infusion rate at which the medication is started (e.g., 0.05, 5).
[unit of measure]: The unit associated with the infusion rate (e.g., mcg/kg/min, units/hr, mL/hr).
Respond with only a JSON object. Use exactly: {"answer": "YES"} or {"answer": "NO"}. Do not include any other text, explanation, or formatting.

### Medication order:
<order>""",
}


def build_prompt1(order_text: str) -> List[Dict[str, str]]:
    """LLM Prompt 1: system prompt + user (no MedMatch format). order_text = medication order from data CSV (Drug column)."""
    user_content = PROMPT1_USER.replace("<order>", order_text)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_prompt2(route: str, order_text: str) -> List[Dict[str, str]]:
    """LLM Prompt 2: system prompt + user (with MedMatch preferred format). order_text = medication order from data CSV (Drug column)."""
    if route not in PROMPT2_BODY_BY_ROUTE:
        raise KeyError(
            f"Route '{route}' has no Prompt 2 definition. "
            f"Data and prompts must match. Valid routes: {list(PROMPT2_BODY_BY_ROUTE.keys())}"
        )
    body = PROMPT2_BODY_BY_ROUTE[route]
    user_content = body.replace("<order>", order_text)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def print_first_data_prompts(df: pd.DataFrame) -> None:
    """Print the first row's Prompt 1 and Prompt 2 (for reporting and verification)."""
    if df is None or len(df) == 0:
        return
    first = df.iloc[0]
    order_text = first["order"]
    route = first["route"]
    comm_type = first.get("communication_type", "")
    med_cat = first.get("medication_category", "")
    print("\n" + "=" * 60)
    print("FIRST DATA PROMPTS (first row in dataset)")
    print("=" * 60)
    print(f"Order (from CSV Drug column): {order_text!r}")
    print(f"Route: {route}  |  Communication type: {comm_type}  |  Medication category: {med_cat}")
    print()
    msg1 = build_prompt1(order_text)
    print("--- Prompt 1 (LLM, no MedMatch format) ---")
    for m in msg1:
        role = m.get("role", "user").capitalize()
        content = m.get("content", "")
        print(f"[{role}]\n{content}")
        print()
    msg2 = build_prompt2(route, order_text)
    print("--- Prompt 2 (LLM With MedMatch format) ---")
    for m in msg2:
        role = m.get("role", "user").capitalize()
        content = m.get("content", "")
        # Truncate very long content for display
        if len(content) > 1200:
            content = content[:1200] + "\n... [truncated]"
        print(f"[{role}]\n{content}")
        print()
    print("=" * 60 + "\n")


#####################################################################
#                       Data loading                                #
#####################################################################

def normalize_communication_type(raw: str) -> str:
    """Normalize 'type of order' to formal_written | brief_written | verbal."""
    raw = (raw or "").strip().lower()
    for key, value in ORDER_TYPE_MAP.items():
        if key.lower() in raw or raw == value or raw.replace(" ", "") == key.replace(" ", ""):
            return value
    base = re.sub(r"\s*-\s*(green|blue|orange)\s*$", "", raw, flags=re.IGNORECASE).strip()
    base = base.replace(" ", "_").lower()
    if "verbal" in base:
        return "verbal"
    if "formal" in base or ("written" in base and "brief" not in base):
        return "formal_written"
    if "brief" in base:
        return "brief_written"
    return base or "verbal"


def _ensure_data_prompt_match() -> None:
    """Ensure every route in ROUTE_CONFIG has a Prompt 2 body and vice versa."""
    config_keys = set(ROUTE_CONFIG.keys())
    prompt_keys = set(PROMPT2_BODY_BY_ROUTE.keys())
    if config_keys != prompt_keys:
        missing_in_prompt = config_keys - prompt_keys
        missing_in_config = prompt_keys - config_keys
        raise ValueError(
            "Data and prompts must use the same route keys. "
            f"Routes in ROUTE_CONFIG but not in PROMPT2_BODY_BY_ROUTE: {missing_in_prompt or 'none'}. "
            f"Routes in PROMPT2_BODY_BY_ROUTE but not in ROUTE_CONFIG: {missing_in_config or 'none'}."
        )


def load_survey2_data(data_dir: str) -> pd.DataFrame:
    """Load all five survey2 CSVs; normalize type of order; attach route and medication_category. Order text = 'Drug' column."""
    rows: List[Dict[str, Any]] = []
    for route_key, (filename, category) in ROUTE_CONFIG.items():
        path = os.path.join(data_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Survey2 CSV not found: {path}")
        df = pd.read_csv(path)
        if "Drug" not in df.columns or "type of order" not in df.columns:
            raise ValueError(f"Expected columns 'Drug' and 'type of order' in {path}")
        for _, r in df.iterrows():
            order_text = (r["Drug"] if pd.notna(r["Drug"]) else "").strip()  # Data: order sentence = Drug column
            raw_type = (r["type of order"] if pd.notna(r["type of order"]) else "").strip()
            comm_type = normalize_communication_type(raw_type)
            # route_key links this row to PROMPT2_BODY_BY_ROUTE[route] so Prompt 2 matches the CSV source
            rows.append({
                "order": order_text,
                "communication_type": comm_type,
                "route": route_key,
                "medication_category": category,
            })
    return pd.DataFrame(rows)


#####################################################################
#                       LLM clients & invocation                    #
#####################################################################

def get_openai_client() -> "OpenAI":
    """Create OpenAI client (requires OPENAI_API_KEY)."""
    if not OPENAI_AVAILABLE:
        raise ImportError("openai package not available. Install with: pip install openai")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found in environment.")
    return OpenAI(api_key=api_key)


def get_azure_client(model_name: str) -> Tuple[Any, str]:
    """
    Create Azure OpenAI client and deployment name.
    model_name: e.g. 'azure-gpt-5-chat' or 'gpt-5-chat' (with --mode azure).
    Returns (client, deployment_name).
    """
    if not AZURE_AVAILABLE or not AZURE_IDENTITY_AVAILABLE:
        raise ImportError(
            "Azure OpenAI mode requires: pip install azure-identity openai"
        )
    if not model_name.startswith("azure-"):
        model_name = f"azure-{model_name}"
    deployment = model_name.replace("azure-", "")
    endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip().rstrip("/")
    if not endpoint:
        raise EnvironmentError(
            "Set AZURE_OPENAI_ENDPOINT to your Azure OpenAI HTTPS endpoint "
            "(e.g. https://<resource-name>.openai.azure.com)"
        )
    credential = DefaultAzureCredential(exclude_managed_identity_credential=True)
    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version="2024-02-15-preview",
    )
    logging.info("Using Azure OpenAI deployment: %s", deployment)
    return client, deployment


def run_chat(
    client: Any,
    model_param: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 20,
) -> str:
    """Single chat completion (OpenAI or Azure OpenAI)."""
    resp = client.chat.completions.create(
        model=model_param,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Flatten chat messages into a single prompt for vLLM."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user").capitalize()
        parts.append(f"{role}: {msg.get('content', '')}")
    parts.append("Assistant:")
    return "\n".join(parts)


def run_vllm(
    llm: "LLM",
    sampling_params: "SamplingParams",
    messages: List[Dict[str, str]],
) -> str:
    """Single vLLM generation."""
    prompt = messages_to_prompt(messages)
    outputs = llm.generate([prompt], sampling_params=sampling_params)
    return (outputs[0].outputs[0].text or "").strip()


def parse_yes_no_json(response: str) -> Optional[bool]:
    """Parse YES/NO from JSON (keys: answer, appropriate, response). If no valid JSON, fall back to plain YES/NO in first line."""
    raw = (response or "").strip()
    # Try JSON first
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            obj = json.loads(raw[start:end])
            for key in ("answer", "appropriate", "response"):
                if key in obj:
                    val = obj[key]
                    if isinstance(val, bool):
                        return val
                    if isinstance(val, str):
                        return val.upper().strip() == "YES"
        except json.JSONDecodeError:
            pass
    # Fallback: plain YES/NO in first line or whole response (e.g. model didn't output JSON)
    first_line = raw.split("\n")[0].strip().upper()
    if first_line in ("YES", "NO"):
        return first_line == "YES"
    if raw.upper().strip() in ("YES", "NO"):
        return raw.upper().strip() == "YES"
    # Check if response starts with YES or NO (with possible punctuation)
    for prefix in ("YES", "NO"):
        if first_line.startswith(prefix) or raw.upper().startswith(prefix):
            return prefix == "YES"
    return None


#####################################################################
#                       Pipeline (run one order, run all)           #
#####################################################################

def run_one_order(
    row: Dict[str, Any],
    mode: str,
    client: Optional[Any],
    model_param: str,
    llm: Optional[Any],
    sampling_params: Optional[Any],
    temperature: float,
    max_tokens: int,
    prompt_mode: str = "both",
) -> Dict[str, Any]:
    """Run prompts for one order; default is Prompt 2 (MedMatch format) only."""
    order_text = row["order"]
    route = row["route"]
    out = dict(row)

    r1 = None
    if prompt_mode == "both":
        msg1 = build_prompt1(order_text)
        if mode in ("openai", "azure") and client is not None:
            r1 = run_chat(client, model_param, msg1, temperature=temperature, max_tokens=max_tokens)
        elif mode == "vllm" and llm is not None and sampling_params is not None:
            r1 = run_vllm(llm, sampling_params, msg1)
        else:
            r1 = ""
        out["response_p1"] = r1
        out["llm_appropriate"] = parse_yes_no_json(r1)
    else:
        out["response_p1"] = None
        out["llm_appropriate"] = None

    msg2 = build_prompt2(route, order_text)
    if mode in ("openai", "azure") and client is not None:
        r2 = run_chat(client, model_param, msg2, temperature=temperature, max_tokens=max_tokens)
    elif mode == "vllm" and llm is not None and sampling_params is not None:
        r2 = run_vllm(llm, sampling_params, msg2)
    else:
        r2 = ""
    out["response_p2"] = r2
    out["llm_medmatch_appropriate"] = parse_yes_no_json(r2)
    return out


def run_llm_pipeline(
    df: pd.DataFrame,
    mode: str,
    model_name: str,
    client: Optional[Any] = None,
    model_param: Optional[str] = None,
    llm: Optional[Any] = None,
    sampling_params: Optional[Any] = None,
    batch_size: int = 10,
    max_workers: Optional[int] = None,
    temperature: float = 0.0,
    max_tokens: int = 20,
    prompt_mode: str = "both",
) -> pd.DataFrame:
    """Run selected prompt mode for each order. OpenAI/Azure: parallel (max_workers). vLLM: sequential."""
    rows = df.to_dict("records")
    results: List[Dict[str, Any]] = []
    param = model_param or model_name

    if mode in ("openai", "azure") and client is not None:
        workers = max(1, max_workers or (1 if mode == "azure" else 10))
        workers = min(workers, len(rows))
        desc = f"LLM ({'Azure' if mode == 'azure' else 'OpenAI'}, {workers} workers)"
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    run_one_order, r, mode, client, param, llm, sampling_params, temperature, max_tokens, prompt_mode
                ): r
                for r in rows
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                results.append(future.result())
    elif mode == "vllm" and llm is not None and sampling_params is not None:
        for r in tqdm(rows, desc="LLM (vLLM)"):
            results.append(
                run_one_order(r, mode, None, param, llm, sampling_params, temperature, max_tokens, prompt_mode)
            )
    else:
        for r in tqdm(rows, desc="LLM"):
            results.append(
                run_one_order(r, mode, client, param, llm, sampling_params, temperature, max_tokens, prompt_mode)
            )
    return pd.DataFrame(results)


#####################################################################
#                       Aggregation & table                         #
#####################################################################

def aggregate_and_build_table(
    df: pd.DataFrame,
    clinician_pct: Optional[Dict[Tuple[str, str], float]] = None,
) -> pd.DataFrame:
    """Build Percentage Appropriate (%) table. Rows: comm type; Columns: Oral, Intravenous, All; cells: Clinician %, LLM %, LLM With MedMatch %."""
    clinician_pct = clinician_pct or {}
    llm_yes: Dict[Tuple[str, str], int] = {}
    llm_total: Dict[Tuple[str, str], int] = {}
    medmatch_yes: Dict[Tuple[str, str], int] = {}
    medmatch_total: Dict[Tuple[str, str], int] = {}

    for _, r in df.iterrows():
        comm, cat = r["communication_type"], r["medication_category"]
        key = (comm, cat)
        llm_total[key] = llm_total.get(key, 0) + 1
        medmatch_total[key] = medmatch_total.get(key, 0) + 1
        if r.get("llm_appropriate") is True:
            llm_yes[key] = llm_yes.get(key, 0) + 1
        if r.get("llm_medmatch_appropriate") is True:
            medmatch_yes[key] = medmatch_yes.get(key, 0) + 1

    for comm in ("formal_written", "brief_written", "verbal"):
        for cat in ("Oral", "Intravenous"):
            k, k_all = (comm, cat), (comm, "All")
            llm_total[k_all] = llm_total.get(k_all, 0) + llm_total.get(k, 0)
            llm_yes[k_all] = llm_yes.get(k_all, 0) + llm_yes.get(k, 0)
            medmatch_total[k_all] = medmatch_total.get(k_all, 0) + medmatch_total.get(k, 0)
            medmatch_yes[k_all] = medmatch_yes.get(k_all, 0) + medmatch_yes.get(k, 0)

    table_rows: List[Dict[str, Any]] = []
    for comm_key, row_label in COMM_TYPE_LABEL.items():
        row_data: Dict[str, Any] = {"Communication type": row_label}
        for col in TABLE_COLUMNS:
            key = (comm_key, col)
            n = llm_total.get(key, 0)
            llm_pct = (100.0 * llm_yes.get(key, 0) / n) if n else None
            mm_pct = (100.0 * medmatch_yes.get(key, 0) / n) if n else None
            cell = {
                "Clinician (%)": clinician_pct.get(key),
                "LLM (%)": round(llm_pct, 1) if llm_pct is not None else None,
                "LLM With MedMatch (%)": round(mm_pct, 1) if mm_pct is not None else None,
            }
            row_data[col] = cell
        table_rows.append(row_data)
    return pd.DataFrame(table_rows)


def load_clinician_percentages(path: str) -> Dict[Tuple[str, str], float]:
    """Load optional clinician percentages from CSV (communication_type, medication_category, percentage)."""
    df = pd.read_csv(path)
    out: Dict[Tuple[str, str], float] = {}
    comm_col = "communication_type" if "communication_type" in df.columns else None
    cat_col = "medication_category" if "medication_category" in df.columns else None
    pct_col = next((c for c in ("percentage", "Percentage", "value", "pct") if c in df.columns), None)
    if pct_col is None and len(df.columns) >= 3:
        pct_col = df.columns[2]
    if not comm_col or not cat_col or not pct_col:
        return out
    for _, r in df.iterrows():
        comm = str(r[comm_col]).strip().lower().replace(" ", "_")
        cat = str(r[cat_col]).strip()
        if cat not in TABLE_COLUMNS:
            continue
        try:
            out[(comm, cat)] = float(r[pct_col])
        except (TypeError, ValueError):
            continue
    return out


def _fmt_pct(x: Any) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.1f}"
    except (TypeError, ValueError):
        return str(x)


def print_table(table_df: pd.DataFrame) -> None:
    """Print Percentage Appropriate (%) table to stdout."""
    print("\n--- Percentage Appropriate (%) ---\n")
    for _, row in table_df.iterrows():
        print(row["Communication type"])
        for col in TABLE_COLUMNS:
            cell = row[col]
            if isinstance(cell, dict):
                c = _fmt_pct(cell.get("Clinician (%)"))
                l = _fmt_pct(cell.get("LLM (%)"))
                m = _fmt_pct(cell.get("LLM With MedMatch (%)"))
                print(f"  {col}: Clinician {c}%  |  LLM {l}%  |  LLM With MedMatch {m}%")
        print()


def write_results_table(output_dir: str, table_df: pd.DataFrame, flat: bool = True) -> str:
    """Write table to CSV. If flat=True, flatten cell dicts into columns like Oral_LLM (%)."""
    os.makedirs(output_dir, exist_ok=True)
    if flat:
        flat_rows = []
        for _, row in table_df.iterrows():
            r = {"Communication type": row["Communication type"]}
            for col in TABLE_COLUMNS:
                cell = row[col]
                if isinstance(cell, dict):
                    for k, v in cell.items():
                        r[f"{col}_{k}"] = v
            flat_rows.append(r)
        out_df = pd.DataFrame(flat_rows)
    else:
        out_df = table_df
    path = os.path.join(output_dir, "appropriateness_table.csv")
    out_df.to_csv(path, index=False)
    return path


def write_raw_results_csv(output_dir: str, df: pd.DataFrame) -> Optional[str]:
    """Write raw per-order results to raw_results.csv. Only writes if df has response columns."""
    cols = ["order", "communication_type", "route", "medication_category"]
    optional = ["response_p1", "response_p2", "llm_appropriate", "llm_medmatch_appropriate"]
    for c in optional:
        if c in df.columns:
            cols.append(c)
    if not all(c in df.columns for c in ["order", "communication_type", "route", "medication_category"]):
        return None
    out_df = df[cols].copy()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "raw_results.csv")
    out_df.to_csv(path, index=False, encoding="utf-8")
    return path


def table_to_long(table_df: pd.DataFrame) -> pd.DataFrame:
    """Convert Percentage Appropriate table to long format: communication_type, medication_category, clinician_pct, llm_pct, llm_medmatch_pct."""
    rows = []
    for _, row in table_df.iterrows():
        comm_label = row["Communication type"]
        for col in TABLE_COLUMNS:
            cell = row[col]
            if isinstance(cell, dict):
                rows.append({
                    "communication_type": comm_label,
                    "medication_category": col,
                    "clinician_pct": cell.get("Clinician (%)"),
                    "llm_pct": cell.get("LLM (%)"),
                    "llm_medmatch_pct": cell.get("LLM With MedMatch (%)"),
                })
    return pd.DataFrame(rows)


def write_percentage_long_csv(output_dir: str, table_df: pd.DataFrame) -> str:
    """Write long-format percentage table: communication_type, medication_category, clinician_pct, llm_pct, llm_medmatch_pct."""
    os.makedirs(output_dir, exist_ok=True)
    long_df = table_to_long(table_df)
    path = os.path.join(output_dir, "percentage_appropriate_long.csv")
    long_df.to_csv(path, index=False)
    return path


def plot_percentage_figure(
    output_dir: str,
    table_df: pd.DataFrame,
    n_per_communication_type: Optional[Dict[str, int]] = None,
) -> List[str]:
    """
    Save Figure 2-style grouped bar chart: Percentage Appropriate (%) by communication type
    and medication category (Oral, Intravenous, All). One figure for LLM, one for LLM With MedMatch.
    Returns paths to saved files.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logging.warning("matplotlib not available; skipping figure generation. Install with: pip install matplotlib")
        return []

    long_df = table_to_long(table_df)
    if long_df.empty:
        return []

    comm_order = [COMM_TYPE_LABEL["formal_written"], COMM_TYPE_LABEL["brief_written"], COMM_TYPE_LABEL["verbal"]]
    cat_order = ["Oral", "Intravenous", "All"]
    n_per_communication_type = n_per_communication_type or {}

    x = np.arange(len(comm_order))
    width = 0.25
    paths_saved = []

    for evaluator, pct_col, title_suffix, filename in [
        ("LLM", "llm_pct", "LLM (Prompt 1)", "figure_appropriate_llm.png"),
        ("LLM With MedMatch", "llm_medmatch_pct", "LLM With MedMatch (Prompt 2)", "figure_appropriate_llm_medmatch.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, cat in enumerate(cat_order):
            vals = []
            for comm in comm_order:
                row = long_df[(long_df["communication_type"] == comm) & (long_df["medication_category"] == cat)]
                val = row[pct_col].iloc[0] if len(row) and pd.notna(row[pct_col].iloc[0]) else 0.0
                vals.append(float(val))
            offset = (i - 1) * width
            bars = ax.bar(x + offset, vals, width, label=cat)
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.1f}%", ha="center", va="bottom", fontsize=8)

        ax.set_ylabel("Percentage")
        ax.set_ylim(0, 105)
        ax.set_xticks(x)
        labels = []
        for comm in comm_order:
            n = n_per_communication_type.get(comm)
            labels.append(f"{comm}\n(n={n})" if n is not None else comm)
        ax.set_xticklabels(labels)
        ax.set_title(f"Percentage Appropriate (%) – {title_suffix}")
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths_saved.append(path)

    # Optional: combined figure with Clinician, LLM, LLM With MedMatch (if clinician data present)
    has_clinician = long_df["clinician_pct"].notna().any()
    if has_clinician:
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, (evaluator, pct_col) in enumerate([
            ("Clinician", "clinician_pct"),
            ("LLM", "llm_pct"),
            ("LLM With MedMatch", "llm_medmatch_pct"),
        ]):
            vals = []
            for comm in comm_order:
                row = long_df[(long_df["communication_type"] == comm) & (long_df["medication_category"] == "All")]
                val = row[pct_col].iloc[0] if len(row) and pd.notna(row[pct_col].iloc[0]) else 0.0
                vals.append(float(val))
            offset = (i - 1) * width
            ax.bar(x + offset, vals, width, label=evaluator)
        ax.set_ylabel("Percentage")
        ax.set_ylim(0, 105)
        ax.set_xticks(x)
        labels = [f"{c}\n(n={n_per_communication_type.get(c)})" if n_per_communication_type.get(c) else c for c in comm_order]
        ax.set_xticklabels(labels)
        ax.set_title("Percentage Appropriate (%) – All Medications by Evaluator")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        path = os.path.join(output_dir, "figure_appropriate_all_evaluators.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths_saved.append(path)

    return paths_saved


#####################################################################
#                       Cache (per-order results)                   #
#####################################################################

def get_cache_path(output_dir: str) -> str:
    return os.path.join(output_dir, "survey2_per_order_results.jsonl")


def load_cached_results(output_dir: str) -> Optional[pd.DataFrame]:
    path = get_cache_path(output_dir)
    if not os.path.isfile(path):
        return None
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(records) if records else None


def save_cached_results(output_dir: str, df: pd.DataFrame) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = get_cache_path(output_dir)
    with open(path, "w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            f.write(json.dumps(dict(r), ensure_ascii=False) + "\n")
    return path


#####################################################################
#                       Validation                                  #
#####################################################################

def validate_args(args: argparse.Namespace) -> None:
    """Validate configuration for the selected mode."""
    if args.mode == "openai":
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI mode requires: pip install openai")
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY not found in environment.")
    elif args.mode == "azure":
        if not AZURE_AVAILABLE or not AZURE_IDENTITY_AVAILABLE:
            raise ImportError("Azure mode requires: pip install azure-identity openai")
        if not os.getenv("AZURE_OPENAI_ENDPOINT"):
            logging.warning("AZURE_OPENAI_ENDPOINT not set; using default endpoint.")
    elif args.mode == "vllm":
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM mode requires: pip install vllm")
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Must be 'openai', 'azure', or 'vllm'.")


#####################################################################
#                       Main                                        #
#####################################################################

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Survey2 LLM appropriateness: run both prompts by default, aggregate Percentage Appropriate (%) table."
    )
    parser.add_argument("--data_dir", default=SURVEY2_BASE, help="Directory containing survey2 CSVs.")
    parser.add_argument("--mode", choices=["openai", "azure", "vllm"], default="openai",
                        help="openai | azure (Azure OpenAI, e.g. azure-gpt-5-chat) | vllm.")
    parser.add_argument("--model_name", default="gpt-4o-mini",
                        help="Model or deployment name (e.g. gpt-4o-mini, azure-gpt-5-chat).")
    parser.add_argument(
        "--prompt_mode",
        choices=["medmatch_only", "both"],
        default="both",
        help="Prompt execution mode: both (default, Prompt 1 + Prompt 2) or medmatch_only.",
    )
    parser.add_argument(
        "--medmatch_only",
        action="store_true",
        help="Dedicated shortcut for MedMatch-format prompts only (equivalent to --prompt_mode medmatch_only).",
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for table and cache.")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size / concurrency hint.")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Max parallel workers for OpenAI/Azure (default: 1 for Azure, 10 for OpenAI).")
    parser.add_argument("--skip_llm", action="store_true",
                        help="Skip LLM calls; recompute table from cached per-order results.")
    parser.add_argument("--dry_run", action="store_true", help="Build table from random YES/NO (no API calls).")
    parser.add_argument("--clinician_csv", default=None, help="Optional CSV with clinician percentages.")
    parser.add_argument("--no_figure", action="store_true", help="Do not generate bar chart figures (CSV and raw results still written).")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=20)
    args = parser.parse_args()
    if args.medmatch_only:
        args.prompt_mode = "medmatch_only"
    return args


def main() -> None:
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)

    # Display configuration
    print("\n" + "=" * 60)
    print("SURVEY2 LLM APPROPRIATENESS")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model_name}")
    print(f"Prompt mode: {args.prompt_mode}")
    if args.mode == "azure":
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "(default)")
        print(f"Azure endpoint: {endpoint}")
    elif args.mode == "vllm":
        print(f"Batch size: {args.batch_size}")
    if args.mode in ("openai", "azure"):
        max_workers = args.max_workers if args.max_workers is not None else (1 if args.mode == "azure" else 10)
        print(f"Max workers: {max_workers}")
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Temperature: {args.temperature}, Max tokens: {args.max_tokens}")
    print("=" * 60 + "\n")

    validate_args(args)
    _ensure_data_prompt_match()
    df = load_survey2_data(data_dir)

    # Optional: load from cache
    if args.skip_llm:
        cached = load_cached_results(output_dir)
        if cached is not None and len(cached) == len(df):
            df = cached
            print("Using cached per-order results (--skip_llm).")
        else:
            print("No cache or size mismatch; running LLM anyway.")
            args.skip_llm = False

    # Report first data prompt(s)
    if args.prompt_mode == "both":
        print_first_data_prompts(df)
    elif df is not None and len(df) > 0:
        first = df.iloc[0]
        order_text = first["order"]
        route = first["route"]
        comm_type = first.get("communication_type", "")
        med_cat = first.get("medication_category", "")
        print("\n" + "=" * 60)
        print("FIRST DATA PROMPT (first row in dataset)")
        print("=" * 60)
        print(f"Order (from CSV Drug column): {order_text!r}")
        print(f"Route: {route}  |  Communication type: {comm_type}  |  Medication category: {med_cat}")
        print()
        msg2 = build_prompt2(route, order_text)
        print("--- Prompt 2 (LLM With MedMatch format) ---")
        for m in msg2:
            role = m.get("role", "user").capitalize()
            content = m.get("content", "")
            if len(content) > 1200:
                content = content[:1200] + "\n... [truncated]"
            print(f"[{role}]\n{content}")
            print()
        print("=" * 60 + "\n")

    if args.dry_run:
        import random
        random.seed(42)
        if args.prompt_mode == "both":
            df["llm_appropriate"] = [random.choice([True, False]) for _ in range(len(df))]
        else:
            df["llm_appropriate"] = [None for _ in range(len(df))]
        df["llm_medmatch_appropriate"] = [random.choice([True, False]) for _ in range(len(df))]
        print("Dry run: using random YES/NO for table build.")
    elif not args.skip_llm:
        client, model_param = None, args.model_name
        llm, sampling_params = None, None

        if args.mode == "openai":
            client = get_openai_client()
        elif args.mode == "azure":
            client, model_param = get_azure_client(args.model_name)
        else:
            llm = LLM(
                model=args.model_name,
                trust_remote_code=True,
                gpu_memory_utilization=0.85,
                max_model_len=4096,
            )
            sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

        df = run_llm_pipeline(
            df,
            mode=args.mode,
            model_name=args.model_name,
            client=client,
            model_param=model_param,
            llm=llm,
            sampling_params=sampling_params,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            prompt_mode=args.prompt_mode,
        )
        save_cached_results(output_dir, df)

    clinician_pct = None
    if args.clinician_csv and os.path.isfile(args.clinician_csv):
        clinician_pct = load_clinician_percentages(args.clinician_csv)

    table_df = aggregate_and_build_table(df, clinician_pct=clinician_pct)
    print_table(table_df)

    # CSV outputs
    out_path = write_results_table(output_dir, table_df)
    print(f"Table written to {out_path}")
    long_path = write_percentage_long_csv(output_dir, table_df)
    print(f"Long-format table written to {long_path}")

    # Raw per-order results
    raw_path = write_raw_results_csv(output_dir, df)
    if raw_path:
        print(f"Raw results written to {raw_path}")

    # Figure (Figure 2-style grouped bar chart)
    if not args.no_figure:
        n_per_comm = df.groupby("communication_type").size()
        n_per_communication_type = {COMM_TYPE_LABEL[k]: int(v) for k, v in n_per_comm.items()}
        figure_paths = plot_percentage_figure(output_dir, table_df, n_per_communication_type=n_per_communication_type)
        for p in figure_paths:
            print(f"Figure written to {p}")
    else:
        print("Skipping figure (--no_figure).")

    print()


if __name__ == "__main__":
    main()
