#!/usr/bin/env python3
"""
MedMatch Single Test — reads input.txt, sends to a locally hosted Gemma model,
runs both zero-shot and one-shot, outputs parsed JSON + Jaccard.

input.txt format:
  Line 1: route type (PO Solid, PO Liquid, IV Intermittent, IV Push, IV Continuous)
  Line 2: medication order sentence
  Line 3+: (optional) ground truth JSON for Jaccard comparison
"""

import json
import os
import re
import sys
from local_llm import OLLAMA_MODEL, chat_completion

MODEL_NAME = os.environ.get("OLLAMA_MODEL", OLLAMA_MODEL)
TEMPERATURE = 0.1

SYSTEM_PROMPT = (
    "You are a clinical pharmacist who formats medication orders. "
    "Only output the MedMatch JSON format."
)

EXPECTED_KEYS = {
    "PO Solid": ["drug name", "numerical dose", "abbreviated unit strength of dose",
                  "amount", "formulation", "route", "frequency"],
    "PO Liquid": ["drug name", "numerical dose", "abbreviated unit strength of dose",
                   "numerical volume", "volume unit of measure", "concentration of formulation",
                   "formulation unit of measure", "formulation", "route", "frequency"],
    "IV Intermittent": ["drug name", "numerical dose", "abbreviated unit strength of dose",
                         "amount of diluent volume", "volume unit of measure",
                         "compatible diluent type", "infusion time", "frequency"],
    "IV Push": ["drug name", "numerical dose", "abbreviated unit strength of dose",
                 "amount of volume", "volume unit of measure", "concentration of solution",
                 "concentration unit of measure", "formulation", "frequency"],
    "IV Continuous": ["drug name", "numerical dose", "abbreviated unit strength of dose",
                       "diluent volume", "volume unit of measure", "compatible diluent type",
                       "starting rate", "unit of measure", "titration dose",
                       "titration unit of measure", "titration frequency",
                       "titration goal based on physiologic response, laboratory result, or assessment score"],
}

# ── Zero-shot instructions (format definition only, no example) ───────────────

ZERO_SHOT = {
    "PO Solid": """Please review the narratives about medications and format them into the MedMatch JSON format. Follow this exact slot order; if a slot is unknown, use an empty string and do not fabricate.

The MedMatch JSON format for oral solid dosage form medications is:
[drug name][numerical dose][abbreviated unit strength of dose][amount][formulation] by mouth [frequency]

[drug name]: The generic or brand name of the medication.
[numerical dose]: The numeric amount of drug administered per dose, representing the total drug amount for that administration (e.g., 5, 10, 500). For orders with multiple identical tablets or capsules, multiply the per-unit strength by the amount (e.g., 2 capsules of 500 mg each -> numerical dose 1000, amount 2).
[abbreviated unit strength of dose]: The standardized abbreviated unit associated with the dose (e.g., mg, mcg, g).
[amount]: The number of dosage units taken per administration (e.g., 1, 2).
[formulation]: The oral solid dosage form (e.g., tablet, capsule, extended-release tablet). Copy the dosage-form wording from the order as closely as possible, including qualifiers such as extended-release or delayed-release.
by mouth: The route of administration, fixed as oral.
[frequency]: How often the medication is taken (e.g., once daily, twice daily, every 8 hours). Preserve the full schedule phrase, including qualifiers such as as needed, at bedtime, or indication text if present.
""",

    "PO Liquid": """Please review the narratives about medications and format them into the MedMatch JSON format. Follow this exact slot order; if a slot is unknown, use an empty string and do not fabricate.

The MedMatch JSON format for oral liquid dosage form medications is:
[drug name][numerical dose][abbreviated unit strength of dose][numerical volume][abbreviated unit strength of volume] of the [concentration of formulation][formulation unit of measure][formulation] by mouth [frequency]

[drug name]: The generic or brand name of the medication.
[numerical dose]: The numeric amount of drug administered per dose (e.g., 250, 5).
[abbreviated unit strength of dose]: The standardized abbreviated unit for the drug dose (e.g., mg, mcg).
[numerical volume]: The numeric volume administered per dose (e.g., 5, 10).
[abbreviated unit strength of volume]: The standardized abbreviated unit for volume (e.g., mL).
[concentration of formulation]: The strength of the medication per 1 mL. For example, if the sentence says 250 mg/5 mL, write 50 here.
[formulation unit of measure]: The unit used in the concentration denominator (e.g., mL).
[formulation]: The oral liquid dosage form (e.g., solution, suspension, syrup).
[route]: The route of administration, fixed as oral.
[frequency]: How often the medication is administered (e.g., once daily, every 6 hours).
""",

    "IV Intermittent": """Please review the narratives about medications and format them into the MedMatch JSON format. Follow this exact slot order; if a slot is unknown, use an empty string and do not fabricate.

The MedMatch JSON format for IV intermittent dosage form medications is:
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
""",

    "IV Push": """Please review the narratives about medications and format them into the MedMatch JSON format. Follow this exact slot order; if a slot is unknown, use an empty string and do not fabricate.

The MedMatch JSON format for IV push dosage form medications is:
[drug name][numerical dose][abbreviated unit strength of dose][amount of volume][volume unit of measure] of the [concentration of solution][concentration unit of measure][formulation] intravenous push [frequency]

[drug name]: The generic or brand name of the medication administered by IV push.
[numerical dose]: The numeric value of the drug dose delivered per administration (e.g., 2, 10).
[abbreviated unit strength of dose]: The standardized abbreviated unit for the dose (e.g., mg, mcg).
[amount of volume]: The numeric volume administered with the IV push (e.g., 2, 5).
[volume unit of measure]: The standardized abbreviated unit for volume (e.g., mL).
[concentration of solution]: The strength of the drug within the solution, normalized to per 1 mL (e.g., 10 if the source says 20 mg/2 mL).
[concentration unit of measure]: The unit basis used to express the concentration (e.g., mg/mL).
[formulation]: The injectable dosage form (e.g., solution).
intravenous push: The fixed route and method of administration.
[frequency]: How often the IV push dose is administered (e.g., every 6 hours, once).
""",

    "IV Continuous": """Please review the narratives about medications and format them into the MedMatch JSON format. Follow this exact slot order; if a slot is unknown, use an empty string and do not fabricate.

The MedMatch JSON format for IV continuous dosage form medications is:

Titratable IV CI:
[drug name][numerical dose][abbreviated unit strength of dose] "in" [diluent volume][volume unit of measure][compatible diluent type] "continuous intravenous infusion starting at" [starting rate][unit of measure] "titrated by" [titration dose][titration unit of measure] [titration frequency] to achieve a goal [titration goal based on physiologic response, laboratory result, or assessment score]

Non-titratable IV CI:
[drug name][numerical dose][abbreviated unit strength of dose][diluent volume][volume unit of measure] "in" [compatible diluent type] "continuous intravenous infusion at" [rate][unit of measure]

Definitions:
[drug name]: The generic or brand name of the medication administered as a continuous IV infusion.
[numerical dose]: The total amount of drug contained in the prepared infusion bag or bottle, not the per-mL concentration. For example, if the order says 200 mg/100 mL, write 200 here. If the order says 1 mg/mL in 100 mL, write 100 here.
[abbreviated unit strength of dose]: The standardized abbreviated unit associated with the dose (e.g., mg, units).
[diluent volume]: The total prepared volume of the infusion bag or bottle (e.g., 100, 250). Do not write 0 when the volume is implied by a concentration such as 1 mg/mL in 100 mL.
[volume unit of measure]: The standardized abbreviated unit for the diluent volume (e.g., mL).
[compatible diluent type]: The IV fluid used to dilute the medication (e.g., 0.9% sodium chloride, D5W).
continuous intravenous infusion: The fixed route and method of administration.
[starting rate]: The initial infusion rate at which the medication is started, or the fixed infusion rate if the infusion is non-titratable (e.g., 0.05, 5).
[unit of measure]: The unit associated with the infusion rate (e.g., mcg/kg/min, units/hr, mL/hr).
[titration dose]: The numeric amount by which the infusion rate is adjusted per titration step (e.g., 0.01, 2).
[titration unit of measure]: The unit associated with the titration increment (e.g., mcg/kg/min, units/hr).
[titration frequency]: The time interval between allowable titrations (e.g., every 5 minutes, every 15 minutes).
[titration goal]: Copy the clinical target guiding titration as closely as possible from the sentence instead of paraphrasing (e.g., MAP >= 65 mmHg, RASS score -1 to 1).
""",
}

# ── One-shot instructions (format definition + example from paper appendix) ───

ONE_SHOT = {
    "PO Solid": ZERO_SHOT["PO Solid"] + """
Example of input:
Gabapentin 2 capsules (total dose 600mg) by mouth three times daily.
Example of MedMatch JSON format:
{ "drug name": "gabapentin",
"numerical dose": 600,
"abbreviated unit strength of dose": "mg",
"amount": 2,
"formulation": "capsules",
"route": "by mouth",
"frequency": "three times daily"}

Example of preserving full frequency text while keeping total dose:
Give the patient a total dose of acetaminophen 1000mg by giving 2 x 500mg tablets by mouth every 8 hours as needed for pain.
Example of MedMatch JSON format:
{ "drug name": "acetaminophen",
"numerical dose": 1000,
"abbreviated unit strength of dose": "mg",
"amount": 2,
"formulation": "tablets",
"route": "by mouth",
"frequency": "every 8 hours as needed for pain"}
""",

    "PO Liquid": ZERO_SHOT["PO Liquid"] + """
Example of input:
An oral solution of diazepam (1mg/mL) is used to administer 5mg (5mL) dose by mouth once daily.
Example of MedMatch JSON Format:
{ "drug name": "diazepam",
"numerical dose": 5,
"abbreviated unit strength of dose": "mg",
"numerical volume": 5,
"volume unit of measure": "mL",
"concentration of formulation": 1,
"formulation unit of measure": "mg/mL",
"formulation": "solution",
"route": "by mouth",
"frequency": "once daily"}

Example of concentration conversion:
Using a 250mg/5mL solution, administer a dose of 500mg of valproic acid (10mL) every 6 hours by mouth.
Example of MedMatch JSON Format:
{ "drug name": "valproic acid",
"numerical dose": 500,
"abbreviated unit strength of dose": "mg",
"numerical volume": 10,
"volume unit of measure": "mL",
"concentration of formulation": 50,
"formulation unit of measure": "mg/mL",
"formulation": "solution",
"route": "by mouth",
"frequency": "every 6 hours"}
""",

    "IV Intermittent": ZERO_SHOT["IV Intermittent"] + """
Example of input:
Cefepime 2000 mg was delivered as a 30 minute intravenous infusion, prepared in 100 mL of 0.9% sodium chloride and administered every 8 hours.
Example of MedMatch JSON Format:
{ "drug name": "cefepime",
"numerical dose": 2000,
"abbreviated unit strength of dose": "mg",
"amount of diluent volume": 100,
"volume unit of measure": "mL",
"compatible diluent type": "0.9% sodium chloride",
"infusion time": "30 minutes",
"frequency": "every 8 hours"}
""",

    "IV Push": ZERO_SHOT["IV Push"] + """
Example of input:
A total of 6mg of adenosine (2 ml) of the 3 mg/ml vial solution intravenous was pushed once
Example of MedMatch JSON formulation:
{ "drug name": "adenosine",
"numerical dose": 6,
"abbreviated unit strength of dose": "mg",
"amount of volume": 2,
"volume unit of measure": "mL",
"concentration of solution": 3,
"concentration unit of measure": "mg/mL",
"formulation": "vial solution",
"frequency": "once"}
""",

    "IV Continuous": ZERO_SHOT["IV Continuous"] + """
Example of titratable input:
The patient was started on Ketamine continuous intravenous infusion at 0.2 mg/kg/hour using a bag of 500 mg/500 ml in 0.9% sodium chloride and titrate by 0.1 mg/kg/hour every 20 minutes to achieve a goal RASS of -4 to -5.
Example of MedMatch JSON Format:
{ "drug name": "ketamine",
"numerical dose": 500,
"abbreviated unit strength of dose": "mg",
"diluent volume": 500,
"volume unit of measure": "mL",
"compatible diluent type": "0.9% sodium chloride",
"starting rate": 0.2,
"unit of measure": "mg/kg/hr",
"titration dose": 0.1,
"titration unit of measure": "mg/kg/hr",
"titration frequency": "every 20 minutes",
"titration goal based on physiologic response, laboratory result, or assessment score": "RASS of -4 to -5"}

Example of non-titratable input:
The patient was started on a vasopressin continuous intravenous infusion at 0.04 units/minute using a 40 units/100 ml bag in 0.9% sodium chloride.
Example of MedMatch JSON Format:
{ "drug name": "vasopressin",
"numerical dose": 40,
"abbreviated unit strength of dose": "units",
"diluent volume": 100,
"volume unit of measure": "mL",
"compatible diluent type": "0.9% sodium chloride",
"starting rate": 0.04,
"unit of measure": "units/minute"}

Example of total-dose interpretation:
The patient received a continuous intravenous infusion of midazolam at a starting rate of 0.5 mg/hr (1 mg/mL in 100 mL of 0.9% sodium chloride), and the infusion was titrated by 0.5 mg/hr every 5 minutes to achieve a RASS goal of -4 to -5.
Example of MedMatch JSON Format:
{ "drug name": "midazolam",
"numerical dose": 100,
"abbreviated unit strength of dose": "mg",
"diluent volume": 100,
"volume unit of measure": "mL",
"compatible diluent type": "0.9% sodium chloride",
"starting rate": 0.5,
"unit of measure": "mg/hr",
"titration dose": 0.5,
"titration unit of measure": "mg/hr",
"titration frequency": "every 5 minutes",
"titration goal based on physiologic response, laboratory result, or assessment score": "RASS of -4 to -5"}
""",
}


# ── Jaccard similarity ────────────────────────────────────────────────────────

def word_set(value):
    """Convert a value to a lowercase word set."""
    return set(str(value).strip().lower().split())


def jaccard(a, b):
    """Word-level Jaccard similarity between two values."""
    sa, sb = word_set(a), word_set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def print_jaccard_fields(parsed, ground_truth):
    """Print per-field Jaccard when ground truth is a JSON dict."""
    all_keys = sorted(set(list(ground_truth.keys()) + list(parsed.keys())))
    scores = []
    for key in all_keys:
        gt_val = str(ground_truth.get(key, ""))
        llm_val = str(parsed.get(key, ""))
        j = jaccard(gt_val, llm_val)
        scores.append(j)
        symbol = "==" if j == 1.0 else "!="
        print(f"  {key:55s} J={j:.3f} {symbol}  (expected: {gt_val}, got: {llm_val})")

    avg = sum(scores) / len(scores) if scores else 0
    perfect = sum(1 for s in scores if s == 1.0)
    print(f"\n  Average Jaccard:      {avg:.3f}")
    print(f"  Perfect match fields: {perfect}/{len(scores)}")
    return avg


def print_jaccard_string(parsed_dict, ground_truth_str):
    """Print Jaccard when ground truth is a MedMatch string (from column B).
    Reconstructs LLM output as a flat string from the parsed JSON values,
    then computes word-level Jaccard against the ground truth string."""
    # Build a flat string from the parsed JSON values in order
    llm_parts = [str(v) for v in parsed_dict.values()]
    llm_string = " ".join(llm_parts)

    j = jaccard(ground_truth_str, llm_string)

    print(f"  Ground truth:  {ground_truth_str}")
    print(f"  LLM output:    {llm_string}")
    print(f"\n  Jaccard similarity: {j:.3f}")
    return j


# ── Parse helpers ─────────────────────────────────────────────────────────────

def parse_json_from_text(text, expected_keys=None):
    """Extract and parse JSON object from LLM response text.
    If the model returns an array instead of an object, map values to expected_keys."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if "```" in text:
            text = text[:text.rfind("```")].strip()

    # Try parsing as-is first
    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try extracting { ... }
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            parsed = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            # Try extracting [ ... ]
            try:
                start = text.index("[")
                end = text.rindex("]") + 1
                parsed = json.loads(text[start:end])
            except (ValueError, json.JSONDecodeError):
                raise ValueError("Could not find valid JSON in response")

    # If model returned an array with a single dict inside, unwrap it
    if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
        parsed = parsed[0]
    # If model returned a plain array of values, zip with expected keys
    if isinstance(parsed, list) and expected_keys:
        parsed = dict(zip(expected_keys, parsed))

    if isinstance(parsed, dict) and expected_keys:
        normalized = {}
        for key, value in parsed.items():
            key_norm = re.sub(r"\s+", " ", str(key).strip().lower())
            # No key aliases: wrong schema keys must fail strictly.
            normalized[key_norm] = value
        return {key: normalized.get(key, "") for key in expected_keys}

    return parsed


def send_to_gemma(instruction, med_order):
    """Build prompt and call the local Gemma model."""
    prompt_lower = med_order.lower()
    extra_lines = [
        "Return one JSON object only.",
        "Do not wrap the JSON in markdown.",
    ]
    if "IV Continuous" in instruction or "continuous intravenous infusion" in instruction:
        if any(token in prompt_lower for token in ("titrate", "titrated", "goal")):
            extra_lines.append(
                "This is a titratable continuous infusion. Use the key 'starting rate' for the initial rate."
            )
        else:
            extra_lines.append(
                "This is a non-titratable continuous infusion. Use the key 'starting rate' for the fixed infusion rate and leave titration fields as empty strings."
            )

    full_prompt = (
        f"{instruction}\n\n"
        + "\n".join(extra_lines)
        + f"\n\nNow process this medication order:\n{med_order}"
    )
    full_prompt += "\n\nReturn valid JSON only."
    return chat_completion(
        SYSTEM_PROMPT,
        full_prompt,
        temperature=TEMPERATURE,
        model=MODEL_NAME,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "input.txt")

    if not os.path.exists(input_path):
        print("ERROR: input.txt not found. Create it with this format:")
        print("  Line 1: route type (PO Solid, PO Liquid, IV Intermittent, IV Push, IV Continuous)")
        print("  Line 2: medication order sentence")
        print("  Line 3:  (optional) ground truth JSON for Jaccard comparison")
        print("  Line 4:  (optional) mode: zero, one, or both (default: both)")
        sys.exit(1)

    with open(input_path) as f:
        lines = f.read().strip().split("\n")

    route_type = lines[0].strip()
    med_order = lines[1].strip()

    # Line 3: optional ground truth (JSON dict OR MedMatch string from column B)
    # Line 4: optional mode (zero, one, both) — default: both
    ground_truth = None      # will be dict or str
    mode = "both"

    # Parse remaining lines — find ground truth and mode
    remaining = [l.strip() for l in lines[2:] if l.strip()]
    for line in remaining:
        if line.lower() in ("zero", "one", "both"):
            mode = line.lower()
        else:
            # Try JSON first; if it fails, treat as MedMatch string
            try:
                ground_truth = json.loads(line)
            except json.JSONDecodeError:
                ground_truth = line  # plain MedMatch string from column B

    if route_type not in ZERO_SHOT:
        print(f"ERROR: Unknown route type '{route_type}'")
        print(f"Valid: {', '.join(ZERO_SHOT.keys())}")
        sys.exit(1)

    # Select which modes to run
    modes_to_run = []
    if mode in ("zero", "both"):
        modes_to_run.append(("ZERO-SHOT", ZERO_SHOT))
    if mode in ("one", "both"):
        modes_to_run.append(("ONE-SHOT", ONE_SHOT))

    print("=" * 70)
    print(f"  Route:  {route_type}")
    print(f"  Model:  {MODEL_NAME}  (temp={TEMPERATURE})")
    print(f"  Mode:   {mode}")
    print(f"  Input:  {med_order}")
    print("=" * 70)

    for mode_name, instructions in modes_to_run:
        print(f"\n{'─'*70}")
        print(f"  {mode_name}")
        print(f"{'─'*70}")

        print(f"Sending to {MODEL_NAME}...")
        try:
            raw = send_to_gemma(instructions[route_type], med_order)
        except Exception as exc:
            print(f"ERROR: local model call failed: {exc}")
            print("Hint: Ollama is reachable, but the model may still be failing to load locally.")
            continue

        print(f"\n--- Raw Response ---\n{raw}")

        try:
            parsed = parse_json_from_text(raw, expected_keys=EXPECTED_KEYS[route_type])
        except (ValueError, json.JSONDecodeError):
            print("ERROR: Could not parse JSON from response.\n")
            continue

        # Ensure it's a dict for Jaccard comparison
        if not isinstance(parsed, dict):
            print(f"WARNING: Response is {type(parsed).__name__}, not a JSON object.\n")
            print(parsed)
            continue

        print("--- Parsed JSON ---")
        print(json.dumps(parsed, indent=2))

        if ground_truth:
            print(f"\n--- Jaccard Similarity ---")
            if isinstance(ground_truth, dict):
                print_jaccard_fields(parsed, ground_truth)
            else:
                print_jaccard_string(parsed, ground_truth)

    print()


if __name__ == "__main__":
    main()
