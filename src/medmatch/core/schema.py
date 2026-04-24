"""Shared MedMatch sheet configs and schema helpers."""

from prompt_medmatch import (
    build_local_normalization_oral_instruction,
)

PO_SOLID_INSTRUCTION = """Please review the narratives about medications and format them into the MedMatch JSON format. Follow this exact slot order; if a slot is unknown, use an empty string and do not fabricate.

The MedMatch JSON format for oral solid dosage form medications is:
[drug name][numerical dose][abbreviated unit strength of dose][amount][formulation] by mouth [frequency]

[drug name]: The generic or brand name of the medication.
[numerical dose]: The numeric amount of drug administered per dose, representing the total drug amount for that administration (e.g., 5, 10, 500). For orders with multiple identical tablets or capsules, multiply the per-unit strength by the amount (e.g., 2 capsules of 500 mg each -> numerical dose 1000, amount 2).
[abbreviated unit strength of dose]: The standardized abbreviated unit associated with the dose (e.g., mg, mcg, g).
[amount]: The number of dosage units taken per administration (e.g., 1, 2).
[formulation]: The oral solid dosage form (e.g., tablet, capsule, extended release tablet).
by mouth: The route of administration, fixed as oral.
[frequency]: How often the medication is taken (e.g., once daily, twice daily, every 8 hours).

Example of input:
Hydroxyurea 2 capsules (total dose 1000mg) by mouth once daily.
Example of MedMatch JSON format:
{ "drug name": "hydroxyurea",
"numerical dose": 1000,
"abbreviated unit strength of dose": "mg",
"amount": 2,
"formulation": "capsules",
"route": "by mouth",
"frequency": "once daily"}
"""

PO_LIQUID_INSTRUCTION = """Please review the narratives about medications and format them into the MedMatch JSON format. Follow this exact slot order; if a slot is unknown, use an empty string and do not fabricate.

The MedMatch JSON format for oral liquid dosage form medications is:
[drug name][numerical dose][abbreviated unit strength of dose][numerical volume][abbreviated unit strength of volume] of the [concentration of formulation][formulation unit of measure][formulation] by mouth [frequency]

[drug name]: The generic or brand name of the medication.
[numerical dose]: The numeric amount of drug administered per dose (e.g., 250, 5).
[abbreviated unit strength of dose]: The standardized abbreviated unit for the drug dose (e.g., mg, mcg).
[numerical volume]: The numeric volume administered per dose (e.g., 5, 10).
[abbreviated unit strength of volume]: The standardized abbreviated unit for volume (e.g., mL).
[concentration of formulation]: The strength of the medication per volume (e.g., 250 mg/5 mL).
[formulation unit of measure]: The unit used in the concentration denominator (e.g., mL).
[formulation]: The oral liquid dosage form (e.g., solution, suspension, syrup).
[route]: The route of administration, fixed as oral.
[frequency]: How often the medication is administered (e.g., once daily, every 6 hours).

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
"""

IV_INTERMITTENT_INSTRUCTION = """Please review the narratives about medications and format them into the MedMatch JSON format. Follow this exact slot order; if a slot is unknown, use an empty string and do not fabricate.

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
"""

IV_PUSH_INSTRUCTION = """Please review the narratives about medications and format them into the MedMatch JSON format. Follow this exact slot order; if a slot is unknown, use an empty string and do not fabricate.

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
"""

IV_CONTINUOUS_INSTRUCTION = """Please review the narratives about medications and format them into the MedMatch JSON format. Follow this exact slot order; if a slot is unknown, use an empty string and do not fabricate.

The MedMatch JSON format for IV continuous dosage form medications is:

Titratable IV CI:
[drug name][numerical dose][abbreviated unit strength of dose] "in" [diluent volume][volume unit of measure][compatible diluent type] "continuous intravenous infusion starting at" [starting rate][unit of measure] "titrated by" [titration dose][titration unit of measure] [titration frequency] to achieve a goal of [titration goal]

Non-titratable IV CI:
[drug name][numerical dose][abbreviated unit strength of dose][diluent volume][volume unit of measure]"in"[compatible diluent type] "continuous intravenous infusion at" [rate][unit of measure]

Example of input:
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

Example of input:
The patient was started on a vasopressin continuous intravenous infusion at 0.04 units/minute using a 40 units/100 ml bag in 0.9% sodium chloride.
Example of MedMatch JSON Format:
{ "drug name": "vasopressin",
"numerical dose": 40,
"abbreviated unit strength of dose": "units",
"diluent volume": 100,
"volume unit of measure": "mL",
"compatible diluent type": "0.9% sodium chloride",
"rate": 0.04,
"unit of measure": "units/minute"}

Definitions: (will apply to titratable and non-titratable)
[drug name]: The generic or brand name of the medication administered as a continuous IV infusion.
[numerical dose]: The numeric amount of drug contained in the prepared infusion (e.g., 50, 250).
[abbreviated unit strength of dose]: The standardized abbreviated unit associated with the dose (e.g., mg, units).
[diluent volume]: The numeric volume of diluent used to prepare the infusion (e.g., 100, 250).
[volume unit of measure]: The standardized abbreviated unit for the diluent volume (e.g., mL).
[compatible diluent type]: The IV fluid used to dilute the medication (e.g., 0.9% sodium chloride, D5W).
continuous intravenous infusion: The fixed route and method of administration.
[starting rate]: The initial infusion rate at which the medication is started (e.g., 0.05, 5).
[unit of measure]: The unit associated with the infusion rate (e.g., mcg/kg/min, units/hr, mL/hr).
[titrated by]: Indicates dose or rate adjustments are permitted.
[titration dose]: The numeric amount by which the infusion rate is adjusted per titration step (e.g., 0.01, 2).
[titration unit of measure]: The unit associated with the titration increment (e.g., mcg/kg/min, units/hr).
[titration frequency]: The time interval between allowable titrations, expressed in minutes (e.g., 5, 15).
[titration goal based on physiologic response, laboratory result, or assessment score]: The clinical target guiding titration (e.g., MAP >= 65 mmHg, RASS score -1 to 1).
"""

BASELINE_SHEET_CONFIG = {
    "PO Solid (40)": {
        "family": "baseline",
        "instruction": PO_SOLID_INSTRUCTION,
        "prompt_col": 3,
        "ground_truth_cols": {
            "drug name": 4,
            "numerical dose": 5,
            "abbreviated unit strength of dose": 6,
            "amount": 7,
            "formulation": 8,
            "route": 9,
            "frequency": 10,
        },
    },
    "PO liquid (10)": {
        "family": "baseline",
        "instruction": PO_LIQUID_INSTRUCTION,
        "prompt_col": 3,
        "ground_truth_cols": {
            "drug name": 4,
            "numerical dose": 5,
            "abbreviated unit strength of dose": 6,
            "numerical volume": 7,
            "volume unit of measure": 8,
            "concentration of formulation": 9,
            "formulation unit of measure": 10,
            "formulation": 11,
            "route": 12,
            "frequency": 13,
        },
    },
    "IV intermittent (16)": {
        "family": "baseline",
        "instruction": IV_INTERMITTENT_INSTRUCTION,
        "prompt_col": 3,
        "ground_truth_cols": {
            "drug name": 4,
            "numerical dose": 5,
            "abbreviated unit strength of dose": 6,
            "amount of diluent volume": 7,
            "volume unit of measure": 8,
            "compatible diluent type": 9,
            "infusion time": 10,
            "frequency": 11,
        },
    },
    "IV push (17)": {
        "family": "baseline",
        "instruction": IV_PUSH_INSTRUCTION,
        "prompt_col": 3,
        "ground_truth_cols": {
            "drug name": 4,
            "numerical dose": 5,
            "abbreviated unit strength of dose": 6,
            "amount of volume": 7,
            "volume unit of measure": 8,
            "concentration of solution": 9,
            "concentration unit of measure": 10,
            "formulation": 11,
            "frequency": 12,
        },
    },
    "IV continuous (16)": {
        "family": "baseline",
        "instruction": IV_CONTINUOUS_INSTRUCTION,
        "prompt_col": 3,
        "ground_truth_cols": {
            "drug name": 4,
            "numerical dose": 5,
            "abbreviated unit strength of dose": 6,
            "diluent volume": 7,
            "volume unit of measure": 8,
            "compatible diluent type": 9,
            "starting rate": 10,
            "unit of measure": 11,
            "titration dose": 12,
            "titration unit of measure": 13,
            "titration frequency": 14,
            "titration goal based on physiologic response, laboratory result, or assessment score": 15,
        },
    },
}

IV_BASELINE_SHEET_CONFIG = {
    name: config for name, config in BASELINE_SHEET_CONFIG.items() if name.startswith("IV ")
}

ORAL_BASELINE_SHEET_CONFIG = {
    name: config for name, config in BASELINE_SHEET_CONFIG.items() if name.startswith("PO ")
}

LOCAL_NORMALIZATION_ORAL_SHEET_CONFIG = {
    "PO Solid (40)": {
        "instruction": build_local_normalization_oral_instruction("PO Solid (40)"),
        "prompt_col": BASELINE_SHEET_CONFIG["PO Solid (40)"]["prompt_col"],
        "ground_truth_cols": dict(BASELINE_SHEET_CONFIG["PO Solid (40)"]["ground_truth_cols"]),
    },
    "PO liquid (10)": {
        "instruction": build_local_normalization_oral_instruction("PO liquid (10)"),
        "prompt_col": BASELINE_SHEET_CONFIG["PO liquid (10)"]["prompt_col"],
        "ground_truth_cols": dict(BASELINE_SHEET_CONFIG["PO liquid (10)"]["ground_truth_cols"]),
    },
}

LOCAL_NORMALIZATION_IV_SHEET_CONFIG = {
    "IV intermittent (16)": {
        "instruction": BASELINE_SHEET_CONFIG["IV intermittent (16)"]["instruction"],
        "prompt_col": BASELINE_SHEET_CONFIG["IV intermittent (16)"]["prompt_col"],
        "ground_truth_cols": dict(BASELINE_SHEET_CONFIG["IV intermittent (16)"]["ground_truth_cols"]),
    },
    "IV push (17)": {
        "instruction": BASELINE_SHEET_CONFIG["IV push (17)"]["instruction"],
        "prompt_col": BASELINE_SHEET_CONFIG["IV push (17)"]["prompt_col"],
        "ground_truth_cols": dict(BASELINE_SHEET_CONFIG["IV push (17)"]["ground_truth_cols"]),
    },
    "IV continuous (16)": {
        "instruction": BASELINE_SHEET_CONFIG["IV continuous (16)"]["instruction"],
        "prompt_col": BASELINE_SHEET_CONFIG["IV continuous (16)"]["prompt_col"],
        "ground_truth_cols": dict(BASELINE_SHEET_CONFIG["IV continuous (16)"]["ground_truth_cols"]),
    },
}


def expected_keys_for_sheet(sheet_name):
    return list(BASELINE_SHEET_CONFIG[sheet_name]["ground_truth_cols"].keys())
