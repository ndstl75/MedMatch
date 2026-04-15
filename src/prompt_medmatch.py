"""
Prompt helpers for MedMatch medication formatting tasks.
"""

from typing import Dict, List


SYSTEM_PROMPT = (
    "You are a clinical pharmacist who formats medication orders. Only output the MedMatch JSON format."
)

# PO_solid - Oral solid dosage forms (zero-shot prompting)
PO_SOLID_INSTRUCTION_ZERO_SHOT = """Please review the following information and format the medication into the MedMatch JSON format. Follow this exact slot order; if a slot is unknown, use an empty string and do not fabricate.

The MedMatch JSON format for oral medications is:
[drug name][numerical dose][abbreviated unit strength of dose][amount][formulation][route][frequency]

Definitions for each slot:
{{
  "drug_name": "The generic or brand name of the medication.",
  "numerical_dose": "The numeric value of the strength per unit (e.g., 5, 10, 500).",
  "abbreviated_unit_strength_of_dose": "The standardized abbreviated unit associated with the dose (e.g., mg, mcg, g).",
  "amount": "The number of dosage units taken per administration (e.g., 1, 2).",
  "formulation": "The oral solid dosage form (e.g., tablet, capsule, extended-release tablet).",
  "route": "The route of administration, fixed as oral.",
  "frequency": "How often the medication is taken (e.g., once daily, twice daily, every 8 hours)."
}}
"""

# PO_solid - System prompt with base instruction
PO_SOLID_SYSTEM_PROMPT_ONE_SHOT = SYSTEM_PROMPT + """

Please review the following information and format the medication into the MedMatch JSON format. Follow this exact slot order; if a slot is unknown, use an empty string and do not fabricate.

The MedMatch JSON format for oral medications is:
[drug name][numerical dose][abbreviated unit strength of dose][amount][formulation][route][frequency]

Definitions for each slot:
{{
  "drug_name": "The generic or brand name of the medication.",
  "numerical_dose": "The numeric value of the strength per unit (e.g., 5, 10, 500).",
  "abbreviated_unit_strength_of_dose": "The standardized abbreviated unit associated with the dose (e.g., mg, mcg, g).",
  "amount": "The number of dosage units taken per administration (e.g., 1, 2).",
  "formulation": "The oral solid dosage form (e.g., tablet, capsule, extended-release tablet).",
  "route": "The route of administration, fixed as oral.",
  "frequency": "How often the medication is taken (e.g., once daily, twice daily, every 8 hours)."
}}

NOTE: Output only a single valid JSON object with the medication information, containing no explanations, additional text, examples, or commentary."""

# PO_solid - Example for multi-turn one-shot prompting
PO_SOLID_EXAMPLE_USER = "Query: Administer oral benztropine four times daily as needed, a dose of 1mg (1 tablet)."

PO_SOLID_EXAMPLE_ASSISTANT = """{{
  "drug_name": "benztropine",
  "numerical_dose": "1",
  "abbreviated_unit_strength_of_dose": "mg",
  "amount": "1",
  "formulation": "tablet",
  "route": "by mouth",
  "frequency": "four times daily as needed"
}}"""

# PO_solid - Backward compatible single-turn one-shot instruction (deprecated)
PO_SOLID_INSTRUCTION_ONE_SHOT = """Please review the following information and format the medication into the MedMatch JSON format. Follow this exact slot order; if a slot is unknown, use an empty string and do not fabricate.

The MedMatch JSON format for oral medications is:
[drug name][numerical dose][abbreviated unit strength of dose][amount][formulation][route][frequency]

Definitions for each slot:
{{
  "drug_name": "The generic or brand name of the medication.",
  "numerical_dose": "The numeric value of the strength per unit (e.g., 5, 10, 500).",
  "abbreviated_unit_strength_of_dose": "The standardized abbreviated unit associated with the dose (e.g., mg, mcg, g).",
  "amount": "The number of dosage units taken per administration (e.g., 1, 2).",
  "formulation": "The oral solid dosage form (e.g., tablet, capsule, extended-release tablet).",
  "route": "The route of administration, fixed as by mouth.",
  "frequency": "How often the medication is taken (e.g., once daily, twice daily, every 8 hours)."
}}

NOTE: Output only a single valid JSON object with the medication information, containing no explanations, additional text, examples, or commentary.

### Example:

Query: Administer oral benztropine four times daily as needed, a dose of 1mg (1 tablet).

Output:
{{
  "drug_name": "benztropine",
  "numerical_dose": "1",
  "abbreviated_unit_strength_of_dose": "mg",
  "amount": "1",
  "formulation": "tablet",
  "route": "by mouth",
  "frequency": "four times daily as needed"
}}
"""

# PO_liquid - Oral liquid dosage forms (zero-shot prompting)
PO_LIQUID_INSTRUCTION_ZERO_SHOT = """Please review the narratives about medications and format them into the MedMatch JSON format. Follow this exact slot order; if a slot is unknown, use an empty string and do not fabricate.

The MedMatch JSON format for oral liquid dosage form medications is:
[drug name][numerical dose][abbreviated unit strength of dose][numerical volume][abbreviated unit strength of volume][concentration of formulation][formulation unit of measure][formulation][route][frequency]

Definitions for each slot:
{{
  "drug_name": "The generic or brand name of the medication.",
  "numerical_dose": "The numeric amount of drug administered per dose (e.g., 250, 5).",
  "abbreviated_unit_strength_of_dose": "The standardized abbreviated unit for the drug dose (e.g., mg, mcg).",
  "numerical_volume": "The numeric volume administered per dose (e.g., 5, 10).",
  "volume_unit_of_measure": "The standardized abbreviated unit for volume (e.g., mL).",
  "concentration_of_formulation": "The strength of the medication per volume (e.g., 250 mg/5 mL).",
  "formulation_unit_of_measure": "The unit used in the concentration denominator (e.g., mL).",
  "formulation": "The oral liquid dosage form (e.g., solution, suspension, syrup).",
  "route": "The route of administration, fixed as oral.",
  "frequency": "How often the medication is administered (e.g., once daily, every 6 hours)."
}}
"""

# PO_liquid - System prompt with base instruction
PO_LIQUID_SYSTEM_PROMPT_ONE_SHOT = SYSTEM_PROMPT + """

Please review the narratives about medications and format them into the MedMatch JSON format. Follow this exact slot order; if a slot is unknown, use an empty string and do not fabricate.

The MedMatch JSON format for oral liquid dosage form medications is:
[drug name][numerical dose][abbreviated unit strength of dose][numerical volume][abbreviated unit strength of volume][concentration of formulation][formulation unit of measure][formulation][route][frequency]

Definitions for each slot:
{{
  "drug_name": "The generic or brand name of the medication.",
  "numerical_dose": "The numeric amount of drug administered per dose (e.g., 250, 5).",
  "abbreviated_unit_strength_of_dose": "The standardized abbreviated unit for the drug dose (e.g., mg, mcg).",
  "numerical_volume": "The numeric volume administered per dose (e.g., 5, 10).",
  "volume_unit_of_measure": "The standardized abbreviated unit for volume (e.g., mL).",
  "concentration_of_formulation": "The strength of the medication per volume (e.g., 250 mg/5 mL).",
  "formulation_unit_of_measure": "The unit used in the concentration denominator (e.g., mL).",
  "formulation": "The oral liquid dosage form (e.g., solution, suspension, syrup).",
  "route": "The route of administration, fixed as oral.",
  "frequency": "How often the medication is administered (e.g., once daily, every 6 hours)."
}}

NOTE: Output only a single valid JSON object with the medication information, containing no explanations, additional text, examples, or commentary."""

# PO_liquid - Example for multi-turn one-shot prompting
PO_LIQUID_EXAMPLE_USER = "Query: An oral solution of diazepam (1mg/mL) is used to administer 5mg/5mL dose by mouth once daily."

PO_LIQUID_EXAMPLE_ASSISTANT = """{{
  "drug_name": "diazepam",
  "numerical_dose": "5",
  "abbreviated_unit_strength_of_dose": "mg",
  "numerical_volume": "5",
  "volume_unit_of_measure": "mL",
  "concentration_of_formulation": "1",
  "formulation_unit_of_measure": "mg/mL",
  "formulation": "solution",
  "route": "by mouth",
  "frequency": "once daily"
}}"""

# PO_liquid - Backward compatible single-turn one-shot instruction (deprecated)
PO_LIQUID_INSTRUCTION_ONE_SHOT = """Please review the narratives about medications and format them into the MedMatch JSON format. Follow this exact slot order; if a slot is unknown, use an empty string and do not fabricate.

The MedMatch JSON format for oral liquid dosage form medications is:
[drug name][numerical dose][abbreviated unit strength of dose][numerical volume][abbreviated unit strength of volume][concentration of formulation][formulation unit of measure][formulation][route][frequency]

Definitions for each slot:
{{
  "drug_name": "The generic or brand name of the medication.",
  "numerical_dose": "The numeric amount of drug administered per dose (e.g., 250, 5).",
  "abbreviated_unit_strength_of_dose": "The standardized abbreviated unit for the drug dose (e.g., mg, mcg).",
  "numerical_volume": "The numeric volume administered per dose (e.g., 5, 10).",
  "volume_unit_of_measure": "The standardized abbreviated unit for volume (e.g., mL).",
  "concentration_of_formulation": "The strength of the medication per volume (e.g., 250 mg/5 mL).",
  "formulation_unit_of_measure": "The unit used in the concentration denominator (e.g., mL).",
  "formulation": "The oral liquid dosage form (e.g., solution, suspension, syrup).",
  "route": "The route of administration, fixed as oral.",
  "frequency": "How often the medication is administered (e.g., once daily, every 6 hours)."
}}

NOTE: Output only a single valid JSON object with the medication information, containing no explanations, additional text, examples, or commentary.

### Example:

Query: An oral solution of diazepam (1mg/mL) is used to administer 5mg/5mL dose by mouth once daily.

Output:
{{
  "drug_name": "diazepam",
  "numerical_dose": "5",
  "abbreviated_unit_strength_of_dose": "mg",
  "numerical_volume": "5",
  "volume_unit_of_measure": "mL",
  "concentration_of_formulation": "1",
  "formulation_unit_of_measure": "mg/mL",
  "formulation": "solution",
  "route": "by mouth",
  "frequency": "once daily"
}}
"""

# IV_intermittent - Intravenous intermittent medications (zero-shot prompting)
IV_INTERMITTENT_INSTRUCTION_ZERO_SHOT = """Please review the following information and format the medication into the MedMatch JSON format. Follow this exact slot order; if a slot is unknown, use an empty string and do not fabricate.

The MedMatch JSON format for intravenous intermittent medications is:
[drug name][numerical dose][abbreviated unit strength of dose][amount of diluent volume][volume unit of measure][compatible diluent type] intravenous infused over [infusion time] [frequency]

Definitions for each slot:
{{
  "drug_name": "The generic or brand name of the medication to be administered intravenously.",
  "numerical_dose": "The numeric value of the drug dose to be given per administration (e.g., 1, 500).",
  "abbreviated_unit_strength_of_dose": "The standardized abbreviated unit associated with the dose (e.g., mg, g, units).",
  "amount_of_diluent_volume": "The numeric volume of diluent used to prepare the IV medication (e.g., 50, 100).",
  "volume_unit_of_measure": "The standardized abbreviated unit for diluent volume (e.g., mL).",
  "compatible_diluent_type": "The IV fluid used for dilution that is compatible with the medication (e.g., 0.9% sodium chloride, D5W).",
  "infusion_time": "The duration over which the medication is infused (e.g., 30 minutes, 1 hour).",
  "frequency": "How often the intermittent IV dose is administered (e.g., every 8 hours, once daily)."
}}
"""

# IV_intermittent - System prompt with base instruction
IV_INTERMITTENT_SYSTEM_PROMPT_ONE_SHOT = SYSTEM_PROMPT + """

Please review the following information and format the medication into the MedMatch JSON format. Follow this exact slot order; if a slot is unknown, use an empty string and do not fabricate.

The MedMatch JSON format for intravenous intermittent medications is:
[drug name][numerical dose][abbreviated unit strength of dose][amount of diluent volume][volume unit of measure][compatible diluent type] intravenous infused over [infusion time] [frequency]

Definitions for each slot:
{{
  "drug_name": "The generic or brand name of the medication to be administered intravenously.",
  "numerical_dose": "The numeric value of the drug dose to be given per administration (e.g., 1, 500).",
  "abbreviated_unit_strength_of_dose": "The standardized abbreviated unit associated with the dose (e.g., mg, g, units).",
  "amount_of_diluent_volume": "The numeric volume of diluent used to prepare the IV medication (e.g., 50, 100).",
  "volume_unit_of_measure": "The standardized abbreviated unit for diluent volume (e.g., mL).",
  "compatible_diluent_type": "The IV fluid used for dilution that is compatible with the medication (e.g., 0.9% sodium chloride, D5W).",
  "infusion_time": "The duration over which the medication is infused (e.g., 30 minutes, 1 hour).",
  "frequency": "How often the intermittent IV dose is administered (e.g., every 8 hours, once daily)."
}}

NOTE: Output only a single valid JSON object with the medication information, containing no explanations, additional text, examples, or commentary."""

# IV_intermittent - Example for multi-turn one-shot prompting
IV_INTERMITTENT_EXAMPLE_USER = "Query: Cefepime 2000 mg was delivered as a 30 minute intravenous infusion, prepared in 100 mL of 0.9% sodium chloride and administered every 8 hours."

IV_INTERMITTENT_EXAMPLE_ASSISTANT = """{{
  "drug_name": "cefepime",
  "numerical_dose": "2000",
  "abbreviated_unit_strength_of_dose": "mg",
  "amount_of_diluent_volume": "100",
  "volume_unit_of_measure": "mL",
  "compatible_diluent_type": "0.9% sodium chloride",
  "infusion_time": "30 minutes",
  "frequency": "every 8 hours"
}}"""

# IV_intermittent - Backward compatible single-turn one-shot instruction (deprecated)
IV_INTERMITTENT_INSTRUCTION_ONE_SHOT = """Please review the following information and format the medication into the MedMatch JSON format. Follow this exact slot order; if a slot is unknown, use an empty string and do not fabricate.

The MedMatch JSON format for intravenous intermittent medications is:
[drug name][numerical dose][abbreviated unit strength of dose][amount of diluent volume][volume unit of measure][compatible diluent type] intravenous infused over [infusion time] [frequency]

Definitions for each slot:
{{
  "drug_name": "The generic or brand name of the medication to be administered intravenously.",
  "numerical_dose": "The numeric value of the drug dose to be given per administration (e.g., 1, 500).",
  "abbreviated_unit_strength_of_dose": "The standardized abbreviated unit associated with the dose (e.g., mg, g, units).",
  "amount_of_diluent_volume": "The numeric volume of diluent used to prepare the IV medication (e.g., 50, 100).",
  "volume_unit_of_measure": "The standardized abbreviated unit for diluent volume (e.g., mL).",
  "compatible_diluent_type": "The IV fluid used for dilution that is compatible with the medication (e.g., 0.9% sodium chloride, D5W).",
  "infusion_time": "The duration over which the medication is infused (e.g., 30 minutes, 1 hour).",
  "frequency": "How often the intermittent IV dose is administered (e.g., every 8 hours, once daily)."
}}

NOTE: Output only a single valid JSON object with the medication information, containing no explanations, additional text, examples, or commentary.

### Example:

Query: Cefepime 2000 mg was delivered as a 30 minute intravenous infusion, prepared in 100 mL of 0.9% sodium chloride and administered every 8 hours.

Output:
{{
  "drug_name": "cefepime",
  "numerical_dose": "2000",
  "abbreviated_unit_strength_of_dose": "mg",
  "amount_of_diluent_volume": "100",
  "volume_unit_of_measure": "mL",
  "compatible_diluent_type": "0.9% sodium chloride",
  "infusion_time": "30 minutes",
  "frequency": "every 8 hours"
}}
"""

# IV_push - Intravenous push medications (zero-shot prompting)
IV_PUSH_INSTRUCTION_ZERO_SHOT = """Please review the narratives about medications and format them into the MedMatch JSON format. Follow this exact slot order; if a slot is unknown, use an empty string and do not fabricate.

The MedMatch JSON format for intravenous push dosage form medications is:
[drug name][numerical dose][abbreviated unit strength of dose][amount of volume][volume unit of measure] of the [concentration of solution][concentration unit of measure][formulation] intravenous push [frequency]

Definitions for each slot:
{{
  "drug_name": "The generic or brand name of the medication administered by IV push.",
  "numerical_dose": "The numeric value of the drug dose delivered per administration (e.g., 2, 10).",
  "abbreviated_unit_strength_of_dose": "The standardized abbreviated unit for the dose (e.g., mg, mcg).",
  "amount_of_volume": "The numeric volume administered with the IV push (e.g., 2, 5).",
  "volume_unit_of_measure": "The standardized abbreviated unit for volume (e.g., mL).",
  "concentration_of_solution": "The strength of the drug within the solution (e.g., 2 mg/2 mL).",
  "concentration_unit_of_measure": "The unit basis used to express the concentration (e.g., mg/mL).",
  "formulation": "The injectable dosage form (e.g., solution).",
  "frequency": "How often the IV push dose is administered (e.g., every 6 hours, once)."
}}
"""

# IV_push - System prompt with base instruction
IV_PUSH_SYSTEM_PROMPT_ONE_SHOT = SYSTEM_PROMPT + """

Please review the narratives about medications and format them into the MedMatch JSON format. Follow this exact slot order; if a slot is unknown, use an empty string and do not fabricate.

The MedMatch JSON format for intravenous push dosage form medications is:
[drug name][numerical dose][abbreviated unit strength of dose][amount of volume][volume unit of measure] of the [concentration of solution][concentration unit of measure][formulation] intravenous push [frequency]

Definitions for each slot:
{{
  "drug_name": "The generic or brand name of the medication administered by IV push.",
  "numerical_dose": "The numeric value of the drug dose delivered per administration (e.g., 2, 10).",
  "abbreviated_unit_strength_of_dose": "The standardized abbreviated unit for the dose (e.g., mg, mcg).",
  "amount_of_volume": "The numeric volume administered with the IV push (e.g., 2, 5).",
  "volume_unit_of_measure": "The standardized abbreviated unit for volume (e.g., mL).",
  "concentration_of_solution": "The strength of the drug within the solution (e.g., 2 mg/2 mL).",
  "concentration_unit_of_measure": "The unit basis used to express the concentration (e.g., mg/mL).",
  "formulation": "The injectable dosage form (e.g., solution).",
  "frequency": "How often the IV push dose is administered (e.g., every 6 hours, once)."
}}

NOTE: Output only a single valid JSON object with the medication information, containing no explanations, additional text, examples, or commentary."""

# IV_push - Example for multi-turn one-shot prompting
IV_PUSH_EXAMPLE_USER = "Query: A total of 6mg of adenosine (2 ml) of the 3 mg/ml vial solution intravenous was pushed once."

IV_PUSH_EXAMPLE_ASSISTANT = """{{
  "drug_name": "adenosine",
  "numerical_dose": "6",
  "abbreviated_unit_strength_of_dose": "mg",
  "amount_of_volume": "2",
  "volume_unit_of_measure": "mL",
  "concentration_of_solution": "3",
  "concentration_unit_of_measure": "mg/mL",
  "formulation": "vial solution",
  "frequency": "once"
}}"""

# IV_push - Backward compatible single-turn one-shot instruction (deprecated)
IV_PUSH_INSTRUCTION_ONE_SHOT = """Please review the narratives about medications and format them into the MedMatch JSON format. Follow this exact slot order; if a slot is unknown, use an empty string and do not fabricate.

The MedMatch JSON format for intravenous push dosage form medications is:
[drug name][numerical dose][abbreviated unit strength of dose][amount of volume][volume unit of measure] of the [concentration of solution][concentration unit of measure][formulation] intravenous push [frequency]

Definitions for each slot:
{{
  "drug_name": "The generic or brand name of the medication administered by IV push.",
  "numerical_dose": "The numeric value of the drug dose delivered per administration (e.g., 2, 10).",
  "abbreviated_unit_strength_of_dose": "The standardized abbreviated unit for the dose (e.g., mg, mcg).",
  "amount_of_volume": "The numeric volume administered with the IV push (e.g., 2, 5).",
  "volume_unit_of_measure": "The standardized abbreviated unit for volume (e.g., mL).",
  "concentration_of_solution": "The strength of the drug within the solution (e.g., 2 mg/2 mL).",
  "concentration_unit_of_measure": "The unit basis used to express the concentration (e.g., mg/mL).",
  "formulation": "The injectable dosage form (e.g., solution).",
  "frequency": "How often the IV push dose is administered (e.g., every 6 hours, once)."
}}

NOTE: Output only a single valid JSON object with the medication information, containing no explanations, additional text, examples, or commentary.

### Example:

Query: A total of 6mg of adenosine (2 ml) of the 3 mg/ml vial solution intravenous was pushed once.

Output:
{{
  "drug_name": "adenosine",
  "numerical_dose": "6",
  "abbreviated_unit_strength_of_dose": "mg",
  "amount_of_volume": "2",
  "volume_unit_of_measure": "mL",
  "concentration_of_solution": "3",
  "concentration_unit_of_measure": "mg/mL",
  "formulation": "vial solution",
  "frequency": "once"
}}
"""

# IV_continuous - Intravenous continuous medications (zero-shot prompting)
IV_CONTINUOUS_INSTRUCTION_ZERO_SHOT = """Please review the narratives about medications and format them into the MedMatch JSON format. Follow this exact slot order; if a slot is unknown, use an empty string and do not fabricate.

First, classify the intravenous continuous infusion as either "titratable" or "non-titratable":
- Titratable: If the infusion mentions titration parameters (e.g., titrate by X every Y minutes to achieve goal Z)
- Non-titratable: If the infusion does not mention titration parameters and has a fixed rate

Then apply the appropriate format:

For titratable IV continuous infusions:
[drug name][numerical dose][abbreviated unit strength of dose] [diluent volume][volume unit of measure] in [compatible diluent type] continuous intravenous infusion starting at [starting rate][unit of measure] titrated by [titration dose][titration unit of measure][titration frequency] minutes to achieve a goal [titration goal based on physiologic response, laboratory result, or assessment score]

For non-titratable IV continuous infusions:
[drug name][numerical dose][abbreviated unit strength of dose][diluent volume][volume unit of measure]"in"[compatible diluent type] "continuous intravenous infusion starting at" [starting rate][unit of measure]

Definitions for each slot:
{{
  "classification": "The type of continuous IV infusion: either 'titratable' (with titration parameters) or 'non-titratable' (fixed rate).",
  "drug_name": "The generic or brand name of the medication administered as a continuous IV infusion.",
  "numerical_dose": "The numeric amount of drug contained in the prepared infusion (e.g., 50, 250).",
  "abbreviated_unit_strength_of_dose": "The standardized abbreviated unit associated with the dose (e.g., mg, units).",
  "diluent_volume": "The numeric volume of diluent used to prepare the infusion (e.g., 100, 250).",
  "volume_unit_of_measure": "The standardized abbreviated unit for the diluent volume (e.g., mL).",
  "compatible_diluent_type": "The IV fluid used to dilute the medication (e.g., 0.9% sodium chloride, D5W).",
  "starting_rate": "The infusion rate at which the medication is administered (e.g., 0.05, 5). For titratable infusions this is the initial rate; for non-titratable infusions this is the fixed rate.",
  "unit_of_measure": "The unit associated with the infusion rate (e.g., mcg/kg/min, units/hr, mL/hr).",
  "titration_dose": "The numeric amount by which the infusion rate is adjusted per titration step (e.g., 0.01, 2).",
  "titration_unit_of_measure": "The unit associated with the titration increment (e.g., mcg/kg/min, units/hr).",
  "titration_frequency": "The time interval between allowable titrations, expressed in minutes (e.g., 5, 15).",
  "titration_goal_based_on_physiologic_response_laboratory_result_or_assessment_score": "The clinical target guiding titration (e.g., MAP ≥ 65 mmHg, RASS score -1 to 1)."
}}
"""

# IV_continuous - System prompt with base instruction
IV_CONTINUOUS_SYSTEM_PROMPT_ONE_SHOT = SYSTEM_PROMPT + """

Please review the narratives about medications and format them into the MedMatch JSON format. Follow this exact slot order; if a slot is unknown, use an empty string and do not fabricate.

First, classify the intravenous continuous infusion as either "titratable" or "non-titratable":
- Titratable: If the infusion mentions titration parameters (e.g., titrate by X every Y minutes to achieve goal Z)
- Non-titratable: If the infusion does not mention titration parameters and has a fixed rate

Then apply the appropriate format:

For titratable IV continuous infusions:
[drug name][numerical dose][abbreviated unit strength of dose] [diluent volume][volume unit of measure] in [compatible diluent type] continuous intravenous infusion starting at [starting rate][unit of measure] titrated by [titration dose][titration unit of measure][titration frequency] minutes to achieve a goal [titration goal based on physiologic response, laboratory result, or assessment score]

For non-titratable IV continuous infusions:
[drug name][numerical dose][abbreviated unit strength of dose][diluent volume][volume unit of measure]"in"[compatible diluent type] "continuous intravenous infusion starting at" [starting rate][unit of measure]

Definitions for each slot:
{{
  "classification": "The type of continuous IV infusion: either 'titratable' (with titration parameters) or 'non-titratable' (fixed rate).",
  "drug_name": "The generic or brand name of the medication administered as a continuous IV infusion.",
  "numerical_dose": "The numeric amount of drug contained in the prepared infusion (e.g., 50, 250).",
  "abbreviated_unit_strength_of_dose": "The standardized abbreviated unit associated with the dose (e.g., mg, units).",
  "diluent_volume": "The numeric volume of diluent used to prepare the infusion (e.g., 100, 250).",
  "volume_unit_of_measure": "The standardized abbreviated unit for the diluent volume (e.g., mL).",
  "compatible_diluent_type": "The IV fluid used to dilute the medication (e.g., 0.9% sodium chloride, D5W).",
  "starting_rate": "The infusion rate at which the medication is administered (e.g., 0.05, 5). For titratable infusions this is the initial rate; for non-titratable infusions this is the fixed rate.",
  "unit_of_measure": "The unit associated with the infusion rate (e.g., mcg/kg/min, units/hr, mL/hr).",
  "titration_dose": "The numeric amount by which the infusion rate is adjusted per titration step (e.g., 0.01, 2).",
  "titration_unit_of_measure": "The unit associated with the titration increment (e.g., mcg/kg/min, units/hr).",
  "titration_frequency": "The time interval between allowable titrations, expressed in minutes (e.g., 5, 15).",
  "titration_goal_based_on_physiologic_response_laboratory_result_or_assessment_score": "The clinical target guiding titration (e.g., MAP ≥ 65 mmHg, RASS score -1 to 1)."
}}

NOTE: Output only a single valid JSON object with the medication information, containing no explanations, additional text, examples, or commentary."""

# IV_continuous - Examples for multi-turn one-shot prompting (titratable example)
IV_CONTINUOUS_EXAMPLE_USER_TITRATABLE = "Query: The patient was started on Ketamine continuous intravenous infusion at 0.2 mg/kg/hour using a bag of 500 mg/500 ml in 0.9% sodium chloride and titrate by 0.1 mg/kg/hour every 20 minutes to achieve a goal RASS of –4 to –5."

IV_CONTINUOUS_EXAMPLE_ASSISTANT_TITRATABLE = """{{
  "classification": "titratable",
  "drug_name": "ketamine",
  "numerical_dose": "500",
  "abbreviated_unit_strength_of_dose": "mg",
  "diluent_volume": "500",
  "volume_unit_of_measure": "mL",
  "compatible_diluent_type": "0.9% sodium chloride",
  "starting_rate": "0.2",
  "unit_of_measure": "mg/kg/hour",
  "titration_dose": "0.1",
  "titration_unit_of_measure": "mg/kg/hour",
  "titration_frequency": "every 20 minutes",
  "titration_goal_based_on_physiologic_response_laboratory_result_or_assessment_score": "RASS of -4 to -5"
}}"""

# IV_continuous - Examples for multi-turn one-shot prompting (non-titratable example)
IV_CONTINUOUS_EXAMPLE_USER_NON_TITRATABLE = "Query: Vasopressin 40 units/100 ml in 0.9% sodium chloride continuous intravenous infusion at 0.04 units/minute."

IV_CONTINUOUS_EXAMPLE_ASSISTANT_NON_TITRATABLE = """{{
  "classification": "non-titratable",
  "drug_name": "vasopressin",
  "numerical_dose": "40",
  "abbreviated_unit_strength_of_dose": "units",
  "diluent_volume": "100",
  "volume_unit_of_measure": "mL",
  "compatible_diluent_type": "0.9% sodium chloride",
  "starting_rate": "0.04",
  "unit_of_measure": "units/minute"
}}"""

# IV_continuous - Backward compatible single-turn one-shot instruction (deprecated)
IV_CONTINUOUS_INSTRUCTION_ONE_SHOT = """Please review the narratives about medications and format them into the MedMatch JSON format. Follow this exact slot order; if a slot is unknown, use an empty string and do not fabricate.

First, classify the intravenous continuous infusion as either "titratable" or "non-titratable":
- Titratable: If the infusion mentions titration parameters (e.g., titrate by X every Y minutes to achieve goal Z)
- Non-titratable: If the infusion does not mention titration parameters and has a fixed rate

Then apply the appropriate format:

For titratable IV continuous infusions:
[drug name][numerical dose][abbreviated unit strength of dose] [diluent volume][volume unit of measure] in [compatible diluent type] continuous intravenous infusion starting at [starting rate][unit of measure] titrated by [titration dose][titration unit of measure][titration frequency] minutes to achieve a goal [titration goal based on physiologic response, laboratory result, or assessment score]

For non-titratable IV continuous infusions:
[drug name][numerical dose][abbreviated unit strength of dose][diluent volume][volume unit of measure]"in"[compatible diluent type] "continuous intravenous infusion starting at" [starting rate][unit of measure]

Definitions for each slot:
{{
  "classification": "The type of continuous IV infusion: either 'titratable' (with titration parameters) or 'non-titratable' (fixed rate).",
  "drug_name": "The generic or brand name of the medication administered as a continuous IV infusion.",
  "numerical_dose": "The numeric amount of drug contained in the prepared infusion (e.g., 50, 250).",
  "abbreviated_unit_strength_of_dose": "The standardized abbreviated unit associated with the dose (e.g., mg, units).",
  "diluent_volume": "The numeric volume of diluent used to prepare the infusion (e.g., 100, 250).",
  "volume_unit_of_measure": "The standardized abbreviated unit for the diluent volume (e.g., mL).",
  "compatible_diluent_type": "The IV fluid used to dilute the medication (e.g., 0.9% sodium chloride, D5W).",
  "starting_rate": "The infusion rate at which the medication is administered (e.g., 0.05, 5). For titratable infusions this is the initial rate; for non-titratable infusions this is the fixed rate.",
  "unit_of_measure": "The unit associated with the infusion rate (e.g., mcg/kg/min, units/hr, mL/hr).",
  "titration_dose": "The numeric amount by which the infusion rate is adjusted per titration step (e.g., 0.01, 2).",
  "titration_unit_of_measure": "The unit associated with the titration increment (e.g., mcg/kg/min, units/hr).",
  "titration_frequency": "The time interval between allowable titrations, expressed in minutes (e.g., 5, 15).",
  "titration_goal_based_on_physiologic_response_laboratory_result_or_assessment_score": "The clinical target guiding titration (e.g., MAP ≥ 65 mmHg, RASS score -1 to 1)."
}}

NOTE: Output only a single valid JSON object with the medication information, containing no explanations, additional text, examples, or commentary.

### Examples:

Query: The patient was started on Ketamine continuous intravenous infusion at 0.2 mg/kg/hour using a bag of 500 mg/500 ml in 0.9% sodium chloride and titrate by 0.1 mg/kg/hour every 20 minutes to achieve a goal RASS of –4 to –5.

Output:
{{
  "classification": "titratable",
  "drug_name": "ketamine",
  "numerical_dose": "500",
  "abbreviated_unit_strength_of_dose": "mg",
  "diluent_volume": "500",
  "volume_unit_of_measure": "mL",
  "compatible_diluent_type": "0.9% sodium chloride",
  "starting_rate": "0.2",
  "unit_of_measure": "mg/kg/hour",
  "titration_dose": "0.1",
  "titration_unit_of_measure": "mg/kg/hour",
  "titration_frequency": "every 20 minutes",
  "titration_goal_based_on_physiologic_response_laboratory_result_or_assessment_score": "RASS of -4 to -5"
}}

Query: Vasopressin 40 units/100 ml in 0.9% sodium chloride continuous intravenous infusion at 0.04 units/minute.

Output:
{{
  "classification": "non-titratable",
  "drug_name": "vasopressin",
  "numerical_dose": "40",
  "abbreviated_unit_strength_of_dose": "units",
  "diluent_volume": "100",
  "volume_unit_of_measure": "mL",
  "compatible_diluent_type": "0.9% sodium chloride",
  "starting_rate": "0.04",
  "unit_of_measure": "units/minute"
}}
"""


def _build_messages(instruction: str, medication_prompt: str) -> List[Dict[str, str]]:
    """Build single-turn messages (deprecated, for backward compatibility)."""
    user_content = f"""{instruction}

Query:
{medication_prompt}"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _build_messages_multi_turn(
    system_prompt_with_instruction: str,
    example_user: str,
    example_assistant: str,
    medication_prompt: str
) -> List[Dict[str, str]]:
    """Build multi-turn messages with instruction in system prompt."""
    return [
        {"role": "system", "content": system_prompt_with_instruction},
        {"role": "user", "content": example_user},
        {"role": "assistant", "content": example_assistant},
        {"role": "user", "content": f"Query: {medication_prompt}"},
    ]


def _build_messages_multi_turn_two_shot(
    system_prompt_with_instruction: str,
    example_user_1: str,
    example_assistant_1: str,
    example_user_2: str,
    example_assistant_2: str,
    medication_prompt: str
) -> List[Dict[str, str]]:
    """Build multi-turn messages with two example user/assistant turns."""
    return [
        {"role": "system", "content": system_prompt_with_instruction},
        {"role": "user", "content": example_user_1},
        {"role": "assistant", "content": example_assistant_1},
        {"role": "user", "content": example_user_2},
        {"role": "assistant", "content": example_assistant_2},
        {"role": "user", "content": f"Query: {medication_prompt}"},
    ]


def build_po_solid_messages_zero_shot(medication_prompt: str) -> List[Dict[str, str]]:
    """Prompt for oral solid medications using zero-shot prompting."""
    return _build_messages(PO_SOLID_INSTRUCTION_ZERO_SHOT, medication_prompt)


def build_po_solid_messages_one_shot(medication_prompt: str) -> List[Dict[str, str]]:
    """Prompt for oral solid medications using one-shot prompting (single-turn, deprecated)."""
    return _build_messages(PO_SOLID_INSTRUCTION_ONE_SHOT, medication_prompt)


def build_po_solid_messages_one_shot_multi_turn(medication_prompt: str) -> List[Dict[str, str]]:
    """Prompt for oral solid medications using one-shot prompting (multi-turn)."""
    return _build_messages_multi_turn(
        PO_SOLID_SYSTEM_PROMPT_ONE_SHOT,
        PO_SOLID_EXAMPLE_USER,
        PO_SOLID_EXAMPLE_ASSISTANT,
        medication_prompt,
    )


# Backward compatibility: deprecated, use build_po_solid_messages_zero_shot() instead
def build_po_messages_zero_shot(medication_prompt: str) -> List[Dict[str, str]]:
    """Prompt for oral solid medications using zero-shot prompting (deprecated)."""
    return build_po_solid_messages_zero_shot(medication_prompt)


# Backward compatibility: deprecated, use build_po_solid_messages_one_shot() instead
def build_po_messages_one_shot(medication_prompt: str) -> List[Dict[str, str]]:
    """Prompt for oral solid medications using one-shot prompting (deprecated)."""
    return build_po_solid_messages_one_shot(medication_prompt)


# Backward compatibility: deprecated, use build_po_solid_messages_one_shot() instead
def build_po_messages(medication_prompt: str) -> List[Dict[str, str]]:
    """Prompt for oral solid medications using one-shot prompting (deprecated)."""
    return build_po_solid_messages_one_shot(medication_prompt)


def build_po_liquid_messages_zero_shot(medication_prompt: str) -> List[Dict[str, str]]:
    """Prompt for oral liquid medications using zero-shot prompting."""
    return _build_messages(PO_LIQUID_INSTRUCTION_ZERO_SHOT, medication_prompt)


def build_po_liquid_messages_one_shot(medication_prompt: str) -> List[Dict[str, str]]:
    """Prompt for oral liquid medications using one-shot prompting (single-turn, deprecated)."""
    return _build_messages(PO_LIQUID_INSTRUCTION_ONE_SHOT, medication_prompt)


def build_po_liquid_messages_one_shot_multi_turn(medication_prompt: str) -> List[Dict[str, str]]:
    """Prompt for oral liquid medications using one-shot prompting (multi-turn)."""
    return _build_messages_multi_turn(
        PO_LIQUID_SYSTEM_PROMPT_ONE_SHOT,
        PO_LIQUID_EXAMPLE_USER,
        PO_LIQUID_EXAMPLE_ASSISTANT,
        medication_prompt,
    )


def build_iv_intermit_messages_zero_shot(medication_prompt: str) -> List[Dict[str, str]]:
    """Prompt for IV intermittent medications using zero-shot prompting."""
    return _build_messages(IV_INTERMITTENT_INSTRUCTION_ZERO_SHOT, medication_prompt)


def build_iv_intermit_messages_one_shot(medication_prompt: str) -> List[Dict[str, str]]:
    """Prompt for IV intermittent medications using one-shot prompting (single-turn, deprecated)."""
    return _build_messages(IV_INTERMITTENT_INSTRUCTION_ONE_SHOT, medication_prompt)


def build_iv_intermit_messages_one_shot_multi_turn(medication_prompt: str) -> List[Dict[str, str]]:
    """Prompt for IV intermittent medications using one-shot prompting (multi-turn)."""
    return _build_messages_multi_turn(
        IV_INTERMITTENT_SYSTEM_PROMPT_ONE_SHOT,
        IV_INTERMITTENT_EXAMPLE_USER,
        IV_INTERMITTENT_EXAMPLE_ASSISTANT,
        medication_prompt,
    )


# Backward compatibility: deprecated, use build_iv_intermit_messages_zero_shot() instead
def build_iv_intermit_messages(medication_prompt: str) -> List[Dict[str, str]]:
    """Prompt for IV intermittent medications using zero-shot prompting (deprecated)."""
    return build_iv_intermit_messages_zero_shot(medication_prompt)


def build_iv_push_messages_zero_shot(medication_prompt: str) -> List[Dict[str, str]]:
    """Prompt for IV push medications using zero-shot prompting."""
    return _build_messages(IV_PUSH_INSTRUCTION_ZERO_SHOT, medication_prompt)


def build_iv_push_messages_one_shot(medication_prompt: str) -> List[Dict[str, str]]:
    """Prompt for IV push medications using one-shot prompting (single-turn, deprecated)."""
    return _build_messages(IV_PUSH_INSTRUCTION_ONE_SHOT, medication_prompt)


def build_iv_push_messages_one_shot_multi_turn(medication_prompt: str) -> List[Dict[str, str]]:
    """Prompt for IV push medications using one-shot prompting (multi-turn)."""
    return _build_messages_multi_turn(
        IV_PUSH_SYSTEM_PROMPT_ONE_SHOT,
        IV_PUSH_EXAMPLE_USER,
        IV_PUSH_EXAMPLE_ASSISTANT,
        medication_prompt,
    )


# Backward compatibility: deprecated, use build_iv_push_messages_zero_shot() instead
def build_iv_push_messages(medication_prompt: str) -> List[Dict[str, str]]:
    """Prompt for IV push medications using zero-shot prompting (deprecated)."""
    return build_iv_push_messages_zero_shot(medication_prompt)


def build_iv_continuous_messages_zero_shot(medication_prompt: str) -> List[Dict[str, str]]:
    """Prompt for IV continuous medications using zero-shot prompting."""
    return _build_messages(IV_CONTINUOUS_INSTRUCTION_ZERO_SHOT, medication_prompt)


def build_iv_continuous_messages_one_shot(medication_prompt: str) -> List[Dict[str, str]]:
    """Prompt for IV continuous medications using one-shot prompting (single-turn, deprecated)."""
    return _build_messages(IV_CONTINUOUS_INSTRUCTION_ONE_SHOT, medication_prompt)


def build_iv_continuous_messages_one_shot_multi_turn(medication_prompt: str) -> List[Dict[str, str]]:
    """Prompt for IV continuous medications using one-shot prompting (multi-turn, titratable example)."""
    return _build_messages_multi_turn(
        IV_CONTINUOUS_SYSTEM_PROMPT_ONE_SHOT,
        IV_CONTINUOUS_EXAMPLE_USER_TITRATABLE,
        IV_CONTINUOUS_EXAMPLE_ASSISTANT_TITRATABLE,
        medication_prompt,
    )


def build_iv_continuous_messages_two_shot_multi_turn(medication_prompt: str) -> List[Dict[str, str]]:
    """Prompt for IV continuous medications using two-shot prompting (multi-turn, both examples)."""
    return _build_messages_multi_turn_two_shot(
        IV_CONTINUOUS_SYSTEM_PROMPT_ONE_SHOT,
        IV_CONTINUOUS_EXAMPLE_USER_TITRATABLE,
        IV_CONTINUOUS_EXAMPLE_ASSISTANT_TITRATABLE,
        IV_CONTINUOUS_EXAMPLE_USER_NON_TITRATABLE,
        IV_CONTINUOUS_EXAMPLE_ASSISTANT_NON_TITRATABLE,
        medication_prompt,
    )


# Backward compatibility: deprecated, use build_iv_continuous_messages_zero_shot() instead
def build_iv_continuous_messages(medication_prompt: str) -> List[Dict[str, str]]:
    """Prompt for IV continuous medications using zero-shot prompting (deprecated)."""
    return build_iv_continuous_messages_zero_shot(medication_prompt)


# Route selection system prompt
ROUTE_SELECTION_SYSTEM_PROMPT = "You are a clinical pharmacist. Select the most appropriate medication administration route."

# Route selection prompt
ROUTE_SELECTION_PROMPT = (
    "Below is a medication order sentence. Please select the route that is most appropriate for this order "
    "(by mouth, intravenous push, intravenous intermittent, or intravenous continuous). "
    "Return only the route and no additional text."
)


COT_REASON_SYSTEM_PROMPT = (
    "You are a clinical pharmacist. Think step by step and be precise with numbers and units."
)

COT_REMOTE_REASON_PROMPTS = {
    "IV intermittent (16)": """\
You are a clinical pharmacist. Analyze the following IV intermittent medication order step by step.

For each step, state what you find:
1. Drug name: identify the generic or brand name (lowercase).
2. Dose: the numeric drug dose and its unit (e.g., mg, g, units).
3. Diluent: the volume of diluent (numeric, in mL) and the compatible diluent type (e.g., 0.9% sodium chloride, D5W).
4. Infusion time: how long the infusion runs (e.g., 30 minutes, 1 hour).
5. Frequency: how often the dose is given (e.g., every 8 hours, once daily). If the order says just "daily", interpret as "once daily".

Medication order:
{prompt}

Think through each step, then summarize your findings clearly.""",
    "IV push (17)": """\
You are a clinical pharmacist. Analyze the following IV push medication order step by step.

For each step, state what you find:
1. Drug name: identify the generic or brand name (lowercase).
2. Dose: the numeric drug dose and its unit (e.g., mg, mcg, units).
3. Volume: the numeric volume administered (in mL).
4. Concentration: calculate the concentration per 1 mL. If the order says "20 mg/2 mL", the concentration is 10 mg/mL. If "50 mcg/mL", it is 50 mcg/mL. Always express as [number] [unit]/mL.
5. Formulation: the dosage form — always "vial solution" if the order mentions a vial.
6. Frequency: how often the push is given. If the order says just "daily", interpret as "once daily".

Medication order:
{prompt}

Think through each step, then summarize your findings clearly.""",
    "IV continuous (16)": """\
You are a clinical pharmacist. Analyze the following IV continuous infusion medication order step by step.

For each step, state what you find:
1. Drug name: identify the generic or brand name (lowercase).
2. Total dose in the bag: This is the TOTAL amount of drug in the prepared bag, NOT the per-mL concentration.
3. Diluent volume: the total volume of the bag in mL.
4. Compatible diluent type: the IV fluid (e.g., 0.9% sodium chloride, D5W).
5. Starting rate: the initial infusion rate and its unit.
6. Titratable or non-titratable?
7. If titratable: titration dose, titration unit, titration frequency, and titration goal.

Medication order:
{prompt}

Think through each step, then summarize your findings clearly.""",
}

COT_LOCAL_REASON_PROMPTS = {
    "IV intermittent (16)": """\
You are a clinical pharmacist. Analyze the following IV intermittent medication order step by step.

For each step, state what you find:
1. Drug name.
2. Dose (numeric value and unit).
3. Diluent volume and compatible diluent type.
4. Infusion time.
5. Frequency.

Medication order:
{prompt}

Think through each step, then summarize your findings clearly.""",
    "IV push (17)": """\
You are a clinical pharmacist. Analyze the following IV push medication order step by step.

For each step, state what you find:
1. Drug name.
2. Dose (numeric value and unit).
3. Volume administered.
4. Concentration of solution.
5. Formulation in canonical MedMatch wording.
6. Frequency in canonical MedMatch wording.

Medication order:
{prompt}

Think through each step, then summarize your findings clearly.""",
    "IV continuous (16)": """\
You are a clinical pharmacist. Analyze the following IV continuous infusion medication order step by step.

For each step, state what you find:
1. Drug name.
2. Drug dose in the prepared infusion (numeric value and unit).
3. Diluent volume.
4. Compatible diluent type.
5. Starting rate (value and unit).
6. Is this order titratable or non-titratable, based on the order text?
7. If titratable: titration dose, titration unit, titration frequency, and titration goal.

Medication order:
{prompt}

Think through each step, then summarize your findings clearly.""",
}

COT_REMOTE_EXTRACT_TEMPLATE = """\
Based on the clinical analysis below, extract the medication information into the MedMatch JSON format.

ANALYSIS:
{reasoning}

ORIGINAL ORDER:
{prompt}

INSTRUCTIONS:
{base_instruction}

Return one JSON object only.
Do not wrap the JSON in markdown.
Use exactly these keys in this order: {keys}.
{extra_guidance}
Now produce the MedMatch JSON:"""

COT_LOCAL_EXTRACT_TEMPLATE = """\
Based on the clinical analysis below, extract the medication information into the MedMatch JSON format.

ANALYSIS:
{reasoning}

ORIGINAL ORDER:
{prompt}

INSTRUCTIONS:
{base_instruction}

Return one JSON object only.
Do not wrap the JSON in markdown.
Use exactly these keys in this order: {keys}.
Now produce the MedMatch JSON:"""

COT_REMOTE_EXTRACT_GUIDANCE = {
    "IV intermittent (16)": 'If frequency is just "daily", write "once daily".',
    "IV push (17)": (
        "For concentration: always convert to per-1-mL basis. "
        'Formulation should be "vial solution" when a vial is mentioned. '
        'If frequency is just "daily", write "once daily".'
    ),
    "IV continuous (16)": (
        "Use the TOTAL drug amount in the bag, not the per-mL concentration. "
        "Normalize hour to hr in rate units. "
        "For non-titratable infusions, leave titration fields as empty strings."
    ),
}

COT_LOCAL_EXTRACT_GUIDANCE = {
    "IV push (17)": """\
Canonicalization for IV push:
- If the source says vial or vial solution, output formulation as "vial solution".
- Normalize frequency wording to MedMatch canonical form:
  - daily -> once daily
  - BID -> twice daily""",
}

REMOTE_NORMALIZATION_ORAL_INSTRUCTIONS = {
    "PO Solid (40)": """Please review the narratives about medications and format them into the MedMatch JSON format. Follow this exact slot order; if a slot is unknown, use an empty string and do not fabricate.

The MedMatch JSON format for oral solid dosage form medications is:
[drug name][numerical dose][abbreviated unit strength of dose][amount][formulation] by mouth [frequency]

[drug name]: The generic or brand name of the medication.
[numerical dose]: The numeric value of the strength per unit.
[abbreviated unit strength of dose]: The standardized abbreviated unit associated with the dose.
[amount]: The number of dosage units taken per administration.
[formulation]: The oral solid dosage form. Copy dosage-form wording from the order as closely as possible, including qualifiers such as extended-release or delayed-release.
by mouth: The route of administration, fixed as oral.
[frequency]: How often the medication is taken. Preserve the full schedule phrase, including qualifiers such as as needed, at bedtime, or indication text if present.""",
    "PO liquid (10)": """Please review the narratives about medications and format them into the MedMatch JSON format. Follow this exact slot order; if a slot is unknown, use an empty string and do not fabricate.

The MedMatch JSON format for oral liquid dosage form medications is:
[drug name][numerical dose][abbreviated unit strength of dose][numerical volume][abbreviated unit strength of volume] of the [concentration of formulation][formulation unit of measure][formulation] by mouth [frequency]

[drug name]: The generic or brand name of the medication.
[numerical dose]: The numeric amount of drug administered per dose.
[abbreviated unit strength of dose]: The standardized abbreviated unit for the drug dose.
[numerical volume]: The numeric volume administered per dose.
[abbreviated unit strength of volume]: The standardized abbreviated unit for volume.
[concentration of formulation]: The strength of the medication per 1 mL.
[formulation unit of measure]: The unit used in the concentration denominator.
[formulation]: The oral liquid dosage form.
[route]: The route of administration, fixed as oral.
[frequency]: How often the medication is administered.""",
}

REMOTE_ORAL_NORMALIZE_PROMPT = """You are a clinical pharmacist reviewing a structured medication order JSON for formatting consistency. The JSON was extracted from a medication order sentence. Your job is to normalize the wording without changing the medical meaning.

Apply these normalization rules:
1. Frequency: If the schedule means once per day, write "once daily" (not just "daily"). Preserve all qualifiers like "as needed", "at bedtime", indication text, and day-of-week schedules.
2. Formulation: Use hyphens for multi-word dosage forms and singular form unless the amount field is greater than 1.
3. Route: Always write "by mouth" (not "oral" or "po").
4. Units: Use standard abbreviations: mg, mcg, g, mL, mg/mL.
5. Do not change numeric values, drug names, or any medical content. Only fix formatting and wording.

Original medication order sentence:
{sentence}

Extracted JSON to normalize:
{raw_json}

Return the normalized JSON only. No extra text."""

REMOTE_IV_NORMALIZE_PROMPT = """You are a clinical pharmacist reviewing a structured IV medication order JSON for formatting consistency. The JSON was extracted from a medication order sentence. Your job is to normalize the wording without changing the medical meaning.

Apply these normalization rules:
1. Frequency canonical form — write schedules in spelled-out canonical English.
2. Infusion time — use the noun form "X minutes" or "X hours".
3. IV push formulation — for IV push orders, the canonical formulation is "vial solution" unless explicitly specified otherwise.
4. Compound dose units — preserve qualifiers like "/kg" exactly as they appear.
5. Concentration per 1 mL (IV push only) — if the order expresses X unit / Y mL with Y greater than 1, reduce to per-mL concentration.
6. Drug name typography — use ASCII hyphens only.
7. Do not change numeric values (other than rule 5), do not change drug identity, and do not add or remove fields. Return every key that appeared in the input JSON with the same key spelling.

Original medication order sentence:
{sentence}

Extracted JSON to normalize:
{raw_json}

Return the normalized JSON object only. No extra text, no markdown fences."""


def get_cot_reason_system_prompt() -> str:
    """Return the shared CoT reasoning system prompt."""
    return COT_REASON_SYSTEM_PROMPT


def build_cot_reason_prompt(sheet_name: str, medication_prompt: str, *, remote_mode: bool) -> str:
    """Return the CoT reasoning prompt for a given IV sheet."""
    prompts = COT_REMOTE_REASON_PROMPTS if remote_mode else COT_LOCAL_REASON_PROMPTS
    return prompts[sheet_name].format(prompt=medication_prompt)


def build_cot_extract_prompt(
    sheet_name: str,
    reasoning: str,
    medication_prompt: str,
    base_instruction: str,
    expected_keys: List[str],
    *,
    remote_mode: bool,
) -> str:
    """Build the CoT extraction prompt from the centralized prompt definitions."""
    template = COT_REMOTE_EXTRACT_TEMPLATE if remote_mode else COT_LOCAL_EXTRACT_TEMPLATE
    guidance_map = COT_REMOTE_EXTRACT_GUIDANCE if remote_mode else COT_LOCAL_EXTRACT_GUIDANCE
    return template.format(
        reasoning=reasoning,
        prompt=medication_prompt,
        base_instruction=base_instruction,
        keys=", ".join(expected_keys),
        extra_guidance=guidance_map.get(sheet_name, ""),
    )


def build_remote_normalization_oral_instruction(sheet_name: str) -> str:
    """Return the remote oral extraction instruction used by normalization runs."""
    return REMOTE_NORMALIZATION_ORAL_INSTRUCTIONS[sheet_name]


def build_remote_normalization_prompt(sentence: str, raw_json: str, *, family: str) -> str:
    """Return the remote normalization prompt for oral or IV runs."""
    template = REMOTE_ORAL_NORMALIZE_PROMPT if family == "oral" else REMOTE_IV_NORMALIZE_PROMPT
    return template.format(sentence=sentence, raw_json=raw_json)


def build_route_selection_messages(medication_prompt: str) -> List[Dict[str, str]]:
    """Build messages for route selection task."""
    return [
        {"role": "system", "content": ROUTE_SELECTION_SYSTEM_PROMPT},
        {"role": "user", "content": f"{ROUTE_SELECTION_PROMPT}\n\n{medication_prompt}"}
    ]
