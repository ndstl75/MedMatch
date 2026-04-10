"""
RougeRx GPT-4o-mini Prompts for Medication Component Extraction

This file contains all prompts used for extracting medication components
directly from healthcare provider responses using GPT-4o-mini.
"""

# Component extraction prompts
COMPONENT_EXTRACTION_PROMPTS = {
    'drug_name': """Extract the medication/drug name from this text: "{response_text}"

Return ONLY the drug name, nothing else. Examples:
Text: "aspirin 81 mg daily" → aspirin
Text: "D5W @ 100 mL/hour IV" → D5W
Text: "Morphine 4 mg IV q4h" → Morphine
Text: "normal saline 500 mL" → normal saline

If no drug name is provided in the text, return an empty string (not "none", "N/A", or any other placeholder).""",

    'dose': """Extract the dose amount (just the number) from this text: "{response_text}"

Return ONLY the numeric dose value, nothing else. Examples:
Text: "aspirin 81 mg daily" → 81
Text: "D5W @ 100 mL/hour IV" → 100
Text: "Morphine 4 mg IV q4h" → 4
Text: "normal saline 500 mL" → 500

If no dose is provided in the text, return an empty string (not "none", "N/A", or any other placeholder).""",

    'unit': """Extract the dose unit from this text: "{response_text}"

Return ONLY the unit, nothing else. Examples:
Text: "aspirin 81 mg daily" → mg
Text: "D5W @ 100 mL/hour IV" → mL
Text: "Morphine 4 mg IV q4h" → mg
Text: "insulin 50 units" → units

If no unit is provided in the text, return an empty string (not "none", "N/A", or any other placeholder).""",

    'route': """Extract the administration route from this text: "{response_text}"

IMPORTANT: Extract the route EXACTLY as written in the text. Do NOT convert phrases to abbreviations.
- If the text says "by mouth", extract "by mouth" (not "PO")
- If the text says "IV", extract "IV" (as written)
- If the text says "subcutaneous", extract "subcutaneous" (not "SC")

Return ONLY the route exactly as written, nothing else. Examples:
Text: "aspirin 81 mg by mouth daily" → by mouth
Text: "D5W @ 100 mL/hour IV" → IV
Text: "Morphine 4 mg IV q4h" → IV
Text: "insulin 50 units SC" → SC
Text: "medication given subcutaneously" → subcutaneously

If no route is provided in the text, return an empty string (not "none", "N/A", or any other placeholder).""",

    'frequency': """Extract the complete dosing frequency from this text: "{response_text}"

Return ONLY the full frequency phrase, including any time specifications. Examples:
Text: "aspirin 81 mg daily" → daily
Text: "Morphine 4 mg IV q4h" → q4h
Text: "insulin 50 units BID" → BID
Text: "pain medication PRN" → PRN
Text: "carvedilol 25 mg tablet, 1 tablet by mouth twice daily at 0900 and 2100" → twice daily at 0900 and 2100
Text: "medication twice daily in the morning and evening" → twice daily in the morning and evening

If no frequency is provided in the text, return an empty string (not "none", "N/A", "No frequency provided", or any other placeholder)."""
}

# System prompt for all extractions
EXTRACTION_SYSTEM_PROMPT = """You are a medication information extraction expert. 
- Extract exactly what is asked and return only that information.
- Extract text exactly as written in the source (do not convert phrases to abbreviations).
- If a component is missing, return an empty string (not "none", "N/A", or any placeholder)."""

def get_component_extraction_prompt(component_name: str, response_text: str) -> str:
    """
    Get the extraction prompt for a specific component.

    Args:
        component_name: Name of component to extract
        response_text: The response text to extract from

    Returns:
        Formatted prompt string
    """
    base_prompt = COMPONENT_EXTRACTION_PROMPTS.get(component_name, f"Extract {component_name} from: {response_text}")
    return base_prompt.format(response_text=response_text)
