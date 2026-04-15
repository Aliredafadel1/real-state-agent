"""
LLM Stage 1: Extract features from user input.

This module uses an LLM to parse user queries and extract house features
relevant to the ML model.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schemas import ExtractedFeatures


# Features expected in Stage 1
REQUIRED_FIELDS = [
    "overall_qual",
    "gr_liv_area",
    "garage_cars",
    "total_bsmt_sf",
    "full_bath",
    "year_built",
    "neighborhood",
    "house_style",
    "garage_type",
    "exter_qual",
]


# Example feature mapping for the LLM
FEATURE_MAPPING = {
    "overall_qual": "overall quality rating (1-10)",
    "gr_liv_area": "above ground living area in square feet",
    "garage_cars": "number of garage car spaces",
    "total_bsmt_sf": "total basement square footage",
    "full_bath": "number of full bathrooms",
    "year_built": "year the house was built",
    "neighborhood": "neighborhood name",
    "house_style": "house style (e.g., 1Story, 2Story, ranch, colonial)",
    "garage_type": "garage type (e.g., attached, detached)",
    "exter_qual": "exterior quality (e.g., Ex, Gd, TA, Fa, Po)",
}


def build_extraction_prompt(user_query: str) -> str:
    """
    Build a prompt for the LLM to extract features from user input.
    """
    features_description = "\n".join(
        [f"- {key}: {value}" for key, value in FEATURE_MAPPING.items()]
    )

    prompt = f"""You are an expert at extracting house features from user descriptions.

Given the user's description, extract as many of the following features as possible:

{features_description}

Rules:
1. Return valid JSON only.
2. Do not guess missing values.
3. Use null for missing or unknown values.
4. Return ONLY the feature keys listed below.
5. Do not include explanations or extra text.

User input: "{user_query}"

Return JSON with exactly these keys:
{{
    "overall_qual": null,
    "gr_liv_area": null,
    "garage_cars": null,
    "total_bsmt_sf": null,
    "full_bath": null,
    "year_built": null,
    "neighborhood": null,
    "house_style": null,
    "garage_type": null,
    "exter_qual": null
}}
"""
    return prompt


def safe_json_extract(text: str) -> dict[str, Any]:
    """
    Safely extract JSON from LLM output.

    Handles:
    - pure JSON
    - extra text before/after JSON
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError("No valid JSON object found in LLM response.")


def normalize_categorical_values(data: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize categorical values to be closer to training categories.
    """
    garage_type_map = {
        "attached": "Attchd",
        "detached": "Detchd",
        "built in": "BuiltIn",
        "builtin": "BuiltIn",
        "car port": "CarPort",
        "carport": "CarPort",
        "basement": "Basment",
        "2 types": "2Types",
    }

    exter_qual_map = {
        "excellent": "Ex",
        "ex": "Ex",
        "very good": "Gd",
        "good": "Gd",
        "gd": "Gd",
        "typical": "TA",
        "average": "TA",
        "ta": "TA",
        "fair": "Fa",
        "fa": "Fa",
        "poor": "Po",
        "po": "Po",
    }

    house_style_map = {
        "1 story": "1Story",
        "one story": "1Story",
        "1story": "1Story",
        "2 story": "2Story",
        "two story": "2Story",
        "2story": "2Story",
        "1.5 story": "1.5Fin",
        "one and half story": "1.5Fin",
        "ranch": "1Story",
    }

    if isinstance(data.get("garage_type"), str):
        value = data["garage_type"].strip().lower()
        data["garage_type"] = garage_type_map.get(value, data["garage_type"])

    if isinstance(data.get("exter_qual"), str):
        value = data["exter_qual"].strip().lower()
        data["exter_qual"] = exter_qual_map.get(value, data["exter_qual"])

    if isinstance(data.get("house_style"), str):
        value = data["house_style"].strip().lower()
        data["house_style"] = house_style_map.get(value, data["house_style"])

    return data


def compute_field_completeness(extracted: ExtractedFeatures) -> ExtractedFeatures:
    """
    Compute provided_fields and missing_fields from actual extracted values.
    """
    data = extracted.model_dump()

    provided_fields = [field for field in REQUIRED_FIELDS if data.get(field) is not None]
    missing_fields = [field for field in REQUIRED_FIELDS if data.get(field) is None]

    extracted.provided_fields = provided_fields
    extracted.missing_fields = missing_fields

    return extracted


def extract_features_from_llm_response(
    llm_response: str,
    user_query: str
) -> ExtractedFeatures:
    """
    Parse LLM response and convert to ExtractedFeatures schema.
    """
    try:
        response_data = safe_json_extract(llm_response)
        response_data = normalize_categorical_values(response_data)

        extracted = ExtractedFeatures(
            overall_qual=response_data.get("overall_qual"),
            gr_liv_area=response_data.get("gr_liv_area"),
            garage_cars=response_data.get("garage_cars"),
            total_bsmt_sf=response_data.get("total_bsmt_sf"),
            full_bath=response_data.get("full_bath"),
            year_built=response_data.get("year_built"),
            neighborhood=response_data.get("neighborhood"),
            house_style=response_data.get("house_style"),
            garage_type=response_data.get("garage_type"),
            exter_qual=response_data.get("exter_qual"),
        )

        extracted = compute_field_completeness(extracted)
        return extracted

    except Exception as e:
        print(f"⚠️ Stage 1 parsing failed for query: {user_query}")
        print(f"Reason: {e}")

        # Safe fallback
        return ExtractedFeatures(
            provided_fields=[],
            missing_fields=REQUIRED_FIELDS.copy()
        )


def stage1_extract_features(user_query: str) -> ExtractedFeatures:
    """
    Main Stage 1 function: Extract features from user input.

    This is a placeholder that builds the prompt. In production, replace the
    mock response with a real LLM API call.
    """
    prompt = build_extraction_prompt(user_query)

    print("=" * 60)
    print("STAGE 1: Feature Extraction Prompt")
    print("=" * 60)
    print(prompt)
    print("=" * 60)

    # Use the llm client switchboard to get a response (defaults to mock)
    from app.llm_client import call_llm

    llm_response = call_llm(prompt)

    extracted = extract_features_from_llm_response(
        llm_response,
        user_query
    )

    return extracted


if __name__ == "__main__":
    test_query = "I have a 2005 ranch house with 2500 sq ft of living space and a 2-car attached garage with 1000 sq ft basement and good exterior quality"

    print(f"\nUser Query: {test_query}\n")

    features = stage1_extract_features(test_query)

    print("\nExtracted Features:")
    print(features.model_dump_json(indent=2))
    print(f"\nProvided fields: {features.provided_fields}")
    print(f"Missing fields: {features.missing_fields}")