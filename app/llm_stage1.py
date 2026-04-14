"""
LLM Stage 1: Extract features from user input.

This module uses an LLM to parse user queries and extract house features
relevant to the ML model.
"""

import json
import sys
from pathlib import Path
from typing import Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schemas import ExtractedFeatures


# Example feature mapping for the LLM
FEATURE_MAPPING = {
    "overall_qual": "overall quality rating (1-10)",
    "gr_liv_area": "above ground living area in square feet",
    "garage_cars": "number of garage car spaces",
    "total_bsmt_sf": "total basement square footage",
    "full_bath": "number of full bathrooms",
    "year_built": "year the house was built",
    "neighborhood": "neighborhood name",
    "house_style": "house style (e.g., ranch, colonial)",
    "garage_type": "garage type (e.g., attached, detached)",
    "exter_qual": "exterior quality",
}


def build_extraction_prompt(user_query: str) -> str:
    """
    Build a prompt for the LLM to extract features from user input.
    
    Args:
        user_query: User's natural language description of a house
        
    Returns:
        Formatted prompt for the LLM
    """
    features_description = "\n".join(
        [f"- {key}: {value}" for key, value in FEATURE_MAPPING.items()]
    )
    
    prompt = f"""You are an expert at extracting house features from user descriptions.

Given the user's description, extract as many of the following features as possible:

{features_description}

User input: "{user_query}"

Return a JSON object with:
1. Extracted features (use null for missing values)
2. List of provided_fields (which fields were found)
3. List of missing_fields (which fields are still needed)

Example format:
{{
    "overall_qual": 8,
    "gr_liv_area": 2500.0,
    "garage_cars": 2.0,
    "total_bsmt_sf": 1000.0,
    "full_bath": 2,
    "year_built": 2005,
    "neighborhood": "Downtown",
    "house_style": "Ranch",
    "garage_type": "Attached",
    "exter_qual": "Good",
    "provided_fields": ["overall_qual", "gr_liv_area", "garage_cars"],
    "missing_fields": ["total_bsmt_sf", "full_bath", "year_built", "neighborhood", "house_style", "garage_type", "exter_qual"]
}}

Return ONLY the JSON object, no additional text."""
    
    return prompt


def extract_features_from_llm_response(
    llm_response: str,
    user_query: str
) -> ExtractedFeatures:
    """
    Parse LLM response and convert to ExtractedFeatures schema.
    
    Args:
        llm_response: Raw response from LLM (should be JSON)
        user_query: Original user query (for fallback/debugging)
        
    Returns:
        ExtractedFeatures object
        
    Raises:
        ValueError: If response cannot be parsed as JSON
    """
    try:
        # Parse JSON from LLM response
        response_data = json.loads(llm_response)
        
        # Extract provided and missing fields
        provided_fields = response_data.pop("provided_fields", [])
        missing_fields = response_data.pop("missing_fields", [])
        
        # Create ExtractedFeatures with remaining fields
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
            provided_fields=provided_fields,
            missing_fields=missing_fields,
        )
        
        return extracted
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}")


def stage1_extract_features(user_query: str) -> ExtractedFeatures:
    """
    Main Stage 1 function: Extract features from user input.
    
    This is a placeholder that builds the prompt. In production, you would
    call your actual LLM API here (e.g., OpenAI, Claude, Cohere).
    
    Args:
        user_query: User's natural language house description
        
    Returns:
        ExtractedFeatures with parsed information
    """
    # Build the extraction prompt
    prompt = build_extraction_prompt(user_query)
    
    print("=" * 60)
    print("STAGE 1: Feature Extraction Prompt")
    print("=" * 60)
    print(prompt)
    print("=" * 60)
    
    # TODO: Replace with actual LLM API call
    # Example for production:
    # response = openai.ChatCompletion.create(
    #     model="gpt-4",
    #     messages=[{"role": "user", "content": prompt}],
    #     temperature=0.1,
    # )
    # llm_response = response.choices[0].message.content
    
    # For now, return mock response
    mock_response = {
        "overall_qual": 8,
        "gr_liv_area": 2500.0,
        "garage_cars": 2.0,
        "total_bsmt_sf": 1000.0,
        "full_bath": 2,
        "year_built": 2005,
        "neighborhood": None,
        "house_style": None,
        "garage_type": None,
        "exter_qual": None,
        "provided_fields": ["overall_qual", "gr_liv_area", "garage_cars", "total_bsmt_sf", "full_bath", "year_built"],
        "missing_fields": ["neighborhood", "house_style", "garage_type", "exter_qual"]
    }
    
    extracted = extract_features_from_llm_response(
        json.dumps(mock_response),
        user_query
    )
    
    return extracted


if __name__ == "__main__":
    # Test Stage 1
    test_query = "I have a 2005 ranch house with 2500 sq ft of living space and a 2-car garage with 1000 sq ft basement"
    
    print(f"\nUser Query: {test_query}\n")
    
    features = stage1_extract_features(test_query)
    
    print("\nExtracted Features:")
    print(features.model_dump_json(indent=2))
    print(f"\nProvided fields: {features.provided_fields}")
    print(f"Missing fields: {features.missing_fields}")
