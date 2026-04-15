"""
Lightweight LLM client switchboard.

Provides a small abstraction to choose which LLM provider to call.
Currently includes a `mock` provider and placeholders for real providers.

Usage:
    from app.llm_client import call_llm
    response_text = call_llm(prompt, provider="mock")
"""
from __future__ import annotations

import json
import os
from typing import Optional

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "mock")


MOCK_RESPONSE = {
    "overall_qual": 8,
    "gr_liv_area": 2500.0,
    "garage_cars": 2.0,
    "total_bsmt_sf": 1000.0,
    "full_bath": 2,
    "year_built": 2005,
    "neighborhood": None,
    "house_style": "ranch",
    "garage_type": "attached",
    "exter_qual": "good",
}


def get_default_provider() -> str:
    """Return default provider name (env LLM_PROVIDER or 'mock')."""
    return os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)


def list_providers() -> list[str]:
    """List available provider keys."""
    return ["mock", "openai", "anthropic"]


def call_llm(prompt: str, provider: Optional[str] = None) -> str:
    """Call the chosen LLM provider and return the raw text response.

    This is a switchboard-style helper. For now the function returns a
    JSON string from the `mock` provider. Placeholders are present for
    real provider implementations; replace them with actual API calls.
    """
    provider = provider or get_default_provider()

    if provider == "mock":
        return json.dumps(MOCK_RESPONSE)

    if provider == "openai":
        # Placeholder: implement OpenAI call here (openai.ChatCompletion or client)
        # For now return the mock response so downstream code can be exercised.
        return json.dumps(MOCK_RESPONSE)

    if provider == "anthropic":
        # Placeholder for Anthropic/Claude integration
        return json.dumps(MOCK_RESPONSE)

    raise ValueError(f"Unknown LLM provider: {provider}")


__all__ = ["call_llm", "get_default_provider", "list_providers"]
