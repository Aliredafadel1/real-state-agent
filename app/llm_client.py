"""
Lightweight LLM client switchboard.

Provides a small abstraction to choose which LLM provider to call.
Supports `mock`, `ollama`, and `anthropic` providers.

Usage:
    from app.llm_client import call_llm
    response_text = call_llm(prompt, provider="ollama")
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Optional

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")


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
    provider = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)
    return provider.strip().lower()


def list_providers() -> list[str]:
    """List available provider keys."""
    return ["mock", "ollama", "anthropic"]


def _call_ollama(prompt: str) -> str:
    """Call local Ollama API and return text output."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3.1")
    timeout_seconds = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0},
    }
    body = json.dumps(payload).encode("utf-8")
    url = f"{base_url}/api/generate"
    request = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_bytes = response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Failed to reach Ollama. Ensure Ollama is running and OLLAMA_BASE_URL is correct."
        ) from exc

    try:
        response_data = json.loads(response_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Ollama returned invalid JSON.") from exc

    output_text = response_data.get("response")
    if not isinstance(output_text, str) or not output_text.strip():
        raise RuntimeError("Ollama response contained no text output.")
    return output_text.strip()


def _call_anthropic(prompt: str) -> str:
    """Call Anthropic Messages API and return text output."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")

    model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")

    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError(
            "Anthropic SDK not installed. Add `anthropic` to requirements and install dependencies."
        ) from exc

    client = Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=1000,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    text_chunks: list[str] = []
    for block in message.content:
        if getattr(block, "type", None) == "text":
            text_chunks.append(block.text)

    output_text = "".join(text_chunks).strip()
    if not output_text:
        raise RuntimeError("Anthropic response contained no text output.")
    return output_text


def call_llm(prompt: str, provider: Optional[str] = None) -> str:
    """Call the chosen LLM provider and return the raw text response.

    The provider defaults to env `LLM_PROVIDER` and currently supports:
    - `mock`: returns deterministic JSON for local testing
    - `ollama`: calls local Ollama API
    - `anthropic`: calls Anthropic Messages API
    """
    provider = (provider or get_default_provider()).strip().lower()

    if provider == "mock":
        return json.dumps(MOCK_RESPONSE)

    if provider == "ollama":
        return _call_ollama(prompt)

    if provider == "anthropic":
        return _call_anthropic(prompt)

    raise ValueError(f"Unknown LLM provider: {provider}")


__all__ = ["call_llm", "get_default_provider", "list_providers"]
