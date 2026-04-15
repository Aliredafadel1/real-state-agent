"""
LLM Stage 2: Explain model prediction in human-readable language.

This module builds a grounded prompt from inference artifacts and asks the LLM
for a short explanation paragraph. It is designed to avoid fabricated details
and to gracefully fall back when the LLM call fails.
"""

from __future__ import annotations

from typing import Any, Optional

from app.llm_client import call_llm


def _fmt_currency(value: float) -> str:
    """Format price values as USD-like currency."""
    return f"${value:,.0f}"


def _compact_dict(data: dict[str, Any]) -> str:
    """Render a dict as stable key/value lines for prompt grounding."""
    if not data:
        return "(none)"
    return "\n".join(f"- {k}: {v}" for k, v in data.items())


def _compact_list(items: list[str]) -> str:
    """Render a list for prompt grounding."""
    if not items:
        return "(none)"
    return ", ".join(items)


def build_stage2_prompt(
    extracted_features: dict[str, Any],
    model_input: dict[str, Any],
    predicted_price: float,
    missing_fields: list[str],
    dataset_summary: Optional[dict[str, Any]] = None,
) -> str:
    """
    Build a grounded prompt for Stage 2 explanation generation.

    Parameters
    ----------
    extracted_features:
        Raw structured features extracted from user text (Stage 1 output).
    model_input:
        Final transformed model-ready features used for inference, expected to
        include keys like OverallQual, GrLivArea, GarageCars, TotalBsmtSF,
        FullBath, YearBuilt, TotalSF, HouseAge, Neighborhood, HouseStyle,
        GarageType, ExterQual.
    predicted_price:
        Price predicted by the ML model.
    missing_fields:
        Fields not provided by the user (or unavailable), which increase
        uncertainty.
    dataset_summary:
        Optional context (e.g., target distribution or neighborhood-level
        statistics) already computed by your pipeline.

    Returns
    -------
    str
        Prompt instructing the LLM to provide one short, grounded paragraph.
    """
    dataset_summary_text = _compact_dict(dataset_summary or {})

    prompt = f"""You are an assistant that explains a house price prediction.

Use ONLY the data provided below. Do not invent facts, values, comparisons, or assumptions.
If information is missing, explicitly state that uncertainty is higher.

Predicted price:
- {_fmt_currency(predicted_price)}

Extracted features (from user text):
{_compact_dict(extracted_features)}

Model input features (actually used by model):
{_compact_dict(model_input)}

Missing fields:
- {_compact_list(missing_fields)}

Optional dataset context:
{dataset_summary_text}

Task:
Write one short paragraph (3-5 sentences) that:
1) clearly states the predicted price,
2) highlights key contributing features from model_input,
3) mentions missing fields and resulting uncertainty,
4) stays strictly grounded in provided values.

Output rules:
- Return plain text only.
- Do NOT return JSON.
- Do NOT include bullet points.
"""
    return prompt


def generate_explanation(
    extracted_features: dict[str, Any],
    model_input: dict[str, Any],
    predicted_price: float,
    missing_fields: list[str],
    dataset_summary: Optional[dict[str, Any]] = None,
    provider: Optional[str] = None,
) -> str:
    """
    Generate Stage 2 explanation text from model outputs.

    This function builds the prompt, calls the configured LLM provider via
    `call_llm`, and returns a cleaned paragraph. If the provider fails or
    returns an invalid result, a deterministic fallback explanation is returned.

    Parameters
    ----------
    extracted_features:
        Raw structured features extracted from user text.
    model_input:
        Final model-ready feature dictionary used for prediction.
    predicted_price:
        Predicted house price from the ML model.
    missing_fields:
        Missing user inputs that may increase uncertainty.
    dataset_summary:
        Optional dataset-level context for explanation grounding.
    provider:
        Optional provider override passed to `call_llm` (e.g., "ollama").

    Returns
    -------
    str
        Human-readable explanation paragraph.
    """
    prompt = build_stage2_prompt(
        extracted_features=extracted_features,
        model_input=model_input,
        predicted_price=predicted_price,
        missing_fields=missing_fields,
        dataset_summary=dataset_summary,
    )

    fallback = (
        f"The model predicted a price of {_fmt_currency(predicted_price)}, "
        "but a detailed explanation is unavailable."
    )

    try:
        raw = call_llm(prompt, provider=provider)
    except Exception:
        return fallback

    if not isinstance(raw, str):
        return fallback

    cleaned = " ".join(raw.strip().split())
    if not cleaned:
        return fallback

    return cleaned


__all__ = ["build_stage2_prompt", "generate_explanation"]
