"""
Utility helpers for Streamlit UI.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Callable

import streamlit as st


def load_css(css_path: str | Path) -> None:
    """Load CSS file contents into Streamlit page."""
    path = Path(css_path)
    if not path.exists():
        st.warning(f"CSS file not found: {path}")
        return
    st.markdown(f"<style>{path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def format_currency(value: float) -> str:
    """Format numeric value as currency."""
    return f"${value:,.0f}"


def demo_extract_features(text: str) -> dict[str, Any]:
    """
    Demo extractor used when project backend function is unavailable.
    """
    lower = text.lower()

    features: dict[str, Any] = {
        "GrLivArea": None,
        "GarageCars": None,
        "Rooms": None,
        "YearBuilt": None,
        "Neighborhood": None,
        "HouseStyle": None,
        "ExterQual": None,
    }

    area_match = re.search(r"(\d{3,5})\s*(sq\s*ft|sqft)", lower)
    if area_match:
        features["GrLivArea"] = float(area_match.group(1))

    garage_match = re.search(r"(\d+)\s*[- ]?car\s+garage", lower)
    if garage_match:
        features["GarageCars"] = float(garage_match.group(1))
    elif "garage" in lower:
        features["GarageCars"] = 1.0

    room_match = re.search(r"(\d+)\s*[- ]?(bedroom|bedrooms|rooms|room)", lower)
    if room_match:
        features["Rooms"] = int(room_match.group(1))

    year_match = re.search(r"\b(18\d{2}|19\d{2}|20\d{2}|2100)\b", lower)
    if year_match:
        features["YearBuilt"] = int(year_match.group(1))

    neighborhood_match = re.search(r"\bin\s+([a-zA-Z][a-zA-Z0-9_-]*)", text)
    if neighborhood_match:
        features["Neighborhood"] = neighborhood_match.group(1).strip()

    if "ranch" in lower:
        features["HouseStyle"] = "1Story"
    elif "2 story" in lower or "two story" in lower or "2story" in lower:
        features["HouseStyle"] = "2Story"
    elif "1 story" in lower or "one story" in lower or "1story" in lower:
        features["HouseStyle"] = "1Story"

    if "excellent" in lower or "exterior ex" in lower:
        features["ExterQual"] = "Ex"
    elif "good" in lower:
        features["ExterQual"] = "Gd"
    elif "average" in lower or "typical" in lower:
        features["ExterQual"] = "TA"
    elif "fair" in lower:
        features["ExterQual"] = "Fa"
    elif "poor" in lower:
        features["ExterQual"] = "Po"

    return features


def demo_predict_price(features: dict[str, Any]) -> float:
    """
    Demo predictor used when project backend function is unavailable.
    """
    base = 120000
    area_component = float(features.get("GrLivArea", 1500)) * 65
    garage_component = float(features.get("GarageCars", 0)) * 12000
    room_component = float(features.get("Rooms", 5)) * 5000
    year_component = (int(features.get("YearBuilt", 2000)) - 1990) * 900
    quality_boost = 14000 if str(features.get("ExterQual", "TA")) in {"Gd", "Ex"} else 0
    return base + area_component + garage_component + room_component + year_component + quality_boost


def demo_generate_explanation(features: dict[str, Any], predicted_price: float) -> str:
    """Demo explanation used when Stage 2 backend is unavailable."""
    area = features.get("GrLivArea", "N/A")
    garage = features.get("GarageCars", "N/A")
    year = features.get("YearBuilt", "N/A")
    return (
        f"The predicted price is {format_currency(predicted_price)} based on the provided home profile. "
        f"Key drivers include living area ({area}), garage capacity ({garage}), and year built ({year}). "
        "This estimate may vary if additional property details are provided."
    )


def resolve_backend_functions() -> tuple[Callable[[str], dict[str, Any]], Callable[[dict[str, Any]], float]]:
    """
    Resolve backend functions with graceful fallback.

    Expected backend contract:
      - extract_features(text: str) -> dict
      - predict_price(features: dict) -> float
    """
    try:
        from app.backend import extract_features, predict_price  # type: ignore

        return extract_features, predict_price
    except Exception:
        return demo_extract_features, demo_predict_price


def resolve_explanation_function() -> Callable[[dict[str, Any], float], str]:
    """
    Resolve Stage 2 explanation function with graceful fallback.
    """
    try:
        from app.llm_stage2 import generate_explanation  # type: ignore

        def _generate(features: dict[str, Any], predicted_price: float) -> str:
            return generate_explanation(
                extracted_features=features,
                model_input=features,
                predicted_price=predicted_price,
                missing_fields=[],
            )

        return _generate
    except Exception:
        return demo_generate_explanation

