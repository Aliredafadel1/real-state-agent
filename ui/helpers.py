"""
Utility helpers for Streamlit UI.
"""

from __future__ import annotations

from pathlib import Path
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
    living_area = 1800
    if "2500" in lower:
        living_area = 2500
    elif "2000" in lower:
        living_area = 2000

    return {
        "GrLivArea": float(living_area),
        "GarageCars": 2.0 if "garage" in lower else 0.0,
        "Rooms": 6 if "3-bedroom" in lower or "3 bedroom" in lower else 5,
        "YearBuilt": 2005 if "2005" in lower else 2000,
        "Neighborhood": "NAmes",
        "HouseStyle": "1Story",
        "ExterQual": "Gd" if "good" in lower else "TA",
    }


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

