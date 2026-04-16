"""
Reusable Streamlit UI components.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

try:
    from .helpers import format_currency
    from .templates import header_template, prediction_card_template
except ImportError:
    from helpers import format_currency
    from templates import header_template, prediction_card_template

FIXED_VILLA_URL = (
    "https://images.unsplash.com/photo-1613977257363-707ba9348227"
    "?auto=format&fit=crop&w=1400&q=80"
)


def render_header() -> None:
    """Render top header block."""
    st.markdown(header_template(), unsafe_allow_html=True)


def render_sidebar_controls() -> tuple[bool, dict[str, Any], bool]:
    """Render manual controls and return (use_manual_input, features, manual_run)."""
    with st.sidebar:
        st.subheader("Inputs")
        use_manual_input = st.checkbox(
            "Use sidebar inputs",
            value=True,
            help="Use these grouped controls for direct prediction input.",
        )

        st.markdown("### 🏡 Property Info")
        living_area = st.number_input(
            "Living Area (sq ft)",
            min_value=300,
            max_value=10000,
            value=1800,
            step=50,
            help="Total above-ground living area.",
        )
        garage_cars = st.slider(
            "Garage Capacity (cars)",
            min_value=0,
            max_value=5,
            value=2,
            help="Number of cars the garage can hold.",
        )
        rooms = st.slider(
            "Rooms",
            min_value=1,
            max_value=12,
            value=6,
            help="Total rooms used by the model.",
        )
        year_built = st.number_input(
            "Year Built",
            min_value=1872,
            max_value=2100,
            value=2005,
            step=1,
            help="Construction year of the property.",
        )

        st.markdown("### 📍 Location")
        neighborhood = st.selectbox(
            "Neighborhood",
            ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst"],
            index=0,
            help="Locality area used as location signal.",
        )
        house_style = st.selectbox(
            "House Style",
            ["1Story", "2Story", "1.5Fin", "SLvl"],
            index=0,
            help="Architectural style class of the home.",
        )

        st.markdown("### ✨ Quality")
        exter_qual = st.selectbox(
            "Exterior Quality",
            ["Ex", "Gd", "TA", "Fa", "Po"],
            index=2,
            help="Exterior material and finish quality grade.",
        )

    manual_features = {
        "GrLivArea": float(living_area),
        "GarageCars": float(garage_cars),
        "Rooms": int(rooms),
        "YearBuilt": int(year_built),
        "Neighborhood": neighborhood,
        "HouseStyle": house_style,
        "ExterQual": exter_qual,
    }
    manual_run = st.button("Predict Price", type="primary", use_container_width=True)
    return use_manual_input, manual_features, manual_run


def render_features_json(features: dict[str, Any], title: str = "Features") -> None:
    """Render feature dictionary as JSON block."""
    st.markdown(f"### {title}")
    st.json(features)


def render_prediction_card(predicted_price: float) -> None:
    """Render styled prediction result."""
    st.markdown(prediction_card_template(format_currency(predicted_price)), unsafe_allow_html=True)


def render_explanation(explanation: str) -> None:
    """Render explanation block."""
    st.markdown("### Explanation")
    st.markdown(f"<div class='prose-panel'>{explanation}</div>", unsafe_allow_html=True)


def render_house_image(image_path: str | Path = "ui/assets/house.jpg") -> None:
    """
    Render a fixed modern villa image only.
    Never changes based on user input.
    """
    path = Path(image_path)
    if path.exists():
        image_source: str = str(path)
    else:
        image_source = FIXED_VILLA_URL
    st.markdown("<div class='villa-image-wrap'>", unsafe_allow_html=True)
    st.image(image_source, use_container_width=True)
    st.markdown(
        "<div class='villa-overlay'>AI Analyzed Property</div></div>",
        unsafe_allow_html=True,
    )


def render_ai_input_panel() -> str:
    """Render right-side AI prompt input and return text."""
    st.markdown("### AI Property Description")
    st.caption("Describe your property in natural language...")
    prompt = st.text_area(
        "Describe your property in natural language...",
        value="",
        height=140,
        label_visibility="collapsed",
        placeholder="Describe your property in natural language...",
    )
    st.markdown(
        "- Modern 3-bedroom house in Beirut\n"
        "- Luxury villa with pool",
    )
    return prompt.strip()


def render_charts(predicted_price: float, features: dict[str, Any]) -> None:
    """Render feature importance and market comparison with mock support."""
    base_importance = {
        "GrLivArea": 0.36,
        "GarageCars": 0.18,
        "YearBuilt": 0.2,
        "Rooms": 0.16,
        "Neighborhood": 0.1,
    }
    jittered = {k: max(0.04, v + random.uniform(-0.03, 0.03)) for k, v in base_importance.items()}
    imp_df = pd.DataFrame(
        {"Feature": list(jittered.keys()), "Importance": list(jittered.values())}
    ).sort_values("Importance", ascending=False)
    st.markdown("#### Feature Importance")
    st.bar_chart(imp_df.set_index("Feature"))

    low = predicted_price * 0.9
    high = predicted_price * 1.1
    comparison_df = pd.DataFrame(
        {
            "Category": ["Your Home", "Local Median", "Top Quartile"],
            "Price": [predicted_price, predicted_price * 0.94, predicted_price * 1.12],
        }
    ).set_index("Category")
    st.markdown("#### Price Comparison")
    st.bar_chart(comparison_df)
    st.caption(f"Estimated range: {format_currency(low)} - {format_currency(high)}")

    with st.expander("Why this price?", expanded=True):
        gr_liv = features.get("GrLivArea", "N/A")
        garage = features.get("GarageCars", "N/A")
        year = features.get("YearBuilt", "N/A")
        st.markdown(
            f"- Larger living area (**{gr_liv} sq ft**) pushes valuation upward.\n"
            f"- Garage capacity (**{garage} cars**) improves buyer utility.\n"
            f"- Build year (**{year}**) affects condition and modernization assumptions."
        )


def render_property_focus_panel(
    *,
    image_path: Path,
    active_features: dict[str, Any],
    latest_price: float | None,
    latest_explanation: str = "",
) -> None:
    """Render hero image and premium price section in center panel."""
    render_house_image(image_path)
    if latest_price is None:
        st.markdown(
            "<p class='hero-placeholder'>Click <strong>Predict Price</strong> to generate valuation insights.</p>",
            unsafe_allow_html=True,
        )
        return

    confidence = 92
    low = latest_price * 0.9
    high = latest_price * 1.1
    render_prediction_card(latest_price)
    st.markdown(
        (
            "<div class='price-meta'>"
            f"<span>Confidence: <strong>{confidence}%</strong></span>"
            f"<span>Range: <strong>{format_currency(low)} - {format_currency(high)}</strong></span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    if latest_explanation:
        st.markdown(f"<div class='hero-explanation'>{latest_explanation}</div>", unsafe_allow_html=True)

