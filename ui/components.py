"""
Reusable Streamlit UI components.
"""

from __future__ import annotations

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


def render_header() -> None:
    """Render top header block."""
    st.markdown(header_template(), unsafe_allow_html=True)


def render_sidebar_controls() -> tuple[bool, dict[str, Any], Any]:
    """Render manual controls and return (use_manual_input, features, uploaded_image)."""
    with st.sidebar:
        st.subheader("Manual Control")
        use_manual_input = st.checkbox("Use manual input", value=False)

        living_area = st.number_input("Living Area (sq ft)", min_value=300, max_value=10000, value=1800, step=50)
        garage = st.selectbox("Garage", ["Yes", "No"], index=0)
        rooms = st.slider("Rooms", min_value=1, max_value=12, value=6)
        year_built = st.number_input("Year Built", min_value=1872, max_value=2100, value=2005, step=1)
        neighborhood = st.selectbox("Neighborhood", ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst"], index=0)
        house_style = st.selectbox("House Style", ["1Story", "2Story", "1.5Fin", "SLvl"], index=0)
        exter_qual = st.selectbox("Exterior Quality", ["Ex", "Gd", "TA", "Fa", "Po"], index=2)
        st.markdown("---")
        uploaded_image = st.file_uploader(
            "Upload house image",
            type=["png", "jpg", "jpeg", "webp"],
            help="Optional: override default house image.",
        )

    manual_features = {
        "GrLivArea": float(living_area),
        "GarageCars": 2.0 if garage == "Yes" else 0.0,
        "Rooms": int(rooms),
        "YearBuilt": int(year_built),
        "Neighborhood": neighborhood,
        "HouseStyle": house_style,
        "ExterQual": exter_qual,
    }
    return use_manual_input, manual_features, uploaded_image


def render_chat_messages() -> None:
    """Render chat history from session state."""
    for item in st.session_state.get("messages", []):
        with st.chat_message(item["role"]):
            st.markdown(item["content"])


def render_features_json(features: dict[str, Any], title: str = "Features") -> None:
    """Render feature dictionary as JSON block."""
    st.markdown(f"### {title}")
    st.json(features)


def render_prediction_card(predicted_price: float) -> None:
    """Render styled prediction result."""
    st.markdown(prediction_card_template(format_currency(predicted_price)), unsafe_allow_html=True)


def render_house_image(image_path: str | Path = "ui/assets/house.jpg", uploaded_image: Any = None) -> None:
    """Render house image if available, else show warning."""
    if uploaded_image is not None:
        st.image(uploaded_image, use_container_width=True, caption="Uploaded House Image")
        return

    path = Path(image_path)
    if path.exists():
        st.image(str(path), use_container_width=True, caption="Sample House")
    else:
        st.info(f"House image not found at `{path}`.")


def render_charts(predicted_price: float, features: dict[str, Any]) -> None:
    """Render optional chart section."""
    with st.expander("Show Charts"):
        avg_price = 220000.0
        target_df = pd.DataFrame(
            {"Category": ["Predicted Price", "Average Price"], "Price": [predicted_price, avg_price]}
        ).set_index("Category")
        st.markdown("#### Target Price Comparison")
        st.bar_chart(target_df)

        numeric_feature_keys = ["GrLivArea", "GarageCars", "Rooms", "YearBuilt"]
        numeric_rows = []
        for key in numeric_feature_keys:
            value = features.get(key)
            if isinstance(value, (int, float)):
                numeric_rows.append({"Feature": key, "Value": float(value)})

        if numeric_rows:
            st.markdown("#### Key Numeric Features")
            feature_df = pd.DataFrame(numeric_rows).set_index("Feature")
            st.bar_chart(feature_df)

        st.markdown("#### Feature Table")
        table_df = pd.DataFrame(list(features.items()), columns=["Feature", "Value"])
        st.dataframe(table_df, use_container_width=True, hide_index=True)

