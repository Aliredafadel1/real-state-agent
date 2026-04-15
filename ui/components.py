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

HOUSE_IMAGE_PRESETS = {
    "None": "",
    "Modern Villa": "https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?auto=format&fit=crop&w=1200&q=80",
    "Suburban Family Home": "https://images.unsplash.com/photo-1568605114967-8130f3a36994?auto=format&fit=crop&w=1200&q=80",
    "Luxury Contemporary House": "https://images.unsplash.com/photo-1613977257363-707ba9348227?auto=format&fit=crop&w=1200&q=80",
}


def render_header() -> None:
    """Render top header block."""
    st.markdown(header_template(), unsafe_allow_html=True)


def render_sidebar_controls() -> tuple[bool, dict[str, Any], Any, str, str]:
    """Render manual controls and return (use_manual_input, features, uploaded_image, image_url, selected_preset)."""
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
        image_url = st.text_input(
            "Or use house image URL",
            value="",
            placeholder="https://.../house.jpg",
            help="Optional: paste a direct image URL for a more realistic house photo.",
        )
        selected_preset = st.selectbox(
            "Or choose realistic house preset",
            options=list(HOUSE_IMAGE_PRESETS.keys()),
            index=0,
            help="Quickly switch to a realistic house photo preset.",
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
    return use_manual_input, manual_features, uploaded_image, image_url.strip(), selected_preset


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


def render_explanation(explanation: str) -> None:
    """Render explanation block."""
    st.markdown("### Prediction Explanation")
    st.markdown(f"<div class='card'>{explanation}</div>", unsafe_allow_html=True)


def render_house_image(
    image_path: str | Path = "ui/assets/house.jpg",
    uploaded_image: Any = None,
    image_url: str = "",
    preset_name: str = "None",
) -> None:
    """Render house image if available, else show warning."""
    if uploaded_image is not None:
        st.image(uploaded_image, use_container_width=True, caption="House Preview")
        return

    if image_url:
        st.image(image_url, use_container_width=True, caption="House Preview")
        return

    preset_url = HOUSE_IMAGE_PRESETS.get(preset_name, "")
    if preset_url:
        st.image(preset_url, use_container_width=True, caption=f"House Preview - {preset_name}")
        return

    path = Path(image_path)
    if path.exists():
        st.image(str(path), use_container_width=True, caption="House Preview")
    else:
        fallback_name = "Suburban Family Home"
        fallback_url = HOUSE_IMAGE_PRESETS.get(fallback_name, "")
        if fallback_url:
            st.image(
                fallback_url,
                use_container_width=True,
                caption=f"House Preview - {fallback_name}",
            )
            st.caption("Local image not found, using preset fallback.")
        else:
            st.info(f"House image not found at `{path}`.")


def render_charts(predicted_price: float, features: dict[str, Any]) -> None:
    """Render optional chart section."""
    with st.expander("Show Charts"):
        if "chart_mode" not in st.session_state:
            st.session_state["chart_mode"] = "chart"

        btn_col_1, btn_col_2 = st.columns(2)
        with btn_col_1:
            if st.button("Chart", use_container_width=True):
                st.session_state["chart_mode"] = "chart"
        with btn_col_2:
            if st.button("Histogram", use_container_width=True):
                st.session_state["chart_mode"] = "histogram"

        avg_price = 220000.0
        target_df = pd.DataFrame(
            {"Category": ["Predicted Price", "Average Price"], "Price": [predicted_price, avg_price]}
        ).set_index("Category")

        numeric_feature_keys = ["GrLivArea", "GarageCars", "Rooms", "YearBuilt"]
        numeric_values = []
        for key in numeric_feature_keys:
            value = features.get(key)
            if isinstance(value, (int, float)):
                numeric_values.append(float(value))

        if st.session_state["chart_mode"] == "chart":
            st.markdown("#### Target Price Comparison")
            st.bar_chart(target_df)

            numeric_rows = []
            for key in numeric_feature_keys:
                value = features.get(key)
                if isinstance(value, (int, float)):
                    numeric_rows.append({"Feature": key, "Value": float(value)})
            if numeric_rows:
                st.markdown("#### Key Numeric Features")
                feature_df = pd.DataFrame(numeric_rows).set_index("Feature")
                st.bar_chart(feature_df)
        else:
            st.markdown("#### Numeric Feature Histogram")
            if numeric_values:
                hist_df = pd.DataFrame({"Value": numeric_values})
                st.bar_chart(hist_df["Value"].value_counts(bins=6).sort_index())
            else:
                st.info("Not enough numeric feature values to build histogram.")

        st.markdown("#### Feature Table")
        table_df = pd.DataFrame(list(features.items()), columns=["Feature", "Value"])
        st.dataframe(table_df, use_container_width=True, hide_index=True)

