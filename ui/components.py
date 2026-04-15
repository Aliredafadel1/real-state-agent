"""
Reusable Streamlit UI components.
"""

from __future__ import annotations

import html
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


def render_sidebar_controls() -> tuple[bool, dict[str, Any], Any, str, str, bool]:
    """Render manual controls and return (use_manual_input, features, uploaded_image, image_url, selected_preset, manual_run)."""
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
    manual_run = False
    if use_manual_input:
        manual_run = st.button("Run estimate", type="primary", use_container_width=True)
    return use_manual_input, manual_features, uploaded_image, image_url.strip(), selected_preset, manual_run


def render_chat_messages() -> None:
    """Render chat history from session state."""
    messages = st.session_state.get("messages", [])
    if not messages:
        st.markdown(
            '<div class="chat-empty-hint">Describe a property (sq ft, beds, year, neighborhood, style, quality) '
            "or use <strong>manual input</strong> in the sidebar and click <strong>Run estimate</strong>.</div>",
            unsafe_allow_html=True,
        )
        return
    for item in messages:
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
    st.markdown("### Explanation")
    st.markdown(f"<div class='prose-panel'>{explanation}</div>", unsafe_allow_html=True)


def render_house_image(
    image_path: str | Path = "ui/assets/house.jpg",
    uploaded_image: Any = None,
    image_url: str = "",
    preset_name: str = "None",
    *,
    show_caption: bool = True,
) -> None:
    """Render house image if available, else show warning."""
    cap = "Property" if show_caption else None
    if uploaded_image is not None:
        st.image(uploaded_image, use_container_width=True, caption=cap)
        return

    if image_url:
        st.image(image_url, use_container_width=True, caption=cap)
        return

    preset_url = HOUSE_IMAGE_PRESETS.get(preset_name, "")
    if preset_url:
        preset_cap = f"{preset_name}" if show_caption else None
        st.image(preset_url, use_container_width=True, caption=preset_cap)
        return

    path = Path(image_path)
    if path.exists():
        st.image(str(path), use_container_width=True, caption=cap)
    else:
        fallback_name = "Suburban Family Home"
        fallback_url = HOUSE_IMAGE_PRESETS.get(fallback_name, "")
        if fallback_url:
            fb_cap = f"{fallback_name}" if show_caption else None
            st.image(
                fallback_url,
                use_container_width=True,
                caption=fb_cap,
            )
            if show_caption:
                st.caption("Local image not found, using preset fallback.")
        else:
            st.info(f"House image not found at `{path}`.")


def render_charts(predicted_price: float, features: dict[str, Any]) -> None:
    """Render charts (no outer expander — parent should gate visibility)."""
    if "chart_mode" not in st.session_state:
        st.session_state["chart_mode"] = "chart"

    btn_col_1, btn_col_2 = st.columns(2)
    with btn_col_1:
        if st.button("Chart", use_container_width=True, key="chart_mode_bar"):
            st.session_state["chart_mode"] = "chart"
    with btn_col_2:
        if st.button("Histogram", use_container_width=True, key="chart_mode_hist"):
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


def render_property_focus_panel(
    *,
    image_path: Path,
    uploaded_image: Any,
    image_url: str,
    selected_preset: str,
    active_features: dict[str, Any],
    latest_price: float | None,
    latest_explanation: str,
    missing_keys: list[str],
) -> None:
    """
    Left column: property photo (always visible), estimate stacked under the photo,
    details collapsed, charts only after explicit toggle.
    """
    st.caption("Listing preview")
    with st.container(border=True):
        render_house_image(
            image_path,
            uploaded_image=uploaded_image,
            image_url=image_url,
            preset_name=selected_preset,
            show_caption=False,
        )

        st.markdown("##### Estimated value")
        if latest_price is not None:
            render_prediction_card(latest_price)
            if latest_explanation:
                safe = html.escape(latest_explanation)
                st.markdown(
                    f"<div class='hero-explanation'>{safe}</div>",
                    unsafe_allow_html=True,
                )
        elif active_features and missing_keys:
            st.markdown(
                "<p class='hero-placeholder'>Add missing details in chat or the sidebar to see an estimate.</p>",
                unsafe_allow_html=True,
            )
            st.caption(f"Still needed: {', '.join(missing_keys)}")
        elif active_features:
            st.markdown(
                "<p class='hero-placeholder'>Inputs look complete — run an estimate from chat or the sidebar.</p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<p class='hero-placeholder'>Describe the property in chat or use manual input, then run an estimate.</p>",
                unsafe_allow_html=True,
            )

    if active_features:
        with st.expander("Property details", expanded=False):
            render_features_json(active_features, title="Current inputs")

    show_charts = st.checkbox(
        "Show charts",
        key="show_charts_panel",
        help="Charts stay hidden until you enable this.",
    )

    if show_charts and active_features and latest_price is not None:
        with st.container(border=True):
            st.caption("Analysis")
            render_charts(latest_price, active_features)

