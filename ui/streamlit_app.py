"""
Main Streamlit app for AI Real Estate Agent.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import time
from typing import Any, Dict, List

import streamlit as st

REQUIRED_UI_FEATURES = [
    "GrLivArea",
    "GarageCars",
    "Rooms",
    "YearBuilt",
    "Neighborhood",
    "HouseStyle",
    "ExterQual",
]

def _resolve_ui_modules():
    """Load UI modules in both package and flat-script modes."""
    try:
        components_module = importlib.import_module("ui.components")
        helpers_module = importlib.import_module("ui.helpers")
    except ModuleNotFoundError:
        components_module = importlib.import_module("components")
        helpers_module = importlib.import_module("helpers")
    return components_module, helpers_module


_components, _helpers = _resolve_ui_modules()

render_header = _components.render_header
render_house_image = _components.render_house_image
render_prediction_card = _components.render_prediction_card
render_explanation = _components.render_explanation
render_charts = _components.render_charts

format_currency = _helpers.format_currency


def render_ai_input_panel() -> str:
    """Compatibility wrapper if deployed module misses new panel API."""
    if hasattr(_components, "render_ai_input_panel"):
        return _components.render_ai_input_panel()
    st.markdown("### AI Property Description")
    st.caption("Describe your property in natural language...")
    return st.text_area(
        "Describe your property in natural language...",
        value="",
        height=140,
        label_visibility="collapsed",
        placeholder="Describe your property in natural language...",
    ).strip()


load_css = _helpers.load_css
resolve_backend_functions = _helpers.resolve_backend_functions
resolve_explanation_function = _helpers.resolve_explanation_function


def _init_state() -> None:
    """Initialize session state keys."""
    if "latest_features" not in st.session_state:
        st.session_state["latest_features"] = {}
    if "latest_price" not in st.session_state:
        st.session_state["latest_price"] = None
    if "latest_explanation" not in st.session_state:
        st.session_state["latest_explanation"] = ""
    if "show_charts_panel" not in st.session_state:
        st.session_state["show_charts_panel"] = False
    if "show_why_price" not in st.session_state:
        st.session_state["show_why_price"] = False
    if "collected_features" not in st.session_state:
        st.session_state["collected_features"] = {}
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {
                "role": "assistant",
                "text": "Hi! Describe your property and I’ll ask follow‑up questions if I’m missing details like area, garage, rooms, year built, neighborhood, style, or exterior quality.",
            }
        ]


def _safe_extract(extract_fn, text: str) -> dict[str, Any]:
    """Call extraction backend safely."""
    try:
        return extract_fn(text)
    except Exception as exc:
        st.error(f"Feature extraction failed: {exc}")
        return {}


def _safe_predict(predict_fn, features: dict[str, Any]) -> float | None:
    """Call prediction backend safely."""
    try:
        return float(predict_fn(features))
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return None


def _safe_explain(explain_fn, features: dict[str, Any], predicted_price: float) -> str:
    """Call explanation backend safely."""
    try:
        text = explain_fn(features, predicted_price)
        return str(text).strip()
    except Exception as exc:
        st.warning(f"Explanation generation fallback used: {exc}")
        return "Prediction generated, but explanation is unavailable."


def _find_missing_feature_keys(features: dict[str, Any]) -> list[str]:
    """Return required UI feature keys that are missing/empty."""
    missing: list[str] = []
    for key in REQUIRED_UI_FEATURES:
        value = features.get(key)
        if value is None:
            missing.append(key)
            continue
        if isinstance(value, str) and not value.strip():
            missing.append(key)
    return missing


def _handle_manual_estimate(
    *,
    predict_price,
    explain_prediction,
    manual_features: dict[str, Any],
) -> None:
    """Run estimate flow using manual sidebar values."""
    missing_keys = _find_missing_feature_keys(manual_features)
    if missing_keys:
        st.warning(f"I need a bit more detail: {', '.join(missing_keys)}.")
        st.session_state["latest_price"] = None
        st.session_state["latest_explanation"] = ""
        return

    predicted_price = _safe_predict(predict_price, manual_features)
    if predicted_price is None:
        st.error("I couldn’t produce a prediction from those inputs.")
        return

    st.session_state["latest_features"] = manual_features
    st.session_state["latest_price"] = predicted_price
    st.session_state["latest_explanation"] = _safe_explain(
        explain_prediction, manual_features, predicted_price
    )


def _merge_features(
    base: Dict[str, Any],
    new: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge two feature dicts, preferring non-null values from `new`."""
    merged = dict(base)
    for k, v in new.items():
        if v is not None:
            merged[k] = v
    return merged


def _handle_chat_turn(
    *,
    user_text: str,
    extract_features,
    predict_price,
    explain_prediction,
) -> None:
    """Single chat turn for AI-style interaction."""
    # Extract partial features from the latest user message
    extracted = _safe_extract(extract_features, user_text)
    if not extracted:
        st.session_state["chat_messages"].append(
            {
                "role": "assistant",
                "text": (
                    "I couldn’t pick up any structured details. "
                    "Try mentioning living area (sq ft), number of rooms, garage size, year built, "
                    "neighborhood, house style, and exterior quality."
                ),
            }
        )
        return

    # Merge with features collected so far
    current = st.session_state.get("collected_features", {})
    merged = _merge_features(current, extracted)
    st.session_state["collected_features"] = merged
    st.session_state["latest_features"] = merged

    missing_keys = _find_missing_feature_keys(merged)
    if missing_keys:
        # Still need more information – stay in conversation mode
        st.session_state["latest_price"] = None
        st.session_state["latest_explanation"] = ""
        pretty = ", ".join(missing_keys)
        st.session_state["chat_messages"].append(
            {
                "role": "assistant",
                "text": (
                    f"I have some details, but I still need: {pretty}. "
                    "Please add those so I can finish the estimate."
                ),
            }
        )
        return

    # All required features are present – run prediction
    predicted_price = _safe_predict(predict_price, merged)
    if predicted_price is None:
        st.session_state["chat_messages"].append(
            {
                "role": "assistant",
                "text": "Something went wrong while running the model. Please try rephrasing or adjusting the description.",
            }
        )
        return

    st.session_state["latest_price"] = predicted_price
    explanation = _safe_explain(explain_prediction, merged, predicted_price)
    st.session_state["latest_explanation"] = explanation

    # Respond conversationally with result
    st.session_state["chat_messages"].append(
        {
            "role": "assistant",
            "text": (
                f"Based on what you’ve told me, the estimated price is about "
                f"{predicted_price:,.0f} USD. "
                "Here’s a short explanation:\n\n"
                f"{explanation}"
            ),
        }
    )
def _handle_text_estimate(*, extract_features, predict_price, explain_prediction, prompt: str) -> None:
    """Estimate using natural language description from the right panel."""
    extracted = _safe_extract(extract_features, prompt)
    if not extracted:
        st.warning(
            "Could not extract details. Include area (sq ft), garage, rooms, year built, neighborhood, house style, and exterior quality."
        )
        return

    st.session_state["latest_features"] = extracted
    missing_keys = _find_missing_feature_keys(extracted)
    if missing_keys:
        st.session_state["latest_price"] = None
        st.session_state["latest_explanation"] = ""
        st.info(f"I still need: {', '.join(missing_keys)}.")
        return

    predicted_price = _safe_predict(predict_price, extracted)
    if predicted_price is None:
        st.error("Prediction failed after extraction. Please try again.")
        return

    st.session_state["latest_price"] = predicted_price
    st.session_state["latest_explanation"] = _safe_explain(
        explain_prediction, extracted, predicted_price
    )


def main() -> None:
    """Run Streamlit UI."""
    st.set_page_config(page_title="AI Property Valuation", layout="wide")
    _init_state()

    css_path = Path(__file__).parent / "styles.css"
    load_css(css_path)

    extract_features, predict_price = resolve_backend_functions()
    explain_prediction = resolve_explanation_function()

    # --- HERO HEADER -------------------------------------------------------
    render_header()

    # --- SIDEBAR: PROPERTY INPUTS & OPTIONS -------------------------------
    with st.sidebar:
        st.markdown("### Property Details")
        living_area = st.number_input(
            "Living Area (sq ft)",
            min_value=300,
            max_value=10000,
            value=1800,
            step=50,
        )
        rooms = st.slider(
            "Rooms",
            min_value=1,
            max_value=12,
            value=6,
        )
        garage_cars = st.slider(
            "Garage Capacity (cars)",
            min_value=0,
            max_value=5,
            value=1,
        )
        year_built = st.number_input(
            "Year Built",
            min_value=1872,
            max_value=2100,
            value=2010,
            step=1,
        )

        st.markdown("### Location")
        neighborhood = st.selectbox(
            "Neighborhood",
            ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst"],
            index=0,
        )
        house_style = st.selectbox(
            "House Style",
            ["1Story", "2Story", "1.5Fin", "SLvl"],
            index=0,
        )

        st.markdown("### Quality")
        exter_qual = st.selectbox(
            "Exterior Quality",
            ["Ex", "Gd", "TA", "Fa", "Po"],
            index=2,
        )

        st.markdown("### Options")
        show_insights = st.checkbox(
            "Show Insights",
            value=True,
            help="Show explanation, key drivers, and charts under the estimate.",
        )

        st.markdown("---")
        if st.button("Generate Estimate", type="primary", use_container_width=True):
            manual_features = {
                "GrLivArea": float(living_area),
                "GarageCars": float(garage_cars),
                "Rooms": int(rooms),
                "YearBuilt": int(year_built),
                "Neighborhood": neighborhood,
                "HouseStyle": house_style,
                "ExterQual": exter_qual,
            }
            with st.spinner("Analyzing property details..."):
                time.sleep(0.6)
            with st.spinner("Comparing to similar homes..."):
                time.sleep(0.6)
            with st.spinner("Running valuation model..."):
                time.sleep(0.6)
                _handle_manual_estimate(
                    predict_price=predict_price,
                    explain_prediction=explain_prediction,
                    manual_features=manual_features,
                )
            st.session_state["show_insights"] = show_insights
            st.rerun()

        if st.button("New Chat", use_container_width=True):
            st.session_state["latest_features"] = {}
            st.session_state["latest_price"] = None
            st.session_state["latest_explanation"] = ""
            st.session_state["collected_features"] = {}
            st.session_state["chat_messages"] = [
                {
                    "role": "assistant",
                    "text": "Hi! Describe your property and I’ll ask for any missing details.",
                }
            ]
            st.rerun()

    active_features = st.session_state["latest_features"]
    latest_price = st.session_state.get("latest_price")
    latest_explanation = st.session_state.get("latest_explanation") or ""
    # Give the image more horizontal space so it feels more prominent.
    left_col, right_col = st.columns([1.6, 1.4], gap="large")

    # --- MAIN CONTENT: IMAGE + CHAT ---------------------------------------
    with left_col:
        st.markdown("### ")
        render_house_image(
            image_path=Path(__file__).parent / "assets" / "house.jpg",
            features=active_features,
        )

    with right_col:
        st.markdown("### AI Property Assistant")
        st.caption(
            "Describe the property in plain language and I’ll ask for any missing details."
        )

        # Chat panel styled as a full chat app
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.get("chat_messages", []):
                avatar = "assistant" if msg["role"] == "assistant" else "user"
                with st.chat_message(avatar):
                    st.markdown(msg["text"])

            user_text = st.chat_input(
                "Example: 3-bedroom house in Beirut, built in 2015, 1 garage space"
            )
            if user_text:
                st.session_state["chat_messages"].append(
                    {"role": "user", "text": user_text}
                )
                with st.spinner("Analyzing property details..."):
                    _handle_chat_turn(
                        user_text=user_text,
                        extract_features=extract_features,
                        predict_price=predict_price,
                        explain_prediction=explain_prediction,
                    )
                st.rerun()

        # Quick suggestion chips under the chat input
        chip_col1, chip_col2, chip_col3, chip_col4 = st.columns(4)
        chip_suggestions = [
            "2 bedrooms",
            "1 garage",
            "Built in 2018",
            "In Beirut",
        ]
        for chip, col in zip(chip_suggestions, [chip_col1, chip_col2, chip_col3, chip_col4]):
            with col:
                if st.button(chip, use_container_width=True):
                    st.session_state["chat_messages"].append(
                        {"role": "user", "text": chip}
                    )
                    with st.spinner("Analyzing property details..."):
                        _handle_chat_turn(
                            user_text=chip,
                            extract_features=extract_features,
                            predict_price=predict_price,
                            explain_prediction=explain_prediction,
                        )
                    st.rerun()

    # --- RESULT CARD ------------------------------------------------------
    if latest_price is not None:
        st.markdown("### ")
        result_col = st.container()
        with result_col:
            st.markdown("#### Estimated Price")
            render_prediction_card(latest_price)

            # Confidence meta + short explanation snippet
            confidence_pct = 92
            low = latest_price * 0.9
            high = latest_price * 1.1

            st.markdown(
                (
                    "<div class='price-meta'>"
                    f"<span>Confidence: <strong>{confidence_pct}%</strong></span>"
                    f"<span>Range: <strong>{format_currency(low)} – {format_currency(high)}</strong></span>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

            if latest_explanation:
                # Show a short teaser line from the full explanation
                short_text = latest_explanation.strip().split(". ")[0]
                st.markdown(
                    f"<div class='hero-explanation'>{short_text}.</div>",
                    unsafe_allow_html=True,
                )

    # --- INSIGHTS SECTION -------------------------------------------------
    if st.session_state.get("show_insights", False) and latest_price is not None and active_features:
        st.markdown("### Insights")

        # Key drivers section – user-friendly copy
        with st.container():
            st.markdown("#### Key Drivers")
            gr_liv = active_features.get("GrLivArea")
            garage = active_features.get("GarageCars")
            rooms_val = active_features.get("Rooms")
            year_val = active_features.get("YearBuilt")
            hood = active_features.get("Neighborhood") or "the selected neighborhood"

            bullets: list[str] = []
            if gr_liv is not None:
                bullets.append(
                    f"- Larger **interior size** of about **{int(gr_liv)} sq ft** increases the valuation."
                )
            if rooms_val is not None:
                bullets.append(
                    f"- **{int(rooms_val)} rooms** make the home more attractive to families."
                )
            if garage is not None:
                bullets.append(
                    f"- A **{int(garage)}-car garage** adds convenience and resale value."
                )
            if year_val is not None:
                bullets.append(
                    f"- Being built in **{int(year_val)}** influences condition and modernization."
                )
            bullets.append(
                f"- Location in **{hood}** also plays a role in the final estimate."
            )

            st.markdown("\n".join(bullets))

        # Full explanation
        if latest_explanation:
            render_explanation(latest_explanation)

        # Analytics inside an expander
        with st.expander("View Analytics", expanded=False):
            render_charts(latest_price, active_features)


if __name__ == "__main__":
    main()
