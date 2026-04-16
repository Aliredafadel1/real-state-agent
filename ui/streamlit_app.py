"""
Main Streamlit app for AI Real Estate Agent.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import time
from typing import Any

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
render_property_focus_panel = _components.render_property_focus_panel
render_sidebar_controls = _components.render_sidebar_controls
render_charts = _components.render_charts


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
    st.set_page_config(page_title="Real Estate Agent", layout="wide")
    _init_state()

    css_path = Path(__file__).parent / "styles.css"
    load_css(css_path)

    extract_features, predict_price = resolve_backend_functions()
    explain_prediction = resolve_explanation_function()

    render_header()

    with st.sidebar:
        if st.button("New chat", use_container_width=True, help="Clear this conversation"):
            st.session_state["latest_features"] = {}
            st.session_state["latest_price"] = None
            st.session_state["latest_explanation"] = ""
            st.session_state.pop("show_charts_panel", None)
            st.session_state.pop("show_why_price", None)
            st.rerun()

    use_manual_input, manual_features, manual_run = render_sidebar_controls()

    if manual_run and use_manual_input:
        with st.spinner("Analyzing neighborhood..."):
            time.sleep(0.8)
        with st.spinner("Comparing similar homes..."):
            time.sleep(0.8)
        with st.spinner("Running ML model..."):
            time.sleep(0.8)
            _handle_manual_estimate(
                predict_price=predict_price,
                explain_prediction=explain_prediction,
                manual_features=manual_features,
            )
        st.rerun()

    active_features = manual_features if use_manual_input else st.session_state["latest_features"]
    latest_price = st.session_state.get("latest_price")
    latest_explanation = st.session_state.get("latest_explanation") or ""
    _, focus_col, right_col = st.columns([0.02, 1.2, 0.9], gap="large")

    with focus_col:
        render_property_focus_panel(
            image_path=Path(__file__).parent / "assets" / "house.jpg",
            active_features=active_features,
            latest_price=latest_price,
            latest_explanation=latest_explanation,
        )
        action_col_1, action_col_2 = st.columns(2)
        with action_col_1:
            if st.button("Show Charts", use_container_width=True):
                st.session_state["show_charts_panel"] = not st.session_state.get("show_charts_panel", False)
        with action_col_2:
            if st.button("Explain Price", use_container_width=True):
                st.session_state["show_why_price"] = not st.session_state.get("show_why_price", False)

        if (
            st.session_state.get("show_charts_panel")
            and latest_price is not None
            and active_features
        ):
            with st.expander("Model Insights", expanded=True):
                render_charts(latest_price, active_features)

        if st.session_state.get("show_why_price") and latest_explanation:
            with st.expander("Why this price?", expanded=True):
                st.write(latest_explanation)

    with right_col:
        prompt = render_ai_input_panel()
        predict_from_text = st.button("Predict from Description", use_container_width=True)
        if predict_from_text and prompt:
            with st.spinner("Analyzing neighborhood..."):
                time.sleep(0.8)
            with st.spinner("Comparing similar homes..."):
                time.sleep(0.8)
            with st.spinner("Running ML model..."):
                time.sleep(0.8)
                _handle_text_estimate(
                    extract_features=extract_features,
                    predict_price=predict_price,
                    explain_prediction=explain_prediction,
                    prompt=prompt,
                )
            st.rerun()
        elif predict_from_text:
            st.info("Please add a short property description first.")


if __name__ == "__main__":
    main()
