"""
Main Streamlit app for AI Real Estate Agent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

try:
    from ui.components import (
        render_charts,
        render_chat_messages,
        render_features_json,
        render_header,
        render_house_image,
        render_prediction_card,
        render_sidebar_controls,
    )
    from ui.helpers import load_css, resolve_backend_functions
except ModuleNotFoundError:
    from components import (
        render_charts,
        render_chat_messages,
        render_features_json,
        render_header,
        render_house_image,
        render_prediction_card,
        render_sidebar_controls,
    )
    from helpers import load_css, resolve_backend_functions


def _init_state() -> None:
    """Initialize session state keys."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "latest_features" not in st.session_state:
        st.session_state["latest_features"] = {}
    if "latest_price" not in st.session_state:
        st.session_state["latest_price"] = None


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


def main() -> None:
    """Run Streamlit UI."""
    st.set_page_config(page_title="AI Real Estate Agent", layout="wide")
    _init_state()

    css_path = Path(__file__).parent / "styles.css"
    load_css(css_path)

    extract_features, predict_price = resolve_backend_functions()

    render_header()

    use_manual_input, manual_features, uploaded_image = render_sidebar_controls()

    left, right = st.columns([1.25, 1], gap="large")

    with left:
        render_chat_messages()

        prompt = st.chat_input("Describe the house you want to evaluate...")
        if prompt:
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            extracted = _safe_extract(extract_features, prompt)
            if extracted:
                st.session_state["messages"].append(
                    {"role": "assistant", "content": "Here are the extracted features."}
                )
                with st.chat_message("assistant"):
                    st.markdown("Here are the extracted features.")
                st.session_state["latest_features"] = extracted

    with right:
        render_house_image(Path(__file__).parent / "assets" / "house.jpg", uploaded_image=uploaded_image)

        active_features = manual_features if use_manual_input else st.session_state["latest_features"]
        if active_features:
            predicted_price = _safe_predict(predict_price, active_features)
            if predicted_price is not None:
                st.session_state["latest_price"] = predicted_price
                render_prediction_card(predicted_price)
                render_charts(predicted_price, active_features)
            render_features_json(
                active_features,
                title="Manual Features" if use_manual_input else "Extracted Features",
            )
        else:
            st.info("Submit a house description in chat or enable manual input from the sidebar.")


if __name__ == "__main__":
    main()

