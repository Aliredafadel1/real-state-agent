"""
Main Streamlit app for AI Real Estate Agent.
"""

from __future__ import annotations

from pathlib import Path
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

try:
    from ui.components import (
        render_chat_messages,
        render_header,
        render_property_focus_panel,
        render_sidebar_controls,
    )
    from ui.helpers import load_css, resolve_backend_functions, resolve_explanation_function
except ModuleNotFoundError:
    from components import (
        render_chat_messages,
        render_header,
        render_property_focus_panel,
        render_sidebar_controls,
    )
    from helpers import load_css, resolve_backend_functions, resolve_explanation_function


def _format_prediction_message(explanation: str) -> str:
    """Chat stays conversational; price and inputs live in the property panel beside the photo."""
    return (
        f"{explanation}\n\n"
        "*The estimated value is shown next to the property image — expand **Property details** there if you want the full input list.*"
    )


def _init_state() -> None:
    """Initialize session state keys."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "latest_features" not in st.session_state:
        st.session_state["latest_features"] = {}
    if "latest_price" not in st.session_state:
        st.session_state["latest_price"] = None
    if "latest_explanation" not in st.session_state:
        st.session_state["latest_explanation"] = ""


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
    """Append user/assistant messages for a manual sidebar estimate."""
    st.session_state["messages"].append(
        {"role": "user", "content": "Run an estimate using my manual sidebar inputs."}
    )
    missing_keys = _find_missing_feature_keys(manual_features)
    if missing_keys:
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": f"I need a bit more detail: **{', '.join(missing_keys)}**.",
            }
        )
        st.session_state["latest_price"] = None
        st.session_state["latest_explanation"] = ""
        return

    predicted_price = _safe_predict(predict_price, manual_features)
    if predicted_price is None:
        st.session_state["messages"].append(
            {"role": "assistant", "content": "I couldn’t produce a prediction from those inputs."}
        )
        return

    st.session_state["latest_features"] = manual_features
    st.session_state["latest_price"] = predicted_price
    st.session_state["latest_explanation"] = _safe_explain(
        explain_prediction, manual_features, predicted_price
    )
    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": _format_prediction_message(st.session_state["latest_explanation"]),
        }
    )


def _handle_chat_prompt(
    *,
    extract_features,
    predict_price,
    explain_prediction,
    prompt: str,
    use_manual_input: bool,
) -> None:
    """Process a chat message and append assistant reply."""
    st.session_state["messages"].append({"role": "user", "content": prompt})

    if use_manual_input:
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": "Manual input is on. Turn off **Use manual input** to chat here, "
                "or click **Run estimate** in the sidebar.",
            }
        )
        return

    extracted = _safe_extract(extract_features, prompt)
    if not extracted:
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": "I couldn’t extract property details from that. Try including living area (sq ft), "
                "garage, bedrooms, year built, neighborhood, house style, and exterior quality.",
            }
        )
        return

    st.session_state["latest_features"] = extracted
    missing_keys = _find_missing_feature_keys(extracted)
    if missing_keys:
        st.session_state["latest_price"] = None
        st.session_state["latest_explanation"] = ""
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": f"Here is what I understood so far. I still need: **{', '.join(missing_keys)}**. "
                "Reply with the missing details.",
            }
        )
        return

    predicted_price = _safe_predict(predict_price, extracted)
    if predicted_price is None:
        st.session_state["messages"].append(
            {"role": "assistant", "content": "I parsed the property, but prediction failed. Please try again."}
        )
        return

    st.session_state["latest_price"] = predicted_price
    st.session_state["latest_explanation"] = _safe_explain(
        explain_prediction, extracted, predicted_price
    )
    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": _format_prediction_message(st.session_state["latest_explanation"]),
        }
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
            st.session_state["messages"] = []
            st.session_state["latest_features"] = {}
            st.session_state["latest_price"] = None
            st.session_state["latest_explanation"] = ""
            if "chart_mode" in st.session_state:
                del st.session_state["chart_mode"]
            st.session_state.pop("show_charts_panel", None)
            st.rerun()

    use_manual_input, manual_features, uploaded_image, image_url, selected_preset, manual_run = (
        render_sidebar_controls()
    )

    if manual_run and use_manual_input:
        with st.spinner("Estimating…"):
            _handle_manual_estimate(
                predict_price=predict_price,
                explain_prediction=explain_prediction,
                manual_features=manual_features,
            )
        st.rerun()

    active_features = manual_features if use_manual_input else st.session_state["latest_features"]
    latest_price = st.session_state.get("latest_price")
    latest_explanation = st.session_state.get("latest_explanation") or ""
    missing_keys = _find_missing_feature_keys(active_features) if active_features else []

    focus_col, chat_col = st.columns([1.05, 1], gap="large")

    with focus_col:
        render_property_focus_panel(
            image_path=Path(__file__).parent / "assets" / "house.jpg",
            uploaded_image=uploaded_image,
            image_url=image_url,
            selected_preset=selected_preset,
            active_features=active_features,
            latest_price=latest_price,
            latest_explanation=latest_explanation,
            missing_keys=missing_keys,
        )

    with chat_col:
        render_chat_messages()
        prompt = st.chat_input("Describe the property…")
    if prompt:
        with st.spinner("Estimating…"):
            _handle_chat_prompt(
                extract_features=extract_features,
                predict_price=predict_price,
                explain_prediction=explain_prediction,
                prompt=prompt,
                use_manual_input=use_manual_input,
            )
        st.rerun()


if __name__ == "__main__":
    main()
