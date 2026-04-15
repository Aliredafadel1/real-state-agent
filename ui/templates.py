"""
HTML templates used by Streamlit components.
"""

from __future__ import annotations


def header_template() -> str:
    """Return app header HTML."""
    return """
    <div class="app-header card">
      <h1>AI Real Estate Agent</h1>
      <p>Describe a property in natural language or use manual controls to estimate price.</p>
    </div>
    """


def prediction_card_template(price_text: str) -> str:
    """Return prediction card HTML with formatted price."""
    return f"""
    <div class="prediction-card card">
      <div class="prediction-label">Predicted House Price</div>
      <div class="prediction-value">{price_text}</div>
    </div>
    """

