"""
HTML templates used by Streamlit components.
"""

from __future__ import annotations


def header_template() -> str:
    """Return app header HTML."""
    return """
    <div class="app-header chatgpt-topbar">
      <div class="chatgpt-brand">
        <span class="chatgpt-logo" aria-hidden="true">RE</span>
        <div>
          <h1>Real Estate Agent</h1>
          <p class="chatgpt-subtitle">
            Natural-language estimates with clear explanations. Use the sidebar for precise manual inputs when you need them.
          </p>
        </div>
      </div>
    </div>
    """


def prediction_card_template(price_text: str) -> str:
    """Return prediction card HTML with formatted price."""
    return f"""
    <div class="prediction-card">
      <div class="prediction-label">Estimated value</div>
      <div class="prediction-value">{price_text}</div>
    </div>
    """
