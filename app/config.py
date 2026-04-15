"""
Application configuration for FastAPI backend.
"""

from __future__ import annotations

from datetime import date
from typing import Any


class Settings:
    """Runtime settings and fallback values for model input bridge."""

    APP_NAME = "AI Real Estate Agent API"
    APP_VERSION = "0.1.0"

    # Fallbacks used when Stage 1 cannot extract a required feature.
    DEFAULT_MODEL_INPUT: dict[str, Any] = {
        "OverallQual": 6,
        "GrLivArea": 1500.0,
        "GarageCars": 2.0,
        "TotalBsmtSF": 800.0,
        "FullBath": 2,
        "YearBuilt": 2000,
        "Neighborhood": "NAmes",
        "HouseStyle": "1Story",
        "GarageType": "Attchd",
        "ExterQual": "TA",
    }

    @staticmethod
    def normalize_year_built(year_built: int | None) -> int:
        """Return a safe year value for inference input."""
        current_year = date.today().year
        if year_built is None:
            return int(Settings.DEFAULT_MODEL_INPUT["YearBuilt"])
        if year_built < 1872:
            return 1872
        if year_built > current_year:
            return current_year
        return year_built


settings = Settings()

