"""
API routes for end-to-end prediction pipeline.
"""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter

from app.config import settings
from app.error_handlers import (
    InferenceExecutionError,
    Stage1ExtractionError,
    Stage2ExplanationError,
)
from app.inference import predict
from app.llm_stage1 import stage1_extract_features
from app.llm_stage2 import generate_explanation
from app.schemas import (
    ExtractedFeatures,
    ModelInput,
    PredictAPIResponse,
    PredictRequest,
)

router = APIRouter()


def _to_model_input(extracted: ExtractedFeatures) -> ModelInput:
    """
    Build ModelInput from Stage 1 extracted features with safe defaults.
    """
    defaults = settings.DEFAULT_MODEL_INPUT

    year_built = settings.normalize_year_built(extracted.year_built)
    current_year = date.today().year

    gr_liv_area = float(extracted.gr_liv_area or defaults["GrLivArea"])
    total_bsmt_sf = float(extracted.total_bsmt_sf or defaults["TotalBsmtSF"])

    model_payload = {
        "OverallQual": int(extracted.overall_qual or defaults["OverallQual"]),
        "GrLivArea": gr_liv_area,
        "GarageCars": float(extracted.garage_cars or defaults["GarageCars"]),
        "TotalBsmtSF": total_bsmt_sf,
        "FullBath": int(extracted.full_bath or defaults["FullBath"]),
        "YearBuilt": year_built,
        "TotalSF": gr_liv_area + total_bsmt_sf,
        "HouseAge": max(0, current_year - year_built),
        "Neighborhood": (extracted.neighborhood or defaults["Neighborhood"]).strip(),
        "HouseStyle": (extracted.house_style or defaults["HouseStyle"]).strip(),
        "GarageType": (extracted.garage_type or defaults["GarageType"]).strip(),
        "ExterQual": (extracted.exter_qual or defaults["ExterQual"]).strip(),
    }

    return ModelInput(**model_payload)


@router.get("/", tags=["health"])
def root() -> dict[str, str]:
    """Root endpoint for quick browser checks."""
    return {"message": "AI Real Estate Agent API", "docs": "/docs", "health": "/health"}


@router.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    """Liveness endpoint."""
    return {"status": "ok"}


@router.post("/predict", response_model=PredictAPIResponse, tags=["prediction"])
def predict_price(payload: PredictRequest) -> PredictAPIResponse:
    """
    Run full prediction pipeline:
    user query -> Stage 1 -> inference -> Stage 2.
    """
    query = payload.query.strip()
    if not query:
        raise Stage1ExtractionError("Query must not be empty.")

    try:
        extracted = stage1_extract_features(query)
    except Exception as exc:
        raise Stage1ExtractionError("Unable to extract house features from query.") from exc

    if not extracted.provided_fields:
        raise Stage1ExtractionError(
            "Could not extract enough features from query. Please provide more house details."
        )

    model_input = _to_model_input(extracted)

    try:
        prediction_result = predict(model_input)
    except Exception as exc:
        raise InferenceExecutionError("Prediction model failed to generate a price.") from exc

    try:
        explanation = generate_explanation(
            extracted_features=extracted.model_dump(),
            model_input=model_input.model_dump(),
            predicted_price=prediction_result.prediction,
            missing_fields=extracted.missing_fields,
        )
    except Exception as exc:
        raise Stage2ExplanationError("Unable to generate explanation text.") from exc

    return PredictAPIResponse(
        extracted_features=extracted,
        predicted_price=prediction_result.prediction,
        explanation=explanation,
        missing_fields=extracted.missing_fields,
    )

