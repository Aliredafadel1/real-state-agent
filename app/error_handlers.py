"""
Custom error types and FastAPI exception handlers.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class Stage1ExtractionError(Exception):
    """Raised when Stage 1 feature extraction fails."""


class InferenceExecutionError(Exception):
    """Raised when ML inference fails."""


class Stage2ExplanationError(Exception):
    """Raised when Stage 2 explanation generation fails."""


def register_error_handlers(app: FastAPI) -> None:
    """Register API-wide exception handlers."""

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "detail": "Invalid request payload.",
                "issues": exc.errors(),
            },
        )

    @app.exception_handler(Stage1ExtractionError)
    async def stage1_exception_handler(
        request: Request, exc: Stage1ExtractionError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={
                "error": "stage1_extraction_failed",
                "detail": str(exc),
            },
        )

    @app.exception_handler(InferenceExecutionError)
    async def inference_exception_handler(
        request: Request, exc: InferenceExecutionError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "error": "inference_failed",
                "detail": str(exc),
            },
        )

    @app.exception_handler(Stage2ExplanationError)
    async def stage2_exception_handler(
        request: Request, exc: Stage2ExplanationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "error": "stage2_explanation_failed",
                "detail": str(exc),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "detail": "Unexpected server error.",
            },
        )

