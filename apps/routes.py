"""Versioned API routes."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException

from apps.model_loader import HF_REPO_ID, HF_REVISION, get_model, model_is_loaded
from apps.schemas import (
    BatchPredictionResponse,
    BatchRequest,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    TicketRequest,
)
from machine_learning.machine_learning import (
    HIGH_AT,
    MEDIUM_AT,
    RULES_VERSION,
    predict_tickets,
)

LOGGER = logging.getLogger(__name__)
router = APIRouter()


def _to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """Normalize single and batch payloads through one tabular boundary."""
    if "text" in payload:
        return pd.DataFrame({"text": [payload["text"]]})
    return pd.DataFrame({"text": payload["texts"]})


def _predict(payload: dict[str, Any]) -> list[dict[str, Any]]:
    frame = _to_frame(payload)
    try:
        return predict_tickets(frame["text"].tolist())
    except (FileNotFoundError, OSError, ValueError) as exc:
        LOGGER.exception("Prediction failed.")
        raise HTTPException(
            status_code=503,
            detail="The classification model is unavailable.",
        ) from exc


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = model_is_loaded()
    return HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded)


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: TicketRequest) -> PredictionResponse:
    return PredictionResponse(**_predict(payload.model_dump())[0])


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchRequest) -> BatchPredictionResponse:
    results = [PredictionResponse(**item) for item in _predict(payload.model_dump())]
    return BatchPredictionResponse(results=results)


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    try:
        bundle = get_model()
    except (FileNotFoundError, OSError, ValueError) as exc:
        LOGGER.exception("Model metadata unavailable.")
        raise HTTPException(
            status_code=503,
            detail="The classification model is unavailable.",
        ) from exc
    return ModelInfoResponse(
        model_name=str(bundle["model_name"]),
        hf_repo_id=HF_REPO_ID,
        hf_revision=HF_REVISION,
        class_names=list(bundle["classes"]),
        temperature=float(bundle["temperature"]),
        rules_version=RULES_VERSION,
        high_at=HIGH_AT,
        medium_at=MEDIUM_AT,
        library_versions={str(key): str(value) for key, value in bundle["versions"].items()},
    )

