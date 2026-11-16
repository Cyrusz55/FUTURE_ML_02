"""API request and response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _strip_text(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("text must be a string")
    stripped = value.strip()
    if not stripped:
        raise ValueError("text must not be blank")
    return stripped


class TicketRequest(BaseModel):
    """A single ticket submitted for classification."""

    text: str = Field(
        min_length=5,
        max_length=5000,
        description="Ticket text, measured in characters.",
    )

    @field_validator("text", mode="before")
    @classmethod
    def validate_text(cls, value: Any) -> str:
        return _strip_text(value)


class BatchRequest(BaseModel):
    """Tickets submitted for batch classification."""

    texts: list[str] = Field(min_length=1, max_length=100)

    @field_validator("texts", mode="before")
    @classmethod
    def validate_texts(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            raise ValueError("texts must be a list")
        return [_strip_text(item) for item in value]


class TopCategory(BaseModel):
    """A calibrated category candidate."""

    category: str
    probability: float


class PredictionResponse(BaseModel):
    """Prediction and transparent priority result."""

    model_config = ConfigDict(extra="forbid")

    category: str
    category_confidence: float
    top_categories: list[TopCategory]
    priority: str
    priority_points: int
    priority_rules_matched: list[str]
    priority_method: str


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    results: list[PredictionResponse]


class HealthResponse(BaseModel):
    """Application health response."""

    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Metadata for the deployed model and rule layer."""

    model_name: str
    hf_repo_id: str
    hf_revision: str
    class_names: list[str]
    temperature: float
    rules_version: int
    high_at: int
    medium_at: int
    library_versions: dict[str, str]

