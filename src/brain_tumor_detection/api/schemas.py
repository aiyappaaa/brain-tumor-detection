"""Pydantic schemas for the Brain Tumor Detection API."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    model_loaded: bool
    version: str


class PredictionResponse(BaseModel):
    """Single image prediction response."""

    filename: str
    prediction: str  # 'Tumor' or 'No Tumor'
    label: int
    confidence: float = Field(..., ge=0.0, le=1.0)
    probability: float = Field(..., ge=0.0, le=1.0)


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    predictions: list[PredictionResponse]
    errors: list[dict[str, str]] = Field(default_factory=list)
    count: int


class ExplanationResponse(BaseModel):
    """Prediction with Grad-CAM explanation response."""

    filename: str
    prediction: str
    confidence: float
    probability: float
    heatmap_path: str


class ModelInfoResponse(BaseModel):
    """Model architecture information response."""

    architecture: str
    input_shape: list[int]
    total_parameters: int
    trainable_parameters: int


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: str | None = None
