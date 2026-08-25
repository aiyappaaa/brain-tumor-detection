"""FastAPI route definitions for the Brain Tumor Detection API."""

import logging
import shutil
import tempfile
import uuid
from pathlib import Path

import tensorflow as tf
from fastapi import APIRouter, File, HTTPException, UploadFile, Request

from brain_tumor_detection.api.schemas import (
    BatchPredictionResponse,
    ErrorResponse,
    ExplanationResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from brain_tumor_detection.inference.gradcam import GradCAM
from brain_tumor_detection.inference.predictor import Predictor

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _validate_image_upload(file: UploadFile) -> None:
    """Validate that the uploaded file is an allowed image type."""
    file_ext = Path(file.filename or "").suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file_ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )


def _save_upload_to_temp(file: UploadFile) -> Path:
    """Save an uploaded file to a temporary location."""
    file_ext = Path(file.filename or "").suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        return Path(tmp.name)


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        model_loaded=hasattr(request.app.state, "predictor") and request.app.state.predictor is not None,
        version="1.0.0",
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def predict(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
    """Upload an MRI image and get tumor prediction."""
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    _validate_image_upload(file)
    temp_path = _save_upload_to_temp(file)

    try:
        result = predictor.predict(temp_path)
        return PredictionResponse(
            filename=file.filename or "unknown",
            prediction=result["class"],
            label=result["label"],
            confidence=result["confidence"],
            probability=result["probability"],
        )
    except Exception as e:
        logger.error("Prediction failed for %s: %s", file.filename, e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        temp_path.unlink(missing_ok=True)


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: Request,
    files: list[UploadFile] = File(...),
) -> BatchPredictionResponse:
    """Upload multiple MRI images for batch prediction."""
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    temp_paths: list[tuple[Path, str]] = []

    try:
        # Save all uploads
        for file in files:
            _validate_image_upload(file)
            temp_path = _save_upload_to_temp(file)
            temp_paths.append((temp_path, file.filename or "unknown"))

        # Batch predict
        paths_only = [p for p, _ in temp_paths]
        results = predictor.predict_batch(paths_only)

        predictions = []
        errors = []
        for result in results:
            if "error" in result:
                errors.append(result)
            else:
                predictions.append(
                    PredictionResponse(
                        filename=result["filename"],
                        prediction=result["class"],
                        label=result["label"],
                        confidence=result["confidence"],
                        probability=result["probability"],
                    )
                )

        return BatchPredictionResponse(
            predictions=predictions,
            errors=errors,
            count=len(predictions),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Batch prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p, _ in temp_paths:
            p.unlink(missing_ok=True)


@router.post("/predict/explain", response_model=ExplanationResponse)
async def predict_with_explanation(
    request: Request,
    file: UploadFile = File(...),
) -> ExplanationResponse:
    """Upload an MRI image and get prediction with Grad-CAM explanation."""
    gradcam = getattr(request.app.state, "gradcam", None)
    if gradcam is None:
        raise HTTPException(status_code=503, detail="GradCAM not initialized")

    _validate_image_upload(file)
    temp_path = _save_upload_to_temp(file)

    try:
        # Generate explanation with unique output filename
        out_id = uuid.uuid4().hex[:12]
        out_dir = Path("static/explanations")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_filename = f"{out_id}.png"
        out_path = out_dir / out_filename

        result = gradcam.generate_explanation(temp_path, save_path=out_path)
        prediction = result["prediction"]

        return ExplanationResponse(
            filename=file.filename or "unknown",
            prediction=prediction["class"],
            confidence=prediction["confidence"],
            probability=prediction["probability"],
            heatmap_path=f"/static/explanations/{out_filename}",
        )
    except Exception as e:
        logger.error("Explanation generation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        temp_path.unlink(missing_ok=True)


@router.get("/model/info", response_model=ModelInfoResponse)
async def model_info(request: Request) -> ModelInfoResponse:
    """Get loaded model architecture information."""
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    model = predictor.model
    trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    non_trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.non_trainable_weights
    )

    return ModelInfoResponse(
        architecture=model.name,
        input_shape=list(model.input_shape)[1:],
        total_parameters=int(trainable_params + non_trainable_params),
        trainable_parameters=int(trainable_params),
    )
