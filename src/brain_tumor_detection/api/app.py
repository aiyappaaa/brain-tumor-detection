"""FastAPI application factory for the Brain Tumor Detection API."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from brain_tumor_detection.api.routes import router
from brain_tumor_detection.inference.gradcam import GradCAM
from brain_tumor_detection.inference.predictor import Predictor

logger = logging.getLogger(__name__)


def create_app(
    model_path: str | Path,
    image_size: tuple[int, int] = (224, 224),
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        model_path: Path to the trained Keras model.
        image_size: Target size for image preprocessing.

    Returns:
        Configured FastAPI application instance.
    """
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Initialize model and GradCAM on application startup."""
        logger.info("Starting API — loading model from %s", model_path)
        try:
            predictor = Predictor(model_path=model_path, image_size=image_size)
            app.state.predictor = predictor

            gradcam = GradCAM(model=predictor.model)
            app.state.gradcam = gradcam

            logger.info("Model and GradCAM initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize model: %s", e)
            raise
        yield

    app = FastAPI(
        title="Brain Tumor Detection API",
        description=(
            "MRI-based brain tumor classification with Grad-CAM explainability. "
            "Upload brain MRI scans to detect tumors and visualize model attention."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS middleware for cross-origin requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static files directory for Grad-CAM output images
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)
    (static_dir / "explanations").mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory="static"), name="static")

    # Register routes
    app.include_router(router)

    return app
