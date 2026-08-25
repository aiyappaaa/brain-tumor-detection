"""Production-ready predictor for brain tumor classification."""

import numpy as np
from pathlib import Path
import logging
from typing import Any

from brain_tumor_detection.data.preprocessing import load_and_preprocess_image
from brain_tumor_detection.models.factory import load_trained_model

logger = logging.getLogger(__name__)


class Predictor:
    """Production-ready predictor for brain tumor classification.

    Loads a trained model once and provides methods for single-image
    and batch predictions with full preprocessing applied automatically.
    """

    CLASS_NAMES: dict[int, str] = {0: "No Tumor", 1: "Tumor"}

    def __init__(
        self,
        model_path: str | Path,
        image_size: tuple[int, int] = (224, 224),
        threshold: float = 0.5,
    ) -> None:
        """Initialize the predictor by loading a trained model.

        Args:
            model_path: Path to the saved Keras model file.
            image_size: Target size for image preprocessing (height, width).
            threshold: Classification threshold for tumor detection.

        Raises:
            FileNotFoundError: If model_path does not exist.
            ValueError: If threshold is not in (0, 1).
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")

        self.model = load_trained_model(str(model_path))
        self.image_size = image_size
        self.threshold = threshold
        logger.info(
            "Predictor initialized with model from %s (threshold=%.2f)",
            model_path,
            threshold,
        )

    def predict(self, image_path: str | Path) -> dict[str, Any]:
        """Predict on a single MRI image.

        Args:
            image_path: Path to the MRI image file.

        Returns:
            Dictionary with keys:
                - 'class': 'Tumor' or 'No Tumor'
                - 'label': 1 or 0
                - 'confidence': float in [0, 1], confidence in the predicted class
                - 'probability': float in [0, 1], raw probability of tumor

        Raises:
            FileNotFoundError: If image_path does not exist.
            ValueError: If image cannot be processed.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        logger.debug("Predicting on %s", image_path)

        # Preprocess the image
        image = load_and_preprocess_image(image_path, target_size=self.image_size)

        # Add batch dimension: (H, W, 3) -> (1, H, W, 3)
        image_batch = np.expand_dims(image, axis=0).astype(np.float32)

        # Run inference
        probability = float(self.model.predict(image_batch, verbose=0)[0][0])

        # Determine class
        label = 1 if probability >= self.threshold else 0
        class_name = self.CLASS_NAMES[label]
        confidence = probability if label == 1 else 1.0 - probability

        result = {
            "class": class_name,
            "label": label,
            "confidence": round(confidence, 4),
            "probability": round(probability, 4),
        }

        logger.info(
            "Prediction for %s: %s (confidence=%.2f%%)",
            image_path.name,
            class_name,
            confidence * 100,
        )
        return result

    def predict_batch(self, image_paths: list[str | Path]) -> list[dict[str, Any]]:
        """Predict on multiple images efficiently.

        Args:
            image_paths: List of paths to MRI image files.

        Returns:
            List of prediction dictionaries (same format as predict()).
        """
        if not image_paths:
            logger.warning("Empty batch provided")
            return []

        logger.info("Running batch prediction on %d images", len(image_paths))

        results = [None] * len(image_paths)
        images = []
        valid_indices = []

        for i, path in enumerate(image_paths):
            path_obj = Path(path)
            try:
                image = load_and_preprocess_image(
                    path_obj, target_size=self.image_size
                )
                images.append(image)
                valid_indices.append(i)
            except Exception as e:
                logger.warning("Skipping %s: %s", path, e)
                results[i] = {"filename": path_obj.name, "error": str(e)}

        if not images:
            logger.warning("No valid images in batch")
            return results

        # Stack into batch array: (N, H, W, 3)
        batch = np.array(images, dtype=np.float32)

        # Run inference
        probabilities = self.model.predict(batch, verbose=0).flatten()

        # Build results
        for idx, probability in zip(valid_indices, probabilities):
            prob = float(probability)
            label = 1 if prob >= self.threshold else 0
            class_name = self.CLASS_NAMES[label]
            confidence = prob if label == 1 else 1.0 - prob
            path_obj = Path(image_paths[idx])

            results[idx] = {
                "class": class_name,
                "label": label,
                "confidence": round(confidence, 4),
                "probability": round(prob, 4),
                "filename": path_obj.name,
            }

        logger.info(
            "Batch prediction complete: %d/%d images processed",
            len(valid_indices),
            len(image_paths),
        )
        return results
