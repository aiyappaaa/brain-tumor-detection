"""Gradient-weighted Class Activation Mapping (Grad-CAM) for model explainability.

Provides visual explanations for CNN predictions by highlighting regions
of the input MRI that most influenced the classification decision.
Critical for medical AI — clinicians need to understand *why* the model
flagged a tumor.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from pathlib import Path
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from brain_tumor_detection.data.preprocessing import (
    load_and_preprocess_image,
    crop_brain_contour,
)

logger = logging.getLogger(__name__)


class GradCAM:
    """Gradient-weighted Class Activation Mapping for model explainability.

    Generates heatmaps showing which regions of an MRI scan most strongly
    influenced the model's tumor prediction.
    """

    def __init__(
        self,
        model: keras.Model,
        layer_name: str | None = None,
    ) -> None:
        """Initialize GradCAM.

        Args:
            model: Trained Keras model.
            layer_name: Name of the convolutional layer to visualize.
                If None, automatically selects the last Conv2D layer.
        """
        self.model = model

        if layer_name is None:
            layer_name = self._find_last_conv_layer()
        self.layer_name = layer_name

        # Build the gradient model: outputs both the conv layer activations
        # and the final predictions
        self.grad_model = keras.Model(
            inputs=self.model.input,
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.model.output,
            ],
        )

        logger.info("GradCAM initialized with layer: %s", self.layer_name)

    def _find_last_conv_layer(self) -> str:
        """Find the name of the last Conv2D layer in the model.

        Returns:
            Name of the last Conv2D layer.

        Raises:
            ValueError: If no Conv2D layer is found.
        """
        last_conv_layer = None
        for layer in self.model.layers:
            if isinstance(layer, keras.layers.Conv2D):
                last_conv_layer = layer.name
            # Also check inside nested models (e.g., MobileNetV2 base)
            if hasattr(layer, "layers"):
                for sub_layer in layer.layers:
                    if isinstance(sub_layer, keras.layers.Conv2D):
                        last_conv_layer = layer.name

        if last_conv_layer is None:
            raise ValueError(
                "No Conv2D layer found in the model. "
                "Grad-CAM requires at least one convolutional layer."
            )

        logger.debug("Auto-detected last Conv2D layer: %s", last_conv_layer)
        return last_conv_layer

    def compute_heatmap(
        self,
        image: np.ndarray,
        pred_index: int | None = None,
    ) -> np.ndarray:
        """Compute Grad-CAM heatmap for a preprocessed image.

        Args:
            image: Preprocessed image, shape (H, W, 3), values in [0, 1].
            pred_index: Class index to compute gradients for.
                If None, uses the model's predicted class.

        Returns:
            Heatmap as numpy array, shape (H, W), values in [0, 1].
        """
        # Add batch dimension
        image_tensor = tf.cast(
            tf.expand_dims(image, axis=0), dtype=tf.float32
        )

        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image_tensor)

            if pred_index is None:
                pred_index = 0  # Binary classification: single output

            loss = predictions[:, pred_index]

        # Gradients of the predicted class w.r.t. the conv layer output
        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            logger.warning(
                "Gradients are None — the layer '%s' may not be connected "
                "to the output. Returning uniform heatmap.",
                self.layer_name,
            )
            return np.ones(image.shape[:2], dtype=np.float32) * 0.5

        # Global average pooling of the gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the feature maps by the gradient importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # ReLU and normalize to [0, 1]
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val

        # Resize heatmap to match input image dimensions
        heatmap_np = heatmap.numpy()
        heatmap_resized = cv2.resize(
            heatmap_np, (image.shape[1], image.shape[0])
        )

        return heatmap_resized.astype(np.float32)

    def overlay_heatmap(
        self,
        heatmap: np.ndarray,
        original_image: np.ndarray,
        alpha: float = 0.4,
        colormap: str = "jet",
    ) -> np.ndarray:
        """Overlay a Grad-CAM heatmap on the original image.

        Args:
            heatmap: Heatmap array, shape (H, W), values in [0, 1].
            original_image: Original image, shape (H, W, 3), values in [0, 255]
                or [0, 1].
            alpha: Transparency of the heatmap overlay (0 = no heatmap, 1 = only heatmap).
            colormap: Matplotlib colormap for the heatmap visualization.

        Returns:
            Overlaid image as numpy array (H, W, 3), values in [0, 255], dtype uint8.
        """
        # Ensure original is in [0, 255] range
        if original_image.max() <= 1.0:
            original_uint8 = (original_image * 255).astype(np.uint8)
        else:
            original_uint8 = original_image.astype(np.uint8)

        # Resize heatmap to match original image if needed
        if heatmap.shape[:2] != original_uint8.shape[:2]:
            heatmap = cv2.resize(
                heatmap,
                (original_uint8.shape[1], original_uint8.shape[0]),
            )

        # Apply colormap to heatmap
        cmap = matplotlib.colormaps.get_cmap(colormap)
        heatmap_colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)

        # Blend
        overlay = cv2.addWeighted(
            original_uint8, 1 - alpha, heatmap_colored, alpha, 0
        )

        return overlay

    def generate_explanation(
        self,
        image_path: str | Path,
        save_path: str | Path | None = None,
        target_size: tuple[int, int] = (224, 224),
    ) -> dict:
        """Full Grad-CAM explanation pipeline.

        Loads image, preprocesses, computes heatmap, overlays, and
        optionally saves the visualization.

        Args:
            image_path: Path to the MRI image file.
            save_path: If provided, save the visualization to this path.
            target_size: Image preprocessing target size.

        Returns:
            Dictionary with keys:
                - 'heatmap': np.ndarray, the raw heatmap
                - 'overlay': np.ndarray, heatmap overlaid on original
                - 'prediction': dict with class, label, confidence, probability
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        logger.info("Generating Grad-CAM explanation for %s", image_path.name)

        # Load and preprocess
        preprocessed = load_and_preprocess_image(image_path, target_size)

        # Get prediction
        image_batch = np.expand_dims(preprocessed, axis=0).astype(np.float32)
        probability = float(self.model.predict(image_batch, verbose=0)[0][0])
        label = 1 if probability >= 0.5 else 0
        confidence = probability if label == 1 else 1.0 - probability

        prediction = {
            "class": "Tumor" if label == 1 else "No Tumor",
            "label": label,
            "confidence": round(confidence, 4),
            "probability": round(probability, 4),
        }

        # Compute heatmap
        heatmap = self.compute_heatmap(preprocessed)

        # Create overlay
        overlay = self.overlay_heatmap(heatmap, preprocessed)

        # Save if requested
        if save_path is not None:
            self.save_explanation(
                original=(preprocessed * 255).astype(np.uint8),
                heatmap=heatmap,
                overlay=overlay,
                prediction=prediction,
                save_path=save_path,
            )

        return {
            "heatmap": heatmap,
            "overlay": overlay,
            "prediction": prediction,
        }

    def save_explanation(
        self,
        original: np.ndarray,
        heatmap: np.ndarray,
        overlay: np.ndarray,
        prediction: dict,
        save_path: str | Path,
    ) -> None:
        """Save a side-by-side visualization: original | heatmap | overlay.

        Args:
            original: Original image (H, W, 3), uint8.
            heatmap: Raw heatmap (H, W), float in [0, 1].
            overlay: Overlaid image (H, W, 3), uint8.
            prediction: Prediction dictionary.
            save_path: Path to save the visualization.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original
        axes[0].imshow(original)
        axes[0].set_title("Original MRI", fontsize=12)
        axes[0].axis("off")

        # Heatmap
        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
        axes[1].axis("off")

        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay", fontsize=12)
        axes[2].axis("off")

        # Add prediction info as suptitle
        pred_text = (
            f"Prediction: {prediction['class']} | "
            f"Confidence: {prediction['confidence']:.1%} | "
            f"Tumor Probability: {prediction['probability']:.4f}"
        )
        fig.suptitle(pred_text, fontsize=13, fontweight="bold", y=1.02)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Grad-CAM explanation saved to %s", save_path)
