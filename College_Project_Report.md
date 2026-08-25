# Brain Tumor Detection from MRI Scans: A Comprehensive Deep Learning Pipeline

## 1. Abstract
This project implements an end-to-end computer vision pipeline for the binary classification of Brain MRI scans to detect the presence of tumors. The project addresses the critical challenge of automating brain tumor detection, which can significantly aid radiologists in diagnosis. It features a complete Machine Learning Operations (MLOps) lifecycle, including robust data preprocessing, dynamic data augmentation, deep learning model architectures (Custom CNN and MobileNetV2), comprehensive evaluation metrics, and model explainability via Grad-CAM. Finally, the project is wrapped in a production-ready FastAPI REST API and a robust Command-Line Interface (CLI).

## 2. Problem Statement
Brain tumors are severe conditions that require early and accurate detection for effective treatment planning. Magnetic Resonance Imaging (MRI) is the standard imaging technique used for this purpose. However, manually examining MRI scans is time-consuming and subject to human error. This project aims to develop a highly accurate, automated deep learning model capable of classifying an MRI scan as either "Tumor" or "No Tumor" while providing visual explanations for its predictions to ensure clinical trust.

## 3. Dataset Description
The dataset consists of Brain MRI images categorized into two classes:
*   **Yes (Tumorous):** 155 original images.
*   **No (Non-Tumorous):** 98 original images.
*   **Total:** 253 original images.

The dataset is inherently imbalanced (roughly 61% positive and 39% negative). To address this, stratified splitting and data augmentation techniques are employed.

## 4. Addressing Critical Data Leakage
In early iterations of ML projects, a common but critical mistake is applying data augmentation to the entire dataset *before* splitting it into training, validation, and test sets. This causes "data leakage," where augmented variations of the same patient's brain scan end up in both the training set and the test set. 

**The Fix:** This pipeline strictly enforces a *split-first* strategy. The original 253 images are split into Training (70%), Validation (15%), and Testing (15%) subsets using stratified splitting to maintain class balance. Data augmentation is then applied *only* to the training subset dynamically during the training loop. This ensures the model's evaluation metrics reflect true generalization to unseen data.

## 5. Preprocessing Pipeline
To ensure the neural network learns relevant features rather than background noise, the following preprocessing steps are applied to every MRI scan:

1.  **Grayscale Conversion & Blurring:** Images are converted to grayscale and a Gaussian blur is applied to reduce noise.
2.  **Thresholding & Morphological Operations:** Binary thresholding separates the brain from the dark background. Erosion and dilation (morphological operations) clean up the edges.
3.  **Contour Detection & Extreme Cropping:** The algorithm finds the largest contour (the brain), identifies its extreme points (top, bottom, left, right), and crops the image to form a tight bounding box around the brain. This step removes useless black background space.
4.  **Resizing:** The cropped image is resized to a standard dimension of `224x224` pixels.
5.  **Normalization:** Pixel intensity values are scaled from `[0, 255]` to `[0, 1]` to help the neural network converge faster during training.

## 6. Data Augmentation
Because the dataset is exceptionally small (253 images), neural networks are prone to overfitting. We use on-the-fly data augmentation via Keras `ImageDataGenerator` on the training set. Transformations include:
*   Random rotations (up to 15 degrees)
*   Width and height shifts (10%)
*   Shear transformations (10%)
*   Brightness variations
*   Horizontal and vertical flips

*Note: Validation and Test sets are never augmented.*

## 7. Model Architectures
The project supports two distinct neural network architectures, configurable via YAML files:

### 7.1. Custom Convolutional Neural Network (CNN)
Designed to be lightweight and train efficiently on a CPU.
*   **Architecture:** Four progressive convolutional blocks. Filters increase progressively (32 → 64 → 128 → 256).
*   **Components:** Each block utilizes a `Conv2D` layer with a 3x3 kernel, `BatchNormalization` for training stability, `ReLU` activation, and `MaxPooling2D` (2x2) to reduce spatial dimensions.
*   **Head:** A `GlobalAveragePooling2D` layer flattens the output, drastically reducing parameter count compared to traditional `Flatten` layers, thus reducing overfitting. This is followed by `Dropout` layers and a final `Dense` layer with a `sigmoid` activation function for binary classification.

### 7.2. Transfer Learning (MobileNetV2)
For higher accuracy, the pipeline supports transfer learning using the MobileNetV2 architecture pretrained on ImageNet. 
*   The base model is initially frozen.
*   A custom classification head (Global Average Pooling → Dropout → Dense) is appended.
*   This approach leverages deep feature representations learned from millions of images, making it highly effective for small datasets.

## 8. Training Methodology
The training process is orchestrated by a central `Trainer` class and includes several MLOps best practices:
*   **Optimizer:** Adam optimizer with an initial learning rate of `0.001`.
*   **Loss Function:** Binary Cross-Entropy.
*   **Callbacks:**
    *   `ModelCheckpoint`: Automatically saves the model weights that achieve the highest validation accuracy.
    *   `EarlyStopping`: Halts training if the validation loss does not improve for 10 consecutive epochs, preventing overfitting.
    *   `ReduceLROnPlateau`: Reduces the learning rate dynamically if the validation loss plateaus.
    *   `TensorBoard & CSVLogger`: Tracks loss and accuracy metrics over time for visualization.

## 9. Evaluation Metrics
Testing is performed on the completely unseen 15% holdout test set. The project evaluates the model using:
*   **Accuracy:** The overall percentage of correct predictions.
*   **Precision & Recall:** Crucial for medical datasets. High recall ensures that tumors are not missed (minimizing false negatives).
*   **F1-Score:** The harmonic mean of precision and recall.
*   **ROC-AUC:** Measures the model's ability to distinguish between the two classes at various thresholds.
*   **Confusion Matrix:** A visual heatmap plotting True Positives, True Negatives, False Positives, and False Negatives.

## 10. Explainable AI (XAI) using Grad-CAM
A critical requirement for AI in healthcare is transparency. Doctors must know *why* a model predicted a tumor.
This project implements **Gradient-weighted Class Activation Mapping (Grad-CAM)**. 
*   **How it works:** It traces the gradients of the target class back to the final convolutional layer to determine which parts of the image had the highest impact on the prediction.
*   **Output:** It generates a color-coded heatmap overlaid on the original MRI scan. Red areas indicate high model attention (where it thinks the tumor is), while blue areas indicate low attention.

## 11. Production-Ready REST API
To simulate a real-world software deployment, the model is served via a REST API built with **FastAPI**.
*   **Endpoints:**
    *   `/health`: System health check.
    *   `/model/info`: Returns model parameter counts and architecture info.
    *   `/predict`: Accepts a single image upload and returns the prediction, confidence score, and raw probability.
    *   `/predict/batch`: Accepts multiple images for batch processing.
    *   `/predict/explain`: Runs inference, generates a Grad-CAM heatmap, saves the image, and returns a URL to view the visual explanation.

## 12. Command-Line Interface (CLI)
The project includes a robust CLI (`python -m brain_tumor_detection`) allowing researchers to easily trigger different parts of the pipeline:
*   `train`: Initiates the training loop.
*   `evaluate`: Evaluates a saved model against the test set.
*   `predict`: Predicts a single image (with an optional `--explain` flag for Grad-CAM).
*   `serve`: Boots up the FastAPI web server.
*   `export`: Converts the `.keras` model into `SavedModel` or `TFLite` formats for mobile or edge deployment.

## 13. Conclusion
This project successfully transitions a standard Jupyter Notebook experiment into a production-grade software engineering pipeline. By fixing data leakage, introducing dynamic augmentation, utilizing global average pooling, providing Grad-CAM explainability, and wrapping the logic in a scalable API, the resulting system is robust, reproducible, and ready for deployment in simulated clinical environments.
