import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def crop_brain_contour(image: np.ndarray) -> np.ndarray:
    """Crop the brain region from an MRI scan using contour detection.
    
    Algorithm:
    1. Convert to grayscale
    2. Apply Gaussian blur (5,5)
    3. Binary threshold at 45
    4. Erosion + dilation (2 iterations each)
    5. Find contours, select largest
    6. Get extreme points and crop bounding box
    
    If no contour found, return original image (fallback, don't crash).
    
    Args:
        image: Numpy array of the input image.
        
    Returns:
        Cropped image as a numpy array.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Binary threshold
        _, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
        
        # Erosion + dilation
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logger.warning("No contours found in the image. Returning original image.")
            return image
            
        # Select largest contour
        c = max(contours, key=cv2.contourArea)
        
        # Get extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        
        if extTop[1] == extBot[1] or extLeft[0] == extRight[0]:
            return image
            
        # Crop
        cropped_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
        
        return cropped_image
    except Exception as e:
        logger.error(f"Error during contour cropping: {e}")
        return image

def load_and_preprocess_image(
    image_path: str | Path,
    target_size: tuple[int, int] = (224, 224)
) -> np.ndarray:
    """Load an image, crop brain contour, resize, and normalize to [0, 1].
    
    Args:
        image_path: Path to the image file.
        target_size: Desired output size as (width, height).
        
    Returns:
        numpy array of shape (height, width, 3) with values in [0, 1].
    
    Raises:
        FileNotFoundError: If image path doesn't exist.
        ValueError: If image cannot be read or is invalid.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found at {path}")
        
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to read image at {path}")
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    # Process
    cropped_image = crop_brain_contour(image)
    resized_image = cv2.resize(cropped_image, dsize=target_size, interpolation=cv2.INTER_CUBIC)
    
    # Normalize
    normalized_image = resized_image.astype('float32') / 255.0
    
    return normalized_image

def preprocess_batch(
    image_paths: list[str | Path],
    target_size: tuple[int, int] = (224, 224)
) -> np.ndarray:
    """Preprocess a batch of images.
    
    Args:
        image_paths: List of paths to images.
        target_size: Desired output size as (width, height).
        
    Returns:
        numpy array of shape (N, height, width, 3).
    """
    processed_images = []
    for path in image_paths:
        try:
            img = load_and_preprocess_image(path, target_size)
            processed_images.append(img)
        except Exception as e:
            logger.error(f"Failed to preprocess image {path}: {e}")
            raise
            
    return np.array(processed_images)
