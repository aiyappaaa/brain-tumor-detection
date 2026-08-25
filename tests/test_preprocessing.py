import pytest
import numpy as np
import cv2
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from brain_tumor_detection.data.preprocessing import (
    crop_brain_contour, load_and_preprocess_image, preprocess_batch
)

@pytest.fixture
def sample_image(tmp_path):
    img_path = tmp_path / "test_img.jpg"
    img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    return img_path

class TestCropBrainContour:
    def test_returns_ndarray(self):
        """crop_brain_contour should return a numpy array."""
        image = np.zeros((240, 240, 3), dtype=np.uint8)
        cv2.circle(image, (120, 120), 80, (200, 200, 200), -1)
        result = crop_brain_contour(image)
        assert isinstance(result, np.ndarray)
    
    def test_crops_to_smaller_size(self):
        """Result should be cropped (potentially smaller than input)."""
        image = np.zeros((240, 240, 3), dtype=np.uint8)
        cv2.circle(image, (120, 120), 60, (200, 200, 200), -1)
        result = crop_brain_contour(image)
        assert result.shape[0] <= image.shape[0]
        assert result.shape[1] <= image.shape[1]
    
    def test_handles_no_contour(self):
        """Should return original image if no contour found (all black)."""
        image = np.zeros((240, 240, 3), dtype=np.uint8)
        result = crop_brain_contour(image)
        assert result.shape == image.shape
    
    def test_handles_3_channel_input(self):
        """Should work with 3-channel color images."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = crop_brain_contour(image)
        assert result.ndim == 3
        assert result.shape[2] == 3


class TestLoadAndPreprocessImage:
    def test_loads_real_image(self, sample_image):
        """Test loading an MRI image."""
        result = load_and_preprocess_image(sample_image, target_size=(224, 224))
        assert result.shape == (224, 224, 3)
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    def test_output_normalized(self, sample_image):
        """Output should be normalized to [0, 1]."""
        result = load_and_preprocess_image(sample_image)
        assert result.dtype == np.float64 or result.dtype == np.float32
    
    def test_file_not_found(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_and_preprocess_image('nonexistent_file.jpg')
    
    def test_custom_target_size(self, sample_image):
        """Should resize to specified target size."""
        result = load_and_preprocess_image(sample_image, target_size=(128, 128))
        assert result.shape == (128, 128, 3)


class TestPreprocessBatch:
    def test_batch_shape(self, tmp_path):
        """Batch output should have shape (N, H, W, 3)."""
        images = []
        for i in range(3):
            img_path = tmp_path / f"test_img_{i}.jpg"
            img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            cv2.imwrite(str(img_path), img)
            images.append(img_path)
            
        result = preprocess_batch(images, target_size=(224, 224))
        assert result.shape == (3, 224, 224, 3)
    
    def test_empty_batch(self):
        """Empty batch should return empty array."""
        result = preprocess_batch([], target_size=(224, 224))
        assert result.shape[0] == 0
