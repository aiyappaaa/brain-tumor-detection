import pytest
import sys
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from brain_tumor_detection.config import AugmentationConfig
from brain_tumor_detection.data.augmentation import create_train_augmentor, create_eval_augmentor

def test_create_train_augmentor_enabled():
    config = AugmentationConfig(
        enabled=True,
        rotation_range=20,
        horizontal_flip=True,
        vertical_flip=False
    )
    augmentor = create_train_augmentor(config)
    assert isinstance(augmentor, ImageDataGenerator)
    assert augmentor.rotation_range == 20
    assert augmentor.horizontal_flip == True
    assert augmentor.vertical_flip == False

def test_create_train_augmentor_disabled():
    config = AugmentationConfig(enabled=False)
    augmentor = create_train_augmentor(config)
    assert isinstance(augmentor, ImageDataGenerator)
    assert augmentor.rotation_range == 0
    assert augmentor.horizontal_flip == False

def test_create_eval_augmentor():
    augmentor = create_eval_augmentor()
    assert isinstance(augmentor, ImageDataGenerator)
    assert augmentor.rotation_range == 0
    assert augmentor.horizontal_flip == False
