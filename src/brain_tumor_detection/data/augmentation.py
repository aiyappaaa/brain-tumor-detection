import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from brain_tumor_detection.config import AugmentationConfig
import logging

logger = logging.getLogger(__name__)

def create_train_augmentor(config: AugmentationConfig) -> ImageDataGenerator:
    """Create an ImageDataGenerator for training data with augmentation.
    
    Applies: rotation, shifts, shear, brightness, flips.
    Images are pre-normalized, so NO rescaling is applied here.
    
    Args:
        config: Augmentation configuration settings.
        
    Returns:
        ImageDataGenerator configured for training.
    """
    if config.enabled:
        logger.info("Creating training augmentor with active augmentation.")
        return ImageDataGenerator(
            rotation_range=config.rotation_range,
            width_shift_range=config.width_shift_range,
            height_shift_range=config.height_shift_range,
            shear_range=config.shear_range,
            brightness_range=config.brightness_range,
            horizontal_flip=config.horizontal_flip,
            vertical_flip=config.vertical_flip,
            fill_mode=config.fill_mode
            # No rescale=1./255 because preprocessing handles it
        )
    else:
        logger.info("Augmentation is disabled. Creating basic training generator.")
        return ImageDataGenerator()

def create_eval_augmentor() -> ImageDataGenerator:
    """Create an ImageDataGenerator for validation/test data.
    
    NO augmentation — and no rescaling since it is pre-normalized.
    
    Returns:
        ImageDataGenerator configured for evaluation.
    """
    return ImageDataGenerator()

def create_data_generators(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    augmentation_config: AugmentationConfig,
    batch_size: int = 32
) -> tuple:
    """Create training and validation data generators.
    
    Training generator applies augmentation.
    Validation generator does NOT apply augmentation.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        augmentation_config: Configuration for data augmentation.
        batch_size: Batch size for generators.
        
    Returns:
        (train_generator, val_generator, steps_per_epoch, validation_steps)
    """
    train_datagen = create_train_augmentor(augmentation_config)
    val_datagen = create_eval_augmentor()
    
    train_generator = train_datagen.flow(
        X_train, y_train,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_generator = val_datagen.flow(
        X_val, y_val,
        batch_size=batch_size,
        shuffle=False
    )
    
    steps_per_epoch = len(X_train) // batch_size if len(X_train) % batch_size == 0 else (len(X_train) // batch_size) + 1
    validation_steps = len(X_val) // batch_size if len(X_val) % batch_size == 0 else (len(X_val) // batch_size) + 1
    
    return train_generator, val_generator, steps_per_epoch, validation_steps
