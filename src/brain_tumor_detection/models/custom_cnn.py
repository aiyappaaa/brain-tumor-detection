from tensorflow import keras
from tensorflow.keras import layers
import logging

logger = logging.getLogger(__name__)

def build_custom_cnn(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.5
) -> keras.Model:
    """Build an improved custom CNN for brain tumor classification.
    
    Architecture:
        Input (224, 224, 3)
        → Conv2D(32, 3×3, padding='same') + BatchNorm + ReLU + MaxPool(2×2)
        → Conv2D(64, 3×3, padding='same') + BatchNorm + ReLU + MaxPool(2×2) 
        → Conv2D(128, 3×3, padding='same') + BatchNorm + ReLU + MaxPool(2×2)
        → Conv2D(256, 3×3, padding='same') + BatchNorm + ReLU + MaxPool(2×2)
        → GlobalAveragePooling2D
        → Dropout(dropout_rate)
        → Dense(256, ReLU) + BatchNorm
        → Dropout(dropout_rate * 0.6)
        → Dense(1, Sigmoid)
    
    Key improvements over original:
    - Progressive feature extraction (32→64→128→256)
    - Standard 2×2 max pooling (not consecutive 4×4)
    - GlobalAveragePooling instead of Flatten (fewer params)
    - Dropout regularization
    - ~600K parameters (trainable on CPU)
    
    Returns:
        Uncompiled Keras Model.
    """
    inputs = keras.Input(shape=input_shape, name="input_layer")
    
    # Block 1
    x = layers.Conv2D(32, (3, 3), padding='same', name="conv_1")(inputs)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Activation('relu', name="relu_1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool_1")(x)
    
    # Block 2
    x = layers.Conv2D(64, (3, 3), padding='same', name="conv_2")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Activation('relu', name="relu_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool_2")(x)
    
    # Block 3
    x = layers.Conv2D(128, (3, 3), padding='same', name="conv_3")(x)
    x = layers.BatchNormalization(name="bn_3")(x)
    x = layers.Activation('relu', name="relu_3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool_3")(x)
    
    # Block 4
    x = layers.Conv2D(256, (3, 3), padding='same', name="conv_4")(x)
    x = layers.BatchNormalization(name="bn_4")(x)
    x = layers.Activation('relu', name="relu_4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool_4")(x)
    
    # Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)
    
    x = layers.Dense(256, name="dense_1")(x)
    x = layers.BatchNormalization(name="bn_5")(x)
    x = layers.Activation('relu', name="relu_5")(x)
    x = layers.Dropout(dropout_rate * 0.6, name="dropout_2")(x)
    
    outputs = layers.Dense(1, activation='sigmoid', name="output_layer")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="brain_tumor_cnn")
    logger.info("Custom CNN built successfully.")
    return model
