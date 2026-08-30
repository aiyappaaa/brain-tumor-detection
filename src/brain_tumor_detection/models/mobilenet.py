from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
import logging

logger = logging.getLogger(__name__)

def build_mobilenet_v2(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    freeze_base: bool = True,
    dropout_rate: float = 0.5
) -> keras.Model:
    """Build a MobileNetV2-based model for brain tumor classification.
    
    Architecture:
        MobileNetV2 backbone (ImageNet weights, no top)
        → GlobalAveragePooling2D
        → Dropout(dropout_rate)
        → Dense(128, ReLU) + BatchNorm
        → Dropout(dropout_rate * 0.6)
        → Dense(1, Sigmoid)
    
    Args:
        input_shape: Input image dimensions.
        freeze_base: If True, freeze all MobileNetV2 layers.
        dropout_rate: Dropout rate for regularization.
    
    Returns:
        Uncompiled Keras Model.
    """
    inputs = keras.Input(shape=input_shape, name="input_layer")
    
    # Preprocess inputs (MobileNetV2 expects values in [-1, 1])
    # Our data loader outputs [0, 1], so we map it to [-1, 1]
    x = layers.Rescaling(scale=2.0, offset=-1.0)(inputs)
    
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_tensor=x
    )
    
    if freeze_base:
        base_model.trainable = False
        logger.info("MobileNetV2 base model frozen.")
    else:
        base_model.trainable = True
        logger.info("MobileNetV2 base model trainable.")
        
    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)
    x = layers.Dense(128, name="dense_1")(x)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Activation('relu', name="relu_1")(x)
    x = layers.Dropout(dropout_rate * 0.6, name="dropout_2")(x)
    
    outputs = layers.Dense(1, activation='sigmoid', name="output_layer")(x)
    
    model = keras.Model(inputs=base_model.input, outputs=outputs, name="mobilenet_v2_tumor")
    logger.info(f"MobileNetV2 model built with {freeze_base=} successfully.")
    return model

def unfreeze_top_layers(model: keras.Model, num_layers: int = 30) -> keras.Model:
    """Unfreeze the top N layers of the base model for fine-tuning.
    
    Call this after initial training with frozen base to fine-tune.
    """
    # Find the base model
    base_model = None
    for layer in model.layers:
        if isinstance(layer, keras.Model) and layer.name == 'mobilenetv2_1.00_224': # standard name
            base_model = layer
            break
            
    if base_model is None:
        # If created functionally with input_tensor, the layers are flat in `model`
        for layer in model.layers[-num_layers:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
        logger.info(f"Unfrozen top {num_layers} layers of the overall model.")
    else:
        base_model.trainable = True
        for layer in base_model.layers[:-num_layers]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
        logger.info(f"Unfrozen top {num_layers} layers of the MobileNetV2 base model.")
        
    return model
