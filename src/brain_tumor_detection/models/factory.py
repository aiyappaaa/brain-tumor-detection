from tensorflow import keras
from brain_tumor_detection.config import ModelConfig, TrainingConfig
from brain_tumor_detection.models.custom_cnn import build_custom_cnn
from brain_tumor_detection.models.mobilenet import build_mobilenet_v2
import logging

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "custom_cnn": build_custom_cnn,
    "mobilenet_v2": build_mobilenet_v2,
}

def create_model(model_config: ModelConfig) -> keras.Model:
    """Create a model by name from config."""
    model_name = model_config.architecture
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model architecture: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
        
    builder_fn = MODEL_REGISTRY[model_name]
    # Pass dropout_rate if it exists in model_config, otherwise assume default is 0.5
    dropout_rate = getattr(model_config, "dropout_rate", 0.5)
    model = builder_fn(input_shape=model_config.input_shape, dropout_rate=dropout_rate)
    logger.info(f"Created model {model_name} with input shape {model_config.input_shape} and dropout_rate {dropout_rate}")
    return model

def compile_model(
    model: keras.Model,
    training_config: TrainingConfig
) -> keras.Model:
    """Compile a model with optimizer, loss, and metrics from config.
    
    Uses binary crossentropy loss and tracks:
    - accuracy
    - AUC
    - precision
    - recall
    """
    optimizer_name = training_config.optimizer.lower()
    lr = training_config.learning_rate
    
    if optimizer_name == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    elif optimizer_name == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    elif optimizer_name == "rmsprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=lr)
    else:
        logger.warning(f"Unknown optimizer {optimizer_name}, defaulting to Adam")
        optimizer = keras.optimizers.Adam(learning_rate=lr)

    metrics = [
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.AUC(name="auc"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall")
    ]
    
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics
    )
    logger.info(f"Compiled model with {optimizer_name} (lr={lr})")
    return model

def load_trained_model(model_path: str) -> keras.Model:
    """Load a previously trained model from disk."""
    try:
        model = keras.models.load_model(model_path)
        logger.info(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise
