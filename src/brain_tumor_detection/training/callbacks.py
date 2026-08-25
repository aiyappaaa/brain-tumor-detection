from tensorflow import keras
from pathlib import Path
from brain_tumor_detection.config import TrainingConfig, OutputConfig
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def create_callbacks(
    training_config: TrainingConfig,
    output_config: OutputConfig,
    run_name: str | None = None
) -> list[keras.callbacks.Callback]:
    """Create the full callback stack for training.
    
    Returns callbacks:
    1. ModelCheckpoint - save best model by val_loss
    2. EarlyStopping - stop if val_loss doesn't improve
    3. ReduceLROnPlateau - reduce LR on val_loss plateau  
    4. TensorBoard - log to output_config.log_dir
    5. CSVLogger - log metrics to CSV
    """
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        
    checkpoint_dir = Path(output_config.checkpoint_dir)
    log_dir = Path(output_config.log_dir) / run_name
    metrics_dir = Path(output_config.metrics_dir)
    
    for d in [checkpoint_dir, log_dir, metrics_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    callbacks = []
    
    # 1. ModelCheckpoint
    checkpoint_path = checkpoint_dir / f"model_{run_name}_best.keras"
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    )
    callbacks.append(checkpoint_cb)
    
    # 2. EarlyStopping
    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=training_config.early_stopping_patience,
        restore_best_weights=True,
        mode="min",
        verbose=1
    )
    callbacks.append(early_stopping_cb)
    
    # 3. ReduceLROnPlateau
    reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=training_config.reduce_lr_factor,
        patience=training_config.reduce_lr_patience,
        min_lr=1e-6,
        mode="min",
        verbose=1
    )
    callbacks.append(reduce_lr_cb)
    
    # 4. TensorBoard
    tensorboard_cb = keras.callbacks.TensorBoard(
        log_dir=str(log_dir),
        histogram_freq=1,
        update_freq="epoch"
    )
    callbacks.append(tensorboard_cb)
    
    # 5. CSVLogger
    csv_path = metrics_dir / f"training_log_{run_name}.csv"
    csv_logger_cb = keras.callbacks.CSVLogger(
        filename=str(csv_path),
        append=True
    )
    callbacks.append(csv_logger_cb)
    
    logger.info(f"Created {len(callbacks)} callbacks for run {run_name}")
    return callbacks
