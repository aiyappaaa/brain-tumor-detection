from tensorflow import keras
from pathlib import Path
from brain_tumor_detection.config import AppConfig
from brain_tumor_detection.models.factory import create_model, compile_model
from brain_tumor_detection.data.augmentation import create_data_generators
from brain_tumor_detection.training.callbacks import create_callbacks
from brain_tumor_detection.training.metrics import evaluate_model, generate_training_report
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class Trainer:
    """Encapsulates the full training workflow."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.model = None
        self.history = None
        self.run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    def setup(self) -> None:
        """Create and compile the model."""
        logger.info("Setting up Trainer...")
        uncompiled_model = create_model(self.config.model)
        self.model = compile_model(uncompiled_model, self.config.training)
        logger.info("Trainer setup complete.")
    
    def train(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray
    ) -> keras.callbacks.History:
        """Train the model with proper callbacks.
        
        Uses:
        - On-the-fly augmentation for training data
        - ModelCheckpoint (save best by val_accuracy)
        - EarlyStopping (patience from config)
        - ReduceLROnPlateau
        - TensorBoard logging
        - CSVLogger
        """
        if self.model is None:
            self.setup()
            
        logger.info("Starting training...")
        
        callbacks = create_callbacks(
            self.config.training,
            self.config.output,
            run_name=self.run_name
        )
        
        # Create augmented data generators
        train_iterator, val_iterator, steps_per_epoch, validation_steps = create_data_generators(
            X_train, y_train,
            X_val, y_val,
            self.config.augmentation,
            self.config.training.batch_size
        )
        
        n_pos = np.sum(y_train)
        n_neg = len(y_train) - n_pos
        total = len(y_train)
        class_weight = {0: total/(2*n_neg), 1: total/(2*n_pos)}
        
        self.history = self.model.fit(
            train_iterator,
            epochs=self.config.training.epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=2
        )
        
        logger.info("Training completed.")
        return self.history
    
    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> dict:
        """Evaluate on test set and return metrics dict."""
        if self.model is None:
            raise ValueError("Model is not initialized or trained yet.")
        return evaluate_model(self.model, X_test, y_test)
    
    def save_model(self, path: str | Path | None = None) -> Path:
        """Save model in Keras native format (.keras)."""
        if self.model is None:
            raise ValueError("Model is not initialized or trained yet.")
            
        best_checkpoint_path = Path(self.config.output.checkpoint_dir) / f"model_{self.run_name}_best.keras"
        if best_checkpoint_path.exists():
            logger.info(f"Loading best weights from {best_checkpoint_path}")
            self.model.load_weights(best_checkpoint_path)
            
        if path is None:
            export_dir = Path(self.config.output.export_dir)
            export_dir.mkdir(parents=True, exist_ok=True)
            path = export_dir / f"model_{self.run_name}.keras"
            
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        logger.info(f"Model saved to {path}")
        return path
    
    def generate_report(
        self,
        X_test: np.ndarray, y_test: np.ndarray
    ) -> None:
        """Generate full training report with plots and metrics."""
        if self.model is None:
            raise ValueError("Model is not initialized or trained yet.")
            
        report_dir = Path(self.config.output.metrics_dir) / f"report_{self.run_name}"
        logger.info(f"Generating training report in {report_dir}")
        generate_training_report(
            model=self.model,
            history=self.history,
            X_test=X_test,
            y_test=y_test,
            output_dir=report_dir
        )
        logger.info("Report generation complete.")
