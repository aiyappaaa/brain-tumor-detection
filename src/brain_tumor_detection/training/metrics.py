import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, accuracy_score
)
from pathlib import Path
from tensorflow import keras
import json
import logging

logger = logging.getLogger(__name__)

def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray, y_test: np.ndarray,
    threshold: float = 0.5
) -> dict:
    """Compute comprehensive metrics on test set.
    
    Returns dict with: accuracy, f1, precision, recall, auc, 
    confusion_matrix, classification_report.
    """
    logger.info("Evaluating model...")
    y_prob = model.predict(X_test, verbose=0)
    y_pred = (y_prob > threshold).astype(int).flatten()
    y_true = y_test.flatten()
    
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True)
    }
    logger.info(f"Evaluation completed. Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")
    return metrics

def plot_training_history(
    history: keras.callbacks.History | dict,
    save_dir: str | Path
) -> None:
    """Plot and save accuracy and loss curves."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if isinstance(history, keras.callbacks.History):
        hist = history.history
    else:
        hist = history
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(hist['loss'], label='train_loss')
    if 'val_loss' in hist:
        ax1.plot(hist['val_loss'], label='val_loss')
    ax1.set_title('Model Loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    
    # Accuracy plot
    acc_key = 'accuracy' if 'accuracy' in hist else 'acc'
    val_acc_key = f"val_{acc_key}"
    if acc_key in hist:
        ax2.plot(hist[acc_key], label=f'train_{acc_key}')
        if val_acc_key in hist:
            ax2.plot(hist[val_acc_key], label=val_acc_key)
        ax2.set_title('Model Accuracy')
        ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.legend(loc='lower right')
        ax2.grid(True)
        
    plt.tight_layout()
    plt.savefig(save_dir / 'accuracy_loss_curves.png', dpi=300)
    plt.close()
    logger.info(f"Saved training history plot to {save_dir / 'accuracy_loss_curves.png'}")

def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray,
    save_path: str | Path,
    class_names: list[str] = ['No Tumor', 'Tumor']
) -> None:
    """Plot and save confusion matrix as heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {save_path}")

def plot_roc_curve(
    y_true: np.ndarray, y_prob: np.ndarray,
    save_path: str | Path
) -> None:
    """Plot and save ROC curve with AUC score."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved ROC curve plot to {save_path}")

def generate_training_report(
    model: keras.Model,
    history: keras.callbacks.History | dict,
    X_test: np.ndarray, y_test: np.ndarray,
    output_dir: str | Path
) -> dict:
    """Generate comprehensive training report.
    
    Saves:
    - metrics.json (all numerical metrics)
    - accuracy_loss_curves.png
    - confusion_matrix.png  
    - roc_curve.png
    - classification_report.txt
    
    Returns:
        Dict of all computed metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Plot training history
    if history:
        plot_training_history(history, output_dir)
        
    # 2. Evaluate model
    logger.info("Evaluating model for report...")
    y_prob = model.predict(X_test, verbose=0)
    y_pred = (y_prob > 0.5).astype(int).flatten()
    y_true = y_test.flatten()
    
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True)
    }
    
    # 3. Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, output_dir / 'confusion_matrix.png')
    
    # 4. Plot ROC curve
    plot_roc_curve(y_true, y_prob, output_dir / 'roc_curve.png')
    
    # 5. Save metrics.json
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # 6. Save classification report txt
    report_str = classification_report(y_true, y_pred)
    with open(output_dir / 'classification_report.txt', 'w') as f:
        f.write("Classification Report\n")
        f.write("=====================\n\n")
        f.write(report_str)
        
    logger.info(f"Generated comprehensive training report in {output_dir}")
    return metrics
