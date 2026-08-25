import argparse
import sys
import logging
from pathlib import Path

import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Export trained Keras model")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained .keras model")
    parser.add_argument("--format", type=str, choices=["saved_model", "tflite"], default="saved_model", help="Export format")
    parser.add_argument("--output", type=str, required=True, help="Output path/directory")
    return parser.parse_args()

def export_saved_model(model_path: Path, output_dir: Path):
    try:
        model = tf.keras.models.load_model(model_path)
        tf.saved_model.save(model, str(output_dir))
        logger.info(f"Successfully exported SavedModel to {output_dir}")
    except Exception as e:
        logger.error(f"Failed to export SavedModel: {e}")
        sys.exit(1)

def export_tflite(model_path: Path, output_path: Path):
    try:
        model = tf.keras.models.load_model(model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Quantization for optimal size
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(tflite_model)
            
        logger.info(f"Successfully exported TFLite model to {output_path}")
    except Exception as e:
        logger.error(f"Failed to export TFLite model: {e}")
        sys.exit(1)

def main():
    args = parse_args()
    
    model_path = Path(args.model)
    output_path = Path(args.output)
    
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)
        
    if args.format == "saved_model":
        export_saved_model(model_path, output_path)
    elif args.format == "tflite":
        export_tflite(model_path, output_path)

if __name__ == "__main__":
    main()
