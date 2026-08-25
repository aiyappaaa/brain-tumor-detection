"""Command-line interface for the Brain Tumor Detection pipeline.

Provides commands for training, evaluation, prediction, serving, and export.

Usage:
    python -m brain_tumor_detection train --config config/default.yaml
    python -m brain_tumor_detection predict --image path/to/mri.jpg --model outputs/checkpoints/best_model.keras
    python -m brain_tumor_detection serve --model outputs/checkpoints/best_model.keras --port 8000
    python -m brain_tumor_detection evaluate --model outputs/checkpoints/best_model.keras
    python -m brain_tumor_detection export --model outputs/checkpoints/best_model.keras --format savedmodel
"""

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure application-wide logging.

    Args:
        verbose: If True, set log level to DEBUG. Otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Suppress overly verbose TensorFlow logging
    logging.getLogger("tensorflow").setLevel(logging.WARNING)


def cmd_train(args: argparse.Namespace) -> None:
    """Train a model from scratch or with transfer learning."""
    from brain_tumor_detection.config import AppConfig
    from brain_tumor_detection.data.loader import prepare_data
    from brain_tumor_detection.training.trainer import Trainer

    config = AppConfig.from_yaml(args.config) if args.config else AppConfig.default()

    # Override config with CLI args if provided
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.architecture is not None:
        config.model.architecture = args.architecture

    logger.info("Starting training with architecture: %s", config.model.architecture)
    logger.info("Config: epochs=%d, batch_size=%d, lr=%.6f",
                config.training.epochs, config.training.batch_size,
                config.training.learning_rate)

    # Prepare data (split first, then augment training only)
    logger.info("Preparing data...")
    data = prepare_data(config)
    logger.info(
        "Data prepared — Train: %d, Val: %d, Test: %d",
        len(data["train"][0]),
        len(data["val"][0]),
        len(data["test"][0]),
    )

    # Train
    trainer = Trainer(config)
    trainer.setup()
    trainer.train(
        data["train"][0], data["train"][1],
        data["val"][0], data["val"][1],
    )

    # Evaluate on test set
    results = trainer.evaluate(data["test"][0], data["test"][1])

    # Generate report (plots, metrics JSON, etc.)
    trainer.generate_report(data["test"][0], data["test"][1])

    # Save final model
    model_path = trainer.save_model()
    logger.info("Model saved to %s", model_path)
    logger.info("Test Results: %s", results)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate a trained model on the test set."""
    from brain_tumor_detection.config import AppConfig
    from brain_tumor_detection.data.loader import prepare_data
    from brain_tumor_detection.models.factory import load_trained_model
    from brain_tumor_detection.training.metrics import evaluate_model, generate_training_report

    config = AppConfig.from_yaml(args.config) if args.config else AppConfig.default()
    logger.info("Evaluating model: %s", args.model)

    model = load_trained_model(args.model)

    # Prepare data
    data = prepare_data(config)
    X_test, y_test = data["test"]

    # Evaluate
    results = evaluate_model(model, X_test, y_test)

    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  F1 Score:  {results['f1']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  AUC:       {results['auc']:.4f}")
    print("=" * 50 + "\n")


def cmd_predict(args: argparse.Namespace) -> None:
    """Predict on a single MRI image."""
    from brain_tumor_detection.inference.predictor import Predictor

    image_path = Path(args.image)
    if not image_path.exists():
        logger.error("Image not found: %s", image_path)
        sys.exit(1)

    predictor = Predictor(args.model, image_size=tuple(args.image_size))
    result = predictor.predict(image_path)

    print("\n" + "=" * 50)
    print("  PREDICTION RESULT")
    print("=" * 50)
    print(f"  Image:       {image_path.name}")
    print(f"  Prediction:  {result['class']}")
    print(f"  Confidence:  {result['confidence']:.2%}")
    print(f"  Probability: {result['probability']:.4f}")
    print("=" * 50 + "\n")

    # Generate Grad-CAM explanation if requested
    if args.explain:
        from brain_tumor_detection.inference.gradcam import GradCAM

        gradcam = GradCAM(model=predictor.model)
        output_path = image_path.parent / f"{image_path.stem}_explanation.png"
        gradcam.generate_explanation(
            image_path,
            save_path=output_path,
            target_size=tuple(args.image_size),
        )
        print(f"  Grad-CAM explanation saved to: {output_path}\n")


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the FastAPI prediction server."""
    import uvicorn

    from brain_tumor_detection.api.app import create_app

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error("Model not found: %s", model_path)
        sys.exit(1)

    logger.info("Starting API server on %s:%d", args.host, args.port)
    app = create_app(model_path=model_path, image_size=tuple(args.image_size))
    uvicorn.run(app, host=args.host, port=args.port)


def cmd_export(args: argparse.Namespace) -> None:
    """Export a trained model to SavedModel or TFLite format."""
    import tensorflow as tf

    from brain_tumor_detection.models.factory import load_trained_model

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error("Model not found: %s", model_path)
        sys.exit(1)

    logger.info("Loading model from %s", model_path)
    model = load_trained_model(str(model_path))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.format == "savedmodel":
        export_path = output_dir / "saved_model"
        model.save(str(export_path))
        logger.info("Exported to SavedModel format: %s", export_path)
        print(f"\nModel exported to SavedModel: {export_path}")

    elif args.format == "tflite":
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        export_path = output_dir / "model.tflite"
        with open(export_path, "wb") as f:
            f.write(tflite_model)
        logger.info("Exported to TFLite format: %s", export_path)
        print(f"\nModel exported to TFLite: {export_path}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="brain-tumor-detection",
        description="Brain Tumor Detection — Production ML Pipeline",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- train ---
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", type=str, help="Path to config YAML file")
    train_parser.add_argument("--epochs", type=int, help="Override number of epochs")
    train_parser.add_argument("--batch-size", type=int, help="Override batch size")
    train_parser.add_argument(
        "--architecture",
        type=str,
        choices=["custom_cnn", "mobilenet_v2"],
        help="Model architecture to use",
    )

    # --- evaluate ---
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    eval_parser.add_argument("--config", type=str, help="Path to config YAML file")

    # --- predict ---
    predict_parser = subparsers.add_parser("predict", help="Predict on an MRI image")
    predict_parser.add_argument("--image", type=str, required=True, help="Path to MRI image")
    predict_parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    predict_parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224], help="Image size (H W)")
    predict_parser.add_argument("--explain", action="store_true", help="Generate Grad-CAM explanation")

    # --- serve ---
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    serve_parser.add_argument("--port", type=int, default=8000, help="Server port")
    serve_parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224], help="Image size (H W)")

    # --- export ---
    export_parser = subparsers.add_parser("export", help="Export model to deployment format")
    export_parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    export_parser.add_argument(
        "--format",
        type=str,
        choices=["savedmodel", "tflite"],
        default="savedmodel",
        help="Export format",
    )
    export_parser.add_argument("--output", type=str, default="outputs/exports", help="Output directory")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    setup_logging(args.verbose)

    commands = {
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "predict": cmd_predict,
        "serve": cmd_serve,
        "export": cmd_export,
    }

    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error("Command '%s' failed: %s", args.command, e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
