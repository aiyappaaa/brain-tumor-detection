import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from brain_tumor_detection.config import AppConfig
from brain_tumor_detection.data.preprocessing import load_and_preprocess_image
import logging

logger = logging.getLogger(__name__)

def discover_images(raw_dir: str | Path) -> list[tuple[Path, int]]:
    """Discover all images in raw_dir/yes and raw_dir/no.
    
    Args:
        raw_dir: Path to the raw data directory.
        
    Returns:
        List of (image_path, label) tuples where label=1 for 'yes' (tumor), 0 for 'no'.
        
    Raises:
        FileNotFoundError: If raw_dir does not exist or missing yes/no subdirectories.
    """
    base_dir = Path(raw_dir)
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {base_dir}")
        
    yes_dir = base_dir / "yes"
    no_dir = base_dir / "no"
    
    if not yes_dir.is_dir() or not no_dir.is_dir():
        raise FileNotFoundError(f"Missing 'yes' or 'no' subdirectory in {base_dir}")
        
    images = []
    
    # Label 1 for yes, 0 for no
    valid_exts = {".jpg", ".jpeg", ".png"}
    for img_path in yes_dir.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in valid_exts:
            images.append((img_path, 1))
            
    for img_path in no_dir.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in valid_exts:
            images.append((img_path, 0))
            
    logger.info(f"Discovered {len(images)} images in {base_dir}")
    return images

def create_splits(
    image_label_pairs: list[tuple[Path, int]],
    split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    random_seed: int = 42
) -> dict[str, list[tuple[Path, int]]]:
    """Split data into train/val/test using stratified splitting.
    
    CRITICAL: This operates on ORIGINAL images only. Augmentation is applied
    separately to the training set only, preventing data leakage.
    
    Args:
        image_label_pairs: List of (image_path, label).
        split_ratios: Tuple of (train, val, test) proportions.
        random_seed: Random seed for reproducibility.
        
    Returns:
        Dict with keys 'train', 'val', 'test', each containing list of (path, label).
    """
    if sum(split_ratios) != 1.0:
        logger.warning(f"Split ratios {split_ratios} do not sum to 1.0. Normalizing.")
        total = sum(split_ratios)
        split_ratios = tuple(x / total for x in split_ratios)
        
    train_ratio, val_ratio, test_ratio = split_ratios
    
    paths = [pair[0] for pair in image_label_pairs]
    labels = [pair[1] for pair in image_label_pairs]
    
    # First split: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        paths, labels, test_size=temp_ratio, random_state=random_seed, stratify=labels
    )
    
    # Second split: Val vs Test
    test_ratio_relative = test_ratio / temp_ratio
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_ratio_relative, random_state=random_seed, stratify=y_temp
    )
    
    train_pairs = list(zip(X_train, y_train))
    val_pairs = list(zip(X_val, y_val))
    test_pairs = list(zip(X_test, y_test))
    
    logger.info(f"Splits created: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")
    
    return {
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs
    }

def save_split_metadata(
    splits: dict[str, list[tuple[Path, int]]],
    output_path: str | Path
) -> None:
    """Save split file lists as JSON for reproducibility.
    
    Args:
        splits: Dictionary of split names to lists of (path, label) tuples.
        output_path: File path to save the JSON metadata.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    serializable_splits = {}
    for split_name, data in splits.items():
        serializable_splits[split_name] = [{"path": p.as_posix(), "label": l} for p, l in data]
        
    with open(path, "w") as f:
        json.dump(serializable_splits, f, indent=4)
        
    logger.info(f"Split metadata saved to {path}")

def load_split_metadata(metadata_path: str | Path) -> dict[str, list[tuple[str, int]]]:
    """Load previously saved split metadata.
    
    Args:
        metadata_path: File path to load the JSON metadata from.
        
    Returns:
        Dictionary of split names to lists of (path, label) tuples.
    """
    path = Path(metadata_path)
    if not path.is_file():
        raise FileNotFoundError(f"Split metadata not found at {path}")
        
    with open(path, "r") as f:
        data = json.load(f)
        
    splits = {}
    for split_name, items in data.items():
        splits[split_name] = [(item["path"], item["label"]) for item in items]
        
    logger.info(f"Split metadata loaded from {path}")
    return splits

def load_dataset(
    split_data: list[tuple[Path, int]],
    target_size: tuple[int, int] = (224, 224)
) -> tuple[np.ndarray, np.ndarray]:
    """Load and preprocess all images for a given split.
    
    Args:
        split_data: List of (image_path, label) tuples.
        target_size: Desired output size as (width, height).
        
    Returns:
        (X, y) where X is shape (N, H, W, 3) and y is shape (N, 1).
    """
    X = []
    y = []
    
    for path, label in split_data:
        try:
            img = load_and_preprocess_image(path, target_size)
            X.append(img)
            y.append(label)
        except Exception as e:
            logger.error(f"Error processing {path}: {e}")
            continue
            
    return np.array(X), np.array(y).reshape(-1, 1)

def prepare_data(config: AppConfig) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Full data preparation pipeline.
    
    1. Discover images from raw_dir
    2. Create stratified splits (on original images only)
    3. Save split metadata
    4. Load and preprocess each split
    
    Args:
        config: Application configuration.
        
    Returns:
        Dict with 'train', 'val', 'test' keys, each containing (X, y) tuple.
    """
    logger.info("Starting data preparation pipeline...")
    
    raw_dir = Path(config.data.raw_dir)
    image_pairs = discover_images(raw_dir)
    
    splits = create_splits(
        image_pairs, 
        split_ratios=config.data.split_ratios, 
        random_seed=config.data.random_seed
    )
    
    metadata_path = Path(config.data.processed_dir) / "split_metadata.json"
    save_split_metadata(splits, metadata_path)
    
    processed_data = {}
    for split_name, split_data in splits.items():
        logger.info(f"Processing {split_name} dataset...")
        X, y = load_dataset(split_data, config.data.image_size)
        processed_data[split_name] = (X, y)
        logger.info(f"{split_name.capitalize()} set shape: X={X.shape}, y={y.shape}")
        
    return processed_data
