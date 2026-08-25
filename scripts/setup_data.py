"""Setup data for brain tumor detection by copying source images."""
import argparse
import logging
import shutil
import cv2
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def validate_image(filepath: Path) -> bool:
    """Validate that an image is readable by OpenCV."""
    try:
        img = cv2.imread(str(filepath))
        return img is not None
    except Exception:
        return False

def setup_data(source_dir: Path, target_dir: Path) -> None:
    """Copy and validate images from source to target directory."""
    if not source_dir.exists():
        logger.error(f"Source directory does not exist: {source_dir}")
        return

    classes = ["yes", "no"]
    stats = {}

    for cls in classes:
        src_cls_dir = source_dir / cls
        tgt_cls_dir = target_dir / cls

        if not src_cls_dir.exists():
            logger.warning(f"Source class directory missing: {src_cls_dir}")
            continue

        tgt_cls_dir.mkdir(parents=True, exist_ok=True)
        copied_count = 0
        invalid_count = 0

        for img_path in src_cls_dir.glob("*"):
            if img_path.is_file():
                if validate_image(img_path):
                    shutil.copy2(img_path, tgt_cls_dir / img_path.name)
                    copied_count += 1
                else:
                    logger.warning(f"Invalid or unreadable image: {img_path}")
                    invalid_count += 1

        stats[cls] = {"copied": copied_count, "invalid": invalid_count}

    logger.info("Data Setup Summary:")
    for cls, info in stats.items():
        logger.info(f"Class '{cls}': {info['copied']} images copied, {info['invalid']} invalid.")

def main() -> None:
    parser = argparse.ArgumentParser(description="Setup dataset for brain tumor detection")
    parser.add_argument("--source", type=str, required=True, help="Source directory containing 'yes' and 'no' folders")
    parser.add_argument("--target", type=str, default="data/raw", help="Target directory for raw data")
    args = parser.parse_args()

    setup_data(Path(args.source), Path(args.target))

if __name__ == "__main__":
    main()
