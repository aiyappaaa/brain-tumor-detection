import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from brain_tumor_detection.config import AppConfig, DataConfig

def test_default_config():
    config = AppConfig.default()
    assert config.data.raw_dir == "data/raw"
    assert config.training.optimizer == "adam"

def test_load_save_config(tmp_path):
    config_file = tmp_path / "config.yaml"
    config = AppConfig.default()
    config.training.epochs = 100
    config.save_yaml(config_file)
    
    loaded = AppConfig.from_yaml(config_file)
    assert loaded.training.epochs == 100
    assert loaded.data.split_ratios == (0.70, 0.15, 0.15)

def test_split_ratios_validation():
    # Valid
    DataConfig(split_ratios=(0.8, 0.1, 0.1))
    
    # Invalid
    with pytest.raises(ValueError, match="split_ratios must sum to 1.0"):
        DataConfig(split_ratios=(0.8, 0.1, 0.2))

def test_image_size_validation():
    with pytest.raises(ValueError, match="image_size dimensions must be positive"):
        DataConfig(image_size=(0, 224))
