"""Configuration management for the brain tumor detection app."""
import yaml
import math
from pathlib import Path
from pydantic import BaseModel, Field, model_validator

class DataConfig(BaseModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    image_size: tuple[int, int] = (224, 224)
    split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
    random_seed: int = 42

    @model_validator(mode='after')
    def validate_split_ratios(self) -> 'DataConfig':
        if not math.isclose(sum(self.split_ratios), 1.0, rel_tol=1e-6):
            raise ValueError("split_ratios must sum to 1.0")
        return self

    @model_validator(mode='after')
    def validate_image_size(self) -> 'DataConfig':
        if self.image_size[0] <= 0 or self.image_size[1] <= 0:
            raise ValueError("image_size dimensions must be positive")
        return self

class AugmentationConfig(BaseModel):
    enabled: bool = True
    rotation_range: int = 15
    width_shift_range: float = 0.1
    height_shift_range: float = 0.1
    shear_range: float = 0.1
    brightness_range: tuple[float, float] = (0.8, 1.2)
    horizontal_flip: bool = True
    vertical_flip: bool = True
    fill_mode: str = "nearest"

class ModelConfig(BaseModel):
    architecture: str = "custom_cnn"
    input_shape: tuple[int, int, int] = (224, 224, 3)
    dropout_rate: float = 0.5
    classification_threshold: float = 0.5

class TrainingConfig(BaseModel):
    optimizer: str = "adam"
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 50
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5

class OutputConfig(BaseModel):
    checkpoint_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"
    metrics_dir: str = "outputs/metrics"
    export_dir: str = "outputs/exports"

class AppConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        """Load config from YAML file."""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @classmethod
    def default(cls) -> "AppConfig":
        """Return default config."""
        return cls()

    def save_yaml(self, path: str | Path) -> None:
        """Save config to YAML file."""
        config_dict = self.model_dump()
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config_dict, f, sort_keys=False)
