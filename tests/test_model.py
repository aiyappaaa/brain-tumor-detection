import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from brain_tumor_detection.models.custom_cnn import build_custom_cnn
from brain_tumor_detection.models.mobilenet import build_mobilenet_v2
from brain_tumor_detection.models.factory import create_model, compile_model
from brain_tumor_detection.config import ModelConfig, TrainingConfig


class TestCustomCNN:
    def test_output_shape(self):
        model = build_custom_cnn(input_shape=(224, 224, 3))
        dummy = np.random.rand(1, 224, 224, 3).astype(np.float32)
        output = model.predict(dummy, verbose=0)
        assert output.shape == (1, 1)
    
    def test_output_range(self):
        """Output should be between 0 and 1 (sigmoid)."""
        model = build_custom_cnn()
        dummy = np.random.rand(2, 224, 224, 3).astype(np.float32)
        output = model.predict(dummy, verbose=0)
        assert np.all(output >= 0.0) and np.all(output <= 1.0)
    
    def test_model_name(self):
        model = build_custom_cnn()
        assert model.name == 'brain_tumor_cnn'
    
    def test_has_batch_norm(self):
        model = build_custom_cnn()
        layer_types = [type(l).__name__ for l in model.layers]
        assert 'BatchNormalization' in layer_types
    
    def test_has_dropout(self):
        model = build_custom_cnn()
        layer_types = [type(l).__name__ for l in model.layers]
        assert 'Dropout' in layer_types
    
    def test_custom_input_shape(self):
        model = build_custom_cnn(input_shape=(128, 128, 3))
        assert model.input_shape == (None, 128, 128, 3)


class TestMobileNetV2:
    def test_output_shape(self):
        model = build_mobilenet_v2(input_shape=(224, 224, 3))
        dummy = np.random.rand(1, 224, 224, 3).astype(np.float32)
        output = model.predict(dummy, verbose=0)
        assert output.shape == (1, 1)
    
    def test_frozen_base(self):
        model = build_mobilenet_v2(freeze_base=True)
        # The MobileNetV2 layers should not be trainable
        base_layers = [l for l in model.layers if 'mobilenet' in l.name.lower() or hasattr(l, 'layers')]
        # At least some layers should be non-trainable
        non_trainable = [l for l in model.layers if not l.trainable]
        assert len(non_trainable) > 0


class TestModelFactory:
    def test_create_custom_cnn(self):
        config = ModelConfig(architecture='custom_cnn')
        model = create_model(config)
        assert model is not None
    
    def test_create_mobilenet(self):
        config = ModelConfig(architecture='mobilenet_v2')
        model = create_model(config)
        assert model is not None
    
    def test_invalid_architecture(self):
        config = ModelConfig(architecture='invalid_arch')
        with pytest.raises((ValueError, KeyError)):
            create_model(config)
    
    def test_compile_model(self):
        config = ModelConfig(architecture='custom_cnn')
        training_config = TrainingConfig()
        model = create_model(config)
        compiled = compile_model(model, training_config)
        assert compiled.optimizer is not None
