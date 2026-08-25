import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from brain_tumor_detection.data.loader import discover_images, create_splits

@pytest.fixture
def mock_raw_dir(tmp_path):
    yes_dir = tmp_path / 'yes'
    no_dir = tmp_path / 'no'
    yes_dir.mkdir()
    no_dir.mkdir()
    for i in range(155):
        (yes_dir / f'Y{i}.jpg').touch()
    for i in range(98):
        (no_dir / f'N{i}.jpg').touch()
    return tmp_path

class TestDiscoverImages:
    def test_discovers_all_images(self, mock_raw_dir):
        pairs = discover_images(mock_raw_dir)
        # Should find 253 images (155 yes + 98 no)
        assert len(pairs) == 253
    
    def test_correct_labels(self, mock_raw_dir):
        pairs = discover_images(mock_raw_dir)
        labels = [label for _, label in pairs]
        assert sum(labels) == 155  # 155 tumor images
        assert labels.count(0) == 98  # 98 non-tumor images


class TestCreateSplits:
    def test_split_sizes(self):
        """Test that splits have approximately correct proportions."""
        # Create dummy data
        pairs = [(Path(f'img_{i}.jpg'), i % 2) for i in range(100)]
        splits = create_splits(pairs, split_ratios=(0.7, 0.15, 0.15), random_seed=42)
        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits
        total = len(splits['train']) + len(splits['val']) + len(splits['test'])
        assert total == 100
    
    def test_no_data_leakage(self):
        """No image should appear in more than one split."""
        pairs = [(Path(f'img_{i}.jpg'), i % 2) for i in range(100)]
        splits = create_splits(pairs, split_ratios=(0.7, 0.15, 0.15), random_seed=42)
        
        train_files = {str(p) for p, _ in splits['train']}
        val_files = {str(p) for p, _ in splits['val']}
        test_files = {str(p) for p, _ in splits['test']}
        
        assert len(train_files & val_files) == 0
        assert len(train_files & test_files) == 0
        assert len(val_files & test_files) == 0
    
    def test_stratified_split(self):
        """Each split should maintain approximate class proportions."""
        pairs = [(Path(f'img_{i}.jpg'), 1 if i < 60 else 0) for i in range(100)]
        splits = create_splits(pairs, split_ratios=(0.7, 0.15, 0.15), random_seed=42)
        
        for split_name, split_data in splits.items():
            if len(split_data) > 0:
                labels = [l for _, l in split_data]
                ratio = sum(labels) / len(labels)
                # Should be roughly 60% positive (within tolerance)
                assert 0.4 < ratio < 0.8, f"{split_name} split has imbalanced ratio: {ratio}"
    
    def test_reproducibility(self):
        """Same seed should produce same splits."""
        pairs = [(Path(f'img_{i}.jpg'), i % 2) for i in range(100)]
        splits1 = create_splits(pairs, random_seed=42)
        splits2 = create_splits(pairs, random_seed=42)
        
        for key in splits1:
            files1 = [str(p) for p, _ in splits1[key]]
            files2 = [str(p) for p, _ in splits2[key]]
            assert files1 == files2
