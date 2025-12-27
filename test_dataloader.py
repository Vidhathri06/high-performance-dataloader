"""
Test suite for High-Performance DataLoader
Run with: pytest test_dataloader.py -v
"""

import pytest 
import torch
import numpy as np
from PIL import Image
import tempfile
import shutil
from pathlib import Path
from dataloader_module import (
    LRUCache, 
    HighPerformanceDataset, 
    PrefetchLoader,
    create_dataloader,
    get_default_transforms
)


@pytest.fixture
def synthetic_dataset():
    """Create a small synthetic dataset for testing."""
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    
    num_images = 50
    img_size = 64
    image_paths = []
    labels = []
    
    for i in range(num_images):
        img_array = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = temp_path / f"img_{i:03d}.jpg"
        img.save(img_path)
        image_paths.append(str(img_path))
        labels.append(i % 5)  # 5 classes
    
    yield image_paths, labels, temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


class TestLRUCache:
    """Test LRU cache implementation."""
    
    def test_cache_basic_operations(self):
        cache = LRUCache(capacity=3)
        
        # Test put and get
        cache.put('a', 1)
        cache.put('b', 2)
        cache.put('c', 3)
        
        assert cache.get('a') == 1
        assert cache.get('b') == 2
        assert cache.get('c') == 3
    
    def test_cache_eviction(self):
        cache = LRUCache(capacity=2)
        
        cache.put('a', 1)
        cache.put('b', 2)
        cache.put('c', 3)  # Should evict 'a'
        
        assert cache.get('a') is None
        assert cache.get('b') == 2
        assert cache.get('c') == 3
    
    def test_cache_lru_order(self):
        cache = LRUCache(capacity=2)
        
        cache.put('a', 1)
        cache.put('b', 2)
        cache.get('a')  # Access 'a', making 'b' least recently used
        cache.put('c', 3)  # Should evict 'b'
        
        assert cache.get('a') == 1
        assert cache.get('b') is None
        assert cache.get('c') == 3
    
    def test_cache_clear(self):
        cache = LRUCache(capacity=3)
        cache.put('a', 1)
        cache.put('b', 2)
        cache.clear()
        
        assert cache.get('a') is None
        assert cache.get('b') is None


class TestHighPerformanceDataset:
    """Test custom Dataset implementation."""
    
    def test_dataset_length(self, synthetic_dataset):
        image_paths, labels, _ = synthetic_dataset
        dataset = HighPerformanceDataset(image_paths, labels)
        
        assert len(dataset) == len(image_paths)
    
    def test_dataset_getitem(self, synthetic_dataset):
        image_paths, labels, _ = synthetic_dataset
        transform = get_default_transforms(img_size=64, augment=False)
        dataset = HighPerformanceDataset(image_paths, labels, transform=transform)
        
        img, label = dataset[0]
        
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 64, 64)
        assert isinstance(label, int)
        assert 0 <= label < 5
    
    def test_dataset_with_cache(self, synthetic_dataset):
        image_paths, labels, _ = synthetic_dataset
        dataset = HighPerformanceDataset(
            image_paths, labels, cache_size=10
        )
        
        # First access
        img1, label1 = dataset[0]
        
        # Second access (should hit cache)
        img2, label2 = dataset[0]
        
        assert label1 == label2
    
    def test_dataset_preload(self, synthetic_dataset):
        image_paths, labels, _ = synthetic_dataset
        dataset = HighPerformanceDataset(
            image_paths[:10], labels[:10], preload=True
        )
        
        assert len(dataset.preloaded_data) == 10
        
        img, label = dataset[0]
        assert img is not None
    
    def test_dataset_transforms(self, synthetic_dataset):
        image_paths, labels, _ = synthetic_dataset
        
        transform = get_default_transforms(img_size=128, augment=True)
        dataset = HighPerformanceDataset(
            image_paths, labels, transform=transform
        )
        
        img, _ = dataset[0]
        assert img.shape == (3, 128, 128)
        assert img.min() >= -3 and img.max() <= 3  # Normalized


class TestDataLoader:
    """Test DataLoader creation and functionality."""
    
    def test_create_dataloader_basic(self, synthetic_dataset):
        image_paths, labels, _ = synthetic_dataset
        
        loader = create_dataloader(
            data_paths=image_paths,
            labels=labels,
            batch_size=8,
            num_workers=0
        )
        
        assert len(loader) == len(image_paths) // 8
        
        batch_images, batch_labels = next(iter(loader))
        assert batch_images.shape[0] == 8
        assert len(batch_labels) == 8
    
    def test_dataloader_batch_sizes(self, synthetic_dataset):
        image_paths, labels, _ = synthetic_dataset
        
        for batch_size in [4, 8, 16]:
            loader = create_dataloader(
                data_paths=image_paths,
                labels=labels,
                batch_size=batch_size,
                num_workers=0
            )
            
            batch = next(iter(loader))
            assert batch[0].shape[0] <= batch_size
    
    def test_dataloader_with_workers(self, synthetic_dataset):
        image_paths, labels, _ = synthetic_dataset
        
        loader = create_dataloader(
            data_paths=image_paths,
            labels=labels,
            batch_size=8,
            num_workers=2
        )
        
        # Should be able to iterate through all batches
        batch_count = 0
        for _ in loader:
            batch_count += 1
        
        assert batch_count > 0
    
    def test_dataloader_with_cache(self, synthetic_dataset):
        image_paths, labels, _ = synthetic_dataset
        
        loader = create_dataloader(
            data_paths=image_paths,
            labels=labels,
            batch_size=8,
            cache_size=20,
            num_workers=0
        )
        
        # Iterate twice - second iteration should be faster due to cache
        for _ in loader:
            pass
        
        for _ in loader:
            pass


class TestPrefetchLoader:
    """Test asynchronous prefetching."""
    
    def test_prefetch_loader_cpu(self, synthetic_dataset):
        image_paths, labels, _ = synthetic_dataset
        
        base_loader = create_dataloader(
            data_paths=image_paths,
            labels=labels,
            batch_size=8,
            num_workers=0
        )
        
        device = torch.device('cpu')
        prefetch_loader = PrefetchLoader(base_loader, device)
        
        batch_count = 0
        for batch_images, batch_labels in prefetch_loader:
            batch_count += 1
            assert batch_images.device.type == 'cpu'
        
        assert batch_count == len(base_loader)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_prefetch_loader_cuda(self, synthetic_dataset):
        image_paths, labels, _ = synthetic_dataset
        
        base_loader = create_dataloader(
            data_paths=image_paths,
            labels=labels,
            batch_size=8,
            num_workers=0
        )
        
        device = torch.device('cuda')
        prefetch_loader = PrefetchLoader(base_loader, device)
        
        for batch_images, batch_labels in prefetch_loader:
            assert batch_images.device.type == 'cuda'
            break  # Just test first batch


class TestTransforms:
    """Test data augmentation transforms."""
    
    def test_training_transforms(self):
        transform = get_default_transforms(img_size=224, augment=True)
        
        # Create random image
        img = Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )
        
        transformed = transform(img)
        
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, 224, 224)
    
    def test_validation_transforms(self):
        transform = get_default_transforms(img_size=224, augment=False)
        
        img = Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )
        
        transformed = transform(img)
        
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, 224, 224)


class TestIntegration:
    """Integration tests for complete workflow."""
    
    def test_end_to_end_training_step(self, synthetic_dataset):
        """Test a complete forward pass through a model."""
        image_paths, labels, _ = synthetic_dataset
        
        # Create loader
        loader = create_dataloader(
            data_paths=image_paths,
            labels=labels,
            batch_size=8,
            num_workers=0,
            img_size=64
        )
        
        # Simple model
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(16, 5)
        )
        
        # Get batch and forward pass
        images, targets = next(iter(loader))
        outputs = model(images)
        
        assert outputs.shape == (images.size(0), 5)
    
    def test_multiple_epochs(self, synthetic_dataset):
        """Test iterating through multiple epochs."""
        image_paths, labels, _ = synthetic_dataset
        
        loader = create_dataloader(
            data_paths=image_paths,
            labels=labels,
            batch_size=8,
            num_workers=0
        )
        
        for epoch in range(3):
            batch_count = 0
            for _ in loader:
                batch_count += 1
            
            assert batch_count == len(loader)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])