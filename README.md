# High-Performance DataLoader for PyTorch

A production-ready, high-performance PyTorch DataLoader implementation with asynchronous prefetching, multi-worker parallelism, in-memory caching, and custom augmentations.

## Motivation

In real-world deep learning systems, data loading often becomes the primary
training bottleneck rather than model computation. This project focuses on
designing an optimized PyTorch input pipeline that maximizes throughput,
minimizes I/O latency, and scales across CPU and GPU environments.


## Features

✅ **Multi-worker Parallelism** - Parallel data loading across multiple CPU cores  
✅ **LRU Caching** - Thread-safe in-memory cache for frequently accessed images  
✅ **Asynchronous GPU Prefetching** - Overlap data transfer with computation  
✅ **Dataset Preloading** - Optional full dataset loading into RAM  
✅ **Custom Augmentations** - Flexible data augmentation pipeline  
✅ **Memory Efficient** - Configurable caching with automatic eviction  
✅ **Comprehensive Benchmarks** - Built-in performance analysis tools  

## Installation

### Requirements

```bash
python >= 3.8
torch >= 1.9.0
torchvision >= 0.10.0
numpy >= 1.19.0
Pillow >= 8.0.0
psutil >= 5.8.0
pytest >= 6.0.0
matplotlib >= 3.3.0
```

### Setup

```bash
cd high-performance-dataloader

# Install dependencies
pip install torch torchvision numpy pillow psutil pytest matplotlib

# Verify installation
python -m pytest test_dataloader.py -v
```

## Dataset Note

This project uses a synthetic image dataset generated at runtime for:
- Reproducibility
- Lightweight repository size
- Consistent benchmarking across systems

The DataLoader is fully compatible with real-world datasets
(ImageNet, CIFAR, medical images, etc.) by simply providing image paths.


## Architecture

### Core Components

1. **HighPerformanceDataset**
   - Custom PyTorch Dataset with caching
   - Optional preloading for small datasets
   - Efficient image loading and preprocessing

2. **LRUCache**
   - Thread-safe least-recently-used cache
   - Configurable capacity
   - Automatic eviction of old items

3. **PrefetchLoader**
   - Asynchronous GPU prefetching
   - CUDA stream-based overlap
   - Non-blocking data transfer

4. **Custom Transforms**
   - Training augmentations (random crop, flip, color jitter)
   - Validation preprocessing
   - ImageNet normalization

### Performance Optimizations

| Optimization | Description | Speedup |
|--------------|-------------|---------|
| Multi-worker | Parallel data loading | 3-5x |
| LRU Cache | Reduce I/O overhead | 1.5-2x |
| GPU Prefetch | Overlap transfer & compute | 1.2-1.5x |
| Pin Memory | Faster CPU→GPU transfer | 1.1-1.3x |
| Preloading | Maximum throughput | 5-10x* |

*Preloading provides best throughput but requires dataset to fit in RAM

## Benchmarking

### Run Benchmarks

```bash
# Basic benchmark
python benchmark_script.py
```

### Sample Results

```
Configuration                             Throughput      Memory
                                         (imgs/s)        (MB)
--------------------------------------------------------------------------------
Baseline (no optimizations)              450.2           342.5
Multi-worker (4 workers)                 1523.7          456.3
With LRU cache (500 images)              1847.9          678.1
Fully preloaded                          4521.3          1823.7
Full optimizations + GPU prefetch        2134.8          712.4

SPEEDUP vs BASELINE
--------------------------------------------------------------------------------
Multi-worker (4 workers)                 3.38x
With LRU cache (500 images)              4.10x
Fully preloaded                          10.04x
Full optimizations + GPU prefetch        4.74x
```

## API Reference

### create_dataloader()

```python
create_dataloader(
    data_paths: List[str],
    labels: List[int],
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
    cache_size: int = 1000,
    preload: bool = False,
    augment: bool = True,
    pin_memory: bool = True,
    prefetch_device: Optional[torch.device] = None
) -> DataLoader
```

**Parameters:**
- `data_paths`: List of image file paths
- `labels`: Corresponding labels for each image
- `batch_size`: Number of samples per batch
- `num_workers`: Number of worker processes for data loading
- `img_size`: Target image size (images resized to img_size × img_size)
- `cache_size`: Number of images to cache (0 = no caching)
- `preload`: If True, load entire dataset into RAM
- `augment`: If True, apply data augmentation
- `pin_memory`: Use pinned memory for faster CPU→GPU transfer
- `prefetch_device`: Device to prefetch batches to (None = no prefetching)

### HighPerformanceDataset

```python
HighPerformanceDataset(
    data_paths: List[str],
    labels: List[int],
    transform: Optional[Callable] = None,
    cache_size: int = 1000,
    preload: bool = False
)
```

Custom Dataset class with caching and preloading capabilities.

### PrefetchLoader

```python
PrefetchLoader(loader: DataLoader, device: torch.device)
```

Wraps a DataLoader to prefetch batches to GPU asynchronously.

## Testing

Run the complete test suite:

```bash
# All tests
pytest test_dataloader.py -v

# Specific test class
pytest test_dataloader.py::TestLRUCache -v

# With coverage
pytest test_dataloader.py --cov=dataloader_module --cov-report=html
```

## Configuration Guide

### Recommended Settings

**Small Dataset (< 10K images, fits in RAM)**
```python
loader = create_dataloader(
    ...,
    num_workers=4,
    cache_size=0,
    preload=True,
    prefetch_device=torch.device('cuda')
)
```

**Medium Dataset (10K-100K images)**
```python
loader = create_dataloader(
    ...,
    num_workers=4,
    cache_size=5000,
    preload=False,
    prefetch_device=torch.device('cuda')
)
```

**Large Dataset (> 100K images)**
```python
loader = create_dataloader(
    ...,
    num_workers=8,
    cache_size=10000,
    preload=False,
    prefetch_device=torch.device('cuda')
)
```

### Memory Considerations

- **Preloading**: ~3-5 bytes per pixel × num_images × (224×224) / (1024²) MB
- **Caching**: Similar formula × cache_size
- **Workers**: ~50-100 MB per worker

Example: 10K images (224×224), preloaded ≈ 1.5 GB RAM

## Troubleshooting

### Common Issues

**Out of Memory**
- Reduce `cache_size` or set to 0
- Reduce `num_workers`
- Disable `preload`
- Reduce `batch_size`

**Slow Performance**
- Increase `num_workers` (typically 4-8)
- Enable `cache_size` for repeated epochs
- Use `prefetch_device` for GPU training
- Ensure SSD storage for faster I/O

**DataLoader Hangs**
- Check file permissions
- Verify all image paths exist
- Try `num_workers=0` to isolate issue
- Check for corrupted images

## Advanced Usage

### Custom Transforms

```python
import torchvision.transforms as T

custom_transform = T.Compose([
    T.Resize(256),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.3, 0.3, 0.3),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = HighPerformanceDataset(
    data_paths=paths,
    labels=labels,
    transform=custom_transform
)
```

## Performance Tips

1. **Use SSD storage** for image data
2. **Pin memory** for CUDA training (`pin_memory=True`)
3. **Tune num_workers** (usually 4-8, depends on CPU cores)
4. **Enable prefetching** for GPU training
5. **Use appropriate cache_size** based on dataset size
6. **Preload small datasets** that fit in RAM
7. **Profile your pipeline** to identify bottlenecks