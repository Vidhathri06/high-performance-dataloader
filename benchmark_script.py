# """
# Benchmark script for High-Performance DataLoader
# Tests throughput, memory usage, and performance across different configurations.
# """

# import torch
# import time
# import psutil
# import os
# from pathlib import Path
# import numpy as np
# from PIL import Image
# import tempfile
# import shutil
# from dataloader_module import create_dataloader, HighPerformanceDataset


# def create_synthetic_dataset(num_images: int, img_size: int = 224, output_dir: str = None):
#     """Create synthetic dataset for benchmarking."""
#     if output_dir is None:
#         output_dir = tempfile.mkdtemp()
    
#     output_path = Path(output_dir)
#     output_path.mkdir(exist_ok=True)
    
#     image_paths = []
#     labels = []
    
#     print(f"Creating {num_images} synthetic images...")
#     for i in range(num_images):
#         # Generate random image
#         img_array = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
#         img = Image.fromarray(img_array)
        
#         # Save image
#         img_path = output_path / f"img_{i:05d}.jpg"
#         img.save(img_path, quality=95)
        
#         image_paths.append(str(img_path))
#         labels.append(i % 10)  # 10 classes
        
#         if (i + 1) % 100 == 0:
#             print(f"Created {i + 1}/{num_images} images")
    
#     return image_paths, labels, output_dir


# def get_memory_usage():
#     """Get current memory usage in MB."""
#     process = psutil.Process(os.getpid())
#     return process.memory_info().rss / 1024 / 1024


# def benchmark_dataloader(
#     data_paths,
#     labels,
#     config_name,
#     batch_size=32,
#     num_workers=4,
#     cache_size=0,
#     preload=False,
#     prefetch_device=None,
#     num_epochs=3
# ):
#     """Benchmark a specific DataLoader configuration."""
#     print(f"\n{'='*60}")
#     print(f"Benchmarking: {config_name}")
#     print(f"{'='*60}")
#     print(f"Batch size: {batch_size}")
#     print(f"Num workers: {num_workers}")
#     print(f"Cache size: {cache_size}")
#     print(f"Preload: {preload}")
#     print(f"Prefetch device: {prefetch_device}")
    
#     # Record initial memory
#     initial_memory = get_memory_usage()
    
#     # Create dataloader
#     start_time = time.time()
#     loader = create_dataloader(
#         data_paths=data_paths,
#         labels=labels,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         cache_size=cache_size,
#         preload=preload,
#         prefetch_device=prefetch_device
#     )
#     setup_time = time.time() - start_time
    
#     setup_memory = get_memory_usage()
    
#     # Benchmark loading
#     batch_times = []
#     samples_processed = 0
    
#     for epoch in range(num_epochs):
#         epoch_start = time.time()
        
#         for batch_idx, (images, targets) in enumerate(loader):
#             batch_start = time.time()
            
#             # Simulate some processing
#             if prefetch_device is None and torch.cuda.is_available():
#                 images = images.cuda(non_blocking=True)
            
#             samples_processed += images.size(0)
#             batch_time = time.time() - batch_start
#             batch_times.append(batch_time)
            
#             if batch_idx % 10 == 0:
#                 print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, "
#                       f"Time: {batch_time:.4f}s, "
#                       f"Throughput: {images.size(0)/batch_time:.1f} imgs/s")
        
#         epoch_time = time.time() - epoch_start
#         epoch_throughput = len(data_paths) / epoch_time
#         print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s, "
#               f"Throughput: {epoch_throughput:.1f} imgs/s")
    
#     peak_memory = get_memory_usage()
    
#     # Calculate statistics
#     results = {
#         'config_name': config_name,
#         'setup_time': setup_time,
#         'mean_batch_time': np.mean(batch_times),
#         'std_batch_time': np.std(batch_times),
#         'min_batch_time': np.min(batch_times),
#         'max_batch_time': np.max(batch_times),
#         'throughput': samples_processed / sum(batch_times),
#         'initial_memory_mb': initial_memory,
#         'setup_memory_mb': setup_memory,
#         'peak_memory_mb': peak_memory,
#         'memory_overhead_mb': peak_memory - initial_memory
#     }
    
#     # Print summary
#     print(f"\n{'='*60}")
#     print(f"Results Summary: {config_name}")
#     print(f"{'='*60}")
#     print(f"Setup time: {results['setup_time']:.2f}s")
#     print(f"Mean batch time: {results['mean_batch_time']:.4f}s ± {results['std_batch_time']:.4f}s")
#     print(f"Throughput: {results['throughput']:.1f} images/second")
#     print(f"Memory overhead: {results['memory_overhead_mb']:.1f} MB")
#     print(f"Peak memory: {results['peak_memory_mb']:.1f} MB")
    
#     return results


# def run_comprehensive_benchmark(num_images=1000, batch_size=32):
#     """Run comprehensive benchmark across multiple configurations."""
    
#     # Create synthetic dataset
#     print("Setting up benchmark dataset...")
#     data_paths, labels, temp_dir = create_synthetic_dataset(num_images)
    
#     # Device setup
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"\nUsing device: {device}")
    
#     # Define configurations to test
#     configs = [
#         {
#             'name': 'Baseline (no optimizations)',
#             'num_workers': 0,
#             'cache_size': 0,
#             'preload': False,
#             'prefetch_device': None
#         },
#         {
#             'name': 'Multi-worker (4 workers)',
#             'num_workers': 4,
#             'cache_size': 0,
#             'preload': False,
#             'prefetch_device': None
#         },
#         {
#             'name': 'With LRU cache (500 images)',
#             'num_workers': 4,
#             'cache_size': 500,
#             'preload': False,
#             'prefetch_device': None
#         },
#         {
#             'name': 'Fully preloaded',
#             'num_workers': 4,
#             'cache_size': 0,
#             'preload': True,
#             'prefetch_device': None
#         },
#     ]
    
#     # Add GPU prefetching if CUDA is available
#     if torch.cuda.is_available():
#         configs.append({
#             'name': 'Full optimizations + GPU prefetch',
#             'num_workers': 4,
#             'cache_size': 500,
#             'preload': False,
#             'prefetch_device': device
#         })
    
#     # Run benchmarks
#     all_results = []
#     for config in configs:
#         try:
#             results = benchmark_dataloader(
#                 data_paths=data_paths,
#                 labels=labels,
#                 config_name=config['name'],
#                 batch_size=batch_size,
#                 num_workers=config['num_workers'],
#                 cache_size=config['cache_size'],
#                 preload=config['preload'],
#                 prefetch_device=config['prefetch_device'],
#                 num_epochs=2
#             )
#             all_results.append(results)
            
#             # Clear cache between runs
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#             time.sleep(2)
            
#         except Exception as e:
#             print(f"Error benchmarking {config['name']}: {e}")
    
#     # Cleanup
#     print(f"\nCleaning up temporary directory: {temp_dir}")
#     shutil.rmtree(temp_dir)
    
#     # Print comparison
#     print(f"\n{'='*80}")
#     print("BENCHMARK COMPARISON")
#     print(f"{'='*80}")
#     print(f"{'Configuration':<40} {'Throughput':<15} {'Memory':<15}")
#     print(f"{'':<40} {'(imgs/s)':<15} {'(MB)':<15}")
#     print(f"{'-'*80}")
    
#     for result in all_results:
#         print(f"{result['config_name']:<40} "
#               f"{result['throughput']:<15.1f} "
#               f"{result['memory_overhead_mb']:<15.1f}")
    
#     # Calculate speedup
#     if len(all_results) > 1:
#         baseline_throughput = all_results[0]['throughput']
#         print(f"\n{'='*80}")
#         print("SPEEDUP vs BASELINE")
#         print(f"{'='*80}")
#         for result in all_results[1:]:
#             speedup = result['throughput'] / baseline_throughput
#             print(f"{result['config_name']:<40} {speedup:.2f}x")
    
#     return all_results


# if __name__ == "__main__":
#     print("High-Performance DataLoader Benchmark")
#     print("=" * 80)
    
#     # Run benchmark with different dataset sizes
#     results = run_comprehensive_benchmark(num_images=1000, batch_size=32)
    
#     print("\n✓ Benchmark completed successfully!")
#     print("\nKey findings:")
#     print("- Multi-worker parallelism provides significant speedup")
#     print("- LRU caching reduces I/O overhead for repeated access")
#     print("- Preloading maximizes throughput but increases memory usage")
#     print("- GPU prefetching overlaps data transfer with computation")

"""
Benchmark script for High-Performance DataLoader
Measures throughput and memory usage across configurations
"""

import torch
import time
import psutil
import os
from pathlib import Path
import numpy as np
from PIL import Image
import tempfile
import shutil

from dataloader_module import create_dataloader


# ---------------------------------------------------------
# Synthetic Dataset Creation
# ---------------------------------------------------------
def create_synthetic_dataset(num_images: int, img_size: int = 224):
    temp_dir = tempfile.mkdtemp()
    paths, labels = [], []

    print(f"Creating {num_images} synthetic images...")
    for i in range(num_images):
        img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
        img = Image.fromarray(img)

        path = Path(temp_dir) / f"img_{i:05d}.jpg"
        img.save(path)

        paths.append(str(path))
        labels.append(i % 10)

        if (i + 1) % 100 == 0:
            print(f"Created {i + 1}/{num_images}")

    return paths, labels, temp_dir


# ---------------------------------------------------------
# Memory Utility
# ---------------------------------------------------------
def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


# ---------------------------------------------------------
# Benchmark Core
# ---------------------------------------------------------
def benchmark(
    name,
    data_paths,
    labels,
    batch_size,
    num_workers,
    cache_size,
    preload,
    prefetch_device=None,
    epochs=2,
):
    print("\n" + "=" * 70)
    print(f"Benchmarking: {name}")
    print("=" * 70)

    start_mem = get_memory_mb()

    loader = create_dataloader(
        data_paths=data_paths,
        labels=labels,
        batch_size=batch_size,
        num_workers=num_workers,
        cache_size=cache_size,
        preload=preload,
        augment=False,
        prefetch_device=prefetch_device,
    )

    total_images = 0
    epoch_times = []

    for epoch in range(epochs):
        epoch_start = time.time()

        for images, targets in loader:
            # Simulate minimal compute
            if torch.cuda.is_available() and prefetch_device is None:
                images = images.cuda(non_blocking=True)

            total_images += images.size(0)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    throughput = len(data_paths) / avg_epoch_time

    end_mem = get_memory_mb()

    print("\nResults:")
    print(f"Average epoch time : {avg_epoch_time:.2f}s")
    print(f"Throughput        : {throughput:.2f} images/sec")
    print(f"Memory overhead   : {end_mem - start_mem:.2f} MB")

    return {
        "name": name,
        "throughput": throughput,
        "memory": end_mem - start_mem,
    }



# ---------------------------------------------------------
# Run All Benchmarks
# ---------------------------------------------------------
def run_benchmark(num_images=1000, batch_size=32):
    data_paths, labels, temp_dir = create_synthetic_dataset(num_images)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    configs = [
        {
            "name": "Baseline (single worker)",
            "num_workers": 0,
            "cache_size": 0,
            "preload": False,
            "prefetch": None,
        },
        {
            "name": "Multi-worker (4 workers)",
            "num_workers": 4,
            "cache_size": 0,
            "preload": False,
            "prefetch": None,
        },
        {
            "name": "LRU Cache (500 images)",
            "num_workers": 4,
            "cache_size": 500,
            "preload": False,
            "prefetch": None,
        },
        {
            "name": "Fully Preloaded",
            "num_workers": 4,
            "cache_size": 0,
            "preload": True,
            "prefetch": None,
        },
    ]

    if torch.cuda.is_available():
        configs.append({
            "name": "GPU Prefetch Enabled",
            "num_workers": 4,
            "cache_size": 500,
            "preload": False,
            "prefetch": device,
        })

    results = []

    for cfg in configs:
        try:
            result = benchmark(
                name=cfg["name"],
                data_paths=data_paths,
                labels=labels,
                batch_size=batch_size,
                num_workers=cfg["num_workers"],
                cache_size=cfg["cache_size"],
                preload=cfg["preload"],
                prefetch_device=cfg["prefetch"],
            )
            results.append(result)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            time.sleep(1)

        except Exception as e:
            print(f"❌ Error in {cfg['name']}: {e}")

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':35} {'Throughput(img/s)':20} {'Memory(MB)':15}")
    print("-" * 80)

    for r in results:
        print(f"{r['name']:35} {r['throughput']:20.2f} {r['memory']:15.2f}")

    shutil.rmtree(temp_dir)
    print("\n✓ Benchmark completed successfully")


# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    run_benchmark(num_images=1000, batch_size=32)
