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
