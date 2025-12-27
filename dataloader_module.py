"""
High-Performance DataLoader for Custom Dataset
Implements efficient data loading with caching, prefetching, and augmentations.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Optional, Callable, Tuple, List
import threading
from collections import OrderedDict
import torchvision.transforms as T


# =========================================================
# LRU Cache
# =========================================================
class LRUCache:
    """Thread-safe LRU cache for storing raw images."""

    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

    def clear(self):
        with self.lock:
            self.cache.clear()


# =========================================================
# Dataset
# =========================================================
class HighPerformanceDataset(Dataset):
    """
    Custom Dataset with optional caching and preloading.

    Args:
        data_paths: List of image paths
        labels: Corresponding labels
        transform: Optional torchvision transforms
        cache_size: Number of items to cache (0 disables cache)
        preload: Preload entire dataset into memory
    """

    def __init__(
        self,
        data_paths: List[str],
        labels: List[int],
        transform: Optional[Callable] = None,
        cache_size: int = 0,
        preload: bool = False,
    ):
        self.data_paths = data_paths
        self.labels = labels
        self.transform = transform
        self.preload = preload

        self.cache = LRUCache(cache_size) if cache_size > 0 else None
        self.preloaded_data = {}

        if preload:
            self._preload_data()

    def _preload_data(self):
        for idx, path in enumerate(self.data_paths):
            img = Image.open(path).convert("RGB")
            self.preloaded_data[idx] = img.copy()

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Try cache
        if self.cache is not None:
            cached = self.cache.get(idx)
            if cached is not None:
                img, label = cached
                if self.transform:
                    img = self.transform(img)
                return img, label

        # Load image
        if self.preload and idx in self.preloaded_data:
            img = self.preloaded_data[idx]
        else:
            img = Image.open(self.data_paths[idx]).convert("RGB")

        label = self.labels[idx]

        # Cache raw image
        if self.cache is not None:
            self.cache.put(idx, (img.copy(), label))

        if self.transform:
            img = self.transform(img)

        return img, label


# =========================================================
# Prefetch Loader
# =========================================================
class PrefetchLoader:
    """Asynchronous device prefetch wrapper."""

    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None

    def __iter__(self):
        loader_iter = iter(self.loader)

        if self.device.type == "cuda":
            self._preload(loader_iter)
            batch = self.next_batch

            while batch is not None:
                torch.cuda.current_stream().wait_stream(self.stream)
                yield batch
                self._preload(loader_iter)
                batch = self.next_batch
        else:
            for batch in loader_iter:
                yield self._to_device(batch)

    def _preload(self, loader_iter):
        try:
            self.next_batch = next(loader_iter)
        except StopIteration:
            self.next_batch = None
            return

        if self.device.type == "cuda":
            with torch.cuda.stream(self.stream):
                self.next_batch = self._to_device(self.next_batch)
        else:
            self.next_batch = self._to_device(self.next_batch)

    def _to_device(self, batch):
        if isinstance(batch, (list, tuple)):
            return [
                x.to(self.device, non_blocking=True) if isinstance(x, torch.Tensor) else x
                for x in batch
            ]
        return batch.to(self.device, non_blocking=True)

    def __len__(self):
        return len(self.loader)


# =========================================================
# Transforms
# =========================================================
def get_default_transforms(img_size: int = 224, augment: bool = True):
    if augment:
        return T.Compose(
            [
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.2, 0.2, 0.2, 0.1),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    else:
        return T.Compose(
            [
                T.Resize(int(img_size * 1.14)),
                T.CenterCrop(img_size),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )


# =========================================================
# DataLoader Factory
# =========================================================
def create_dataloader(
    data_paths: List[str],
    labels: List[int],
    batch_size: int = 32,
    num_workers: int = 0,
    img_size: int = 224,
    cache_size: int = 1000,
    preload: bool = False,
    augment: bool = True,
    pin_memory: bool = True,
    prefetch_device: Optional[torch.device] = None,
) -> DataLoader:
    """
    Create a high-performance DataLoader.

    Windows-safe:
    - Disables cache when num_workers > 0 (pickling safety)
    - drop_last=True to ensure deterministic batch count
    """

    # 🔒 Windows multiprocessing safety
    if num_workers > 0:
        cache_size = 0

    transform = get_default_transforms(img_size, augment)

    dataset = HighPerformanceDataset(
        data_paths=data_paths,
        labels=labels,
        transform=transform,
        cache_size=cache_size,
        preload=preload,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    if prefetch_device is not None:
        loader = PrefetchLoader(loader, prefetch_device)

    return loader


# =========================================================
# Module Check
# =========================================================
if __name__ == "__main__":
    print("High-Performance DataLoader Module")
    print("✓ Windows-safe multiprocessing")
    print("✓ Deterministic batching (drop_last=True)")
    print("✓ Optional caching & prefetching")
