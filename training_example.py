# """
# Example training loop using the high-performance DataLoader.
# Demonstrates integration with a real training scenario.
# """

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import random_split
# import time
# from pathlib import Path
# from typing import List, Tuple
# from dataloader_module import create_dataloader


# class SimpleConvNet(nn.Module):
#     """Simple CNN for demonstration purposes."""
    
#     def __init__(self, num_classes: int = 10):
#         super(SimpleConvNet, self).__init__()
        
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
            
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
            
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
        
#         self.classifier = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(),
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(128, num_classes)
#         )
    
#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x


# def train_epoch(model, loader, criterion, optimizer, device, epoch):
#     """Train for one epoch."""
#     model.train()
    
#     running_loss = 0.0
#     correct = 0
#     total = 0
#     batch_times = []
    
#     epoch_start = time.time()
    
#     for batch_idx, (images, targets) in enumerate(loader):
#         batch_start = time.time()
        
#         # Move to device if not already there (PrefetchLoader handles this)
#         if not isinstance(images, torch.Tensor) or images.device != device:
#             images = images.to(device, non_blocking=True)
#             targets = targets.to(device, non_blocking=True)
        
#         # Forward pass
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, targets)
        
#         # Backward pass
#         loss.backward()
#         optimizer.step()
        
#         # Statistics
#         running_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().item()
        
#         batch_time = time.time() - batch_start
#         batch_times.append(batch_time)
        
#         if batch_idx % 10 == 0:
#             print(f'Epoch: {epoch} | Batch: {batch_idx}/{len(loader)} | '
#                   f'Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}% | '
#                   f'Batch time: {batch_time:.3f}s | '
#                   f'Throughput: {images.size(0)/batch_time:.1f} imgs/s')
    
#     epoch_time = time.time() - epoch_start
#     avg_loss = running_loss / len(loader)
#     accuracy = 100. * correct / total
#     throughput = total / epoch_time
    
#     print(f'\nEpoch {epoch} Summary:')
#     print(f'Time: {epoch_time:.2f}s | Loss: {avg_loss:.4f} | '
#           f'Acc: {accuracy:.2f}% | Throughput: {throughput:.1f} imgs/s')
    
#     return avg_loss, accuracy, throughput


# def validate(model, loader, criterion, device):
#     """Validate the model."""
#     model.eval()
    
#     running_loss = 0.0
#     correct = 0
#     total = 0
    
#     with torch.no_grad():
#         for images, targets in loader:
#             images = images.to(device, non_blocking=True)
#             targets = targets.to(device, non_blocking=True)
            
#             outputs = model(images)
#             loss = criterion(outputs, targets)
            
#             running_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
    
#     avg_loss = running_loss / len(loader)
#     accuracy = 100. * correct / total
    
#     print(f'Validation Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%')
    
#     return avg_loss, accuracy


# def train_model(
#     train_paths: List[str],
#     train_labels: List[int],
#     val_paths: List[str],
#     val_labels: List[int],
#     num_epochs: int = 10,
#     batch_size: int = 32,
#     learning_rate: float = 0.001,
#     num_workers: int = 4,
#     use_cache: bool = True,
#     use_prefetch: bool = True
# ):
#     """
#     Complete training pipeline using high-performance DataLoader.
    
#     Args:
#         train_paths: Training image paths
#         train_labels: Training labels
#         val_paths: Validation image paths
#         val_labels: Validation labels
#         num_epochs: Number of training epochs
#         batch_size: Batch size
#         learning_rate: Learning rate
#         num_workers: Number of DataLoader workers
#         use_cache: Whether to use LRU cache
#         use_prefetch: Whether to use GPU prefetching
#     """
    
#     # Device setup
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f'Using device: {device}')
    
#     # Create DataLoaders with optimizations
#     print('\nCreating DataLoaders...')
    
#     cache_size = 1000 if use_cache else 0
#     prefetch_device = device if (use_prefetch and torch.cuda.is_available()) else None
    
#     train_loader = create_dataloader(
#         data_paths=train_paths,
#         labels=train_labels,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         cache_size=cache_size,
#         augment=True,
#         prefetch_device=prefetch_device
#     )
    
#     val_loader = create_dataloader(
#         data_paths=val_paths,
#         labels=val_labels,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         cache_size=cache_size // 2,
#         augment=False,
#         prefetch_device=prefetch_device
#     )
    
#     print(f'Train batches: {len(train_loader)}')
#     print(f'Val batches: {len(val_loader)}')
    
#     # Create model
#     num_classes = len(set(train_labels))
#     model = SimpleConvNet(num_classes=num_classes).to(device)
    
#     # Loss and optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
#     # Training loop
#     print('\n' + '='*80)
#     print('Starting Training')
#     print('='*80)
    
#     best_val_acc = 0.0
#     training_history = {
#         'train_loss': [],
#         'train_acc': [],
#         'val_loss': [],
#         'val_acc': [],
#         'throughput': []
#     }
    
#     total_start = time.time()
    
#     for epoch in range(1, num_epochs + 1):
#         print(f'\n--- Epoch {epoch}/{num_epochs} ---')
#         print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
#         # Train
#         train_loss, train_acc, throughput = train_epoch(
#             model, train_loader, criterion, optimizer, device, epoch
#         )
        
#         # Validate
#         print('\nValidating...')
#         val_loss, val_acc = validate(model, val_loader, criterion, device)
        
#         # Update learning rate
#         scheduler.step()
        
#         # Save history
#         training_history['train_loss'].append(train_loss)
#         training_history['train_acc'].append(train_acc)
#         training_history['val_loss'].append(val_loss)
#         training_history['val_acc'].append(val_acc)
#         training_history['throughput'].append(throughput)
        
#         # Save best model
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save(model.state_dict(), 'best_model.pth')
#             print(f'✓ Saved best model (val_acc: {val_acc:.2f}%)')
        
#         print(f'\nBest validation accuracy so far: {best_val_acc:.2f}%')
    
#     total_time = time.time() - total_start
    
#     print('\n' + '='*80)
#     print('Training Complete!')
#     print('='*80)
#     print(f'Total training time: {total_time/60:.2f} minutes')
#     print(f'Best validation accuracy: {best_val_acc:.2f}%')
#     print(f'Average throughput: {sum(training_history["throughput"])/len(training_history["throughput"]):.1f} imgs/s')
    
#     return model, training_history


# if __name__ == '__main__':
#     print("High-Performance DataLoader - Training Example")
#     print("=" * 80)
#     print("\nThis script demonstrates how to use the high-performance DataLoader")
#     print("in a real training scenario.\n")
    
#     # Example: Create dummy data paths and labels
#     print("To use this script with your data:")
#     print("1. Prepare lists of image paths and corresponding labels")
#     print("2. Split into train/validation sets")
#     print("3. Call train_model() with your data\n")
    
#     print("Example usage:")
#     print("-" * 80)
#     print("""
# from glob import glob

# # Load your dataset
# train_paths = glob('data/train/**/*.jpg', recursive=True)
# train_labels = [...]  # Your labels

# val_paths = glob('data/val/**/*.jpg', recursive=True)
# val_labels = [...]  # Your labels

# # Train with all optimizations
# model, history = train_model(
#     train_paths=train_paths,
#     train_labels=train_labels,
#     val_paths=val_paths,
#     val_labels=val_labels,
#     num_epochs=20,
#     batch_size=64,
#     learning_rate=0.001,
#     num_workers=4,
#     use_cache=True,
#     use_prefetch=True
# )
#     """)
#     print("-" * 80)

"""
High-Performance DataLoader - Benchmark + Training Example
Fully self-contained using synthetic dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import shutil
import psutil
import os
import csv
from dataloader_module import create_dataloader

# ---------------- Synthetic Dataset Generator ---------------- #
def create_synthetic_dataset(num_images: int = 100, img_size: int = 64, num_classes: int = 5):
    temp_dir = tempfile.mkdtemp()
    image_paths = []
    labels = []
    for i in range(num_images):
        img_array = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = Path(temp_dir) / f"img_{i:03d}.jpg"
        img.save(img_path)
        image_paths.append(str(img_path))
        labels.append(i % num_classes)
    return image_paths, labels, temp_dir

# ---------------- Simple CNN Model ---------------- #
class SimpleConvNet(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2,2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ---------------- Training / Validation ---------------- #
def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    batch_times = []
    epoch_start = time.time()

    for batch_idx, (images, targets) in enumerate(loader):
        batch_start = time.time()
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(loader)} | "
                  f"Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}% | "
                  f"Batch time: {batch_time:.3f}s | Throughput: {images.size(0)/batch_time:.1f} imgs/s")

    epoch_time = time.time() - epoch_start
    avg_loss = running_loss / len(loader)
    accuracy = 100.*correct/total
    throughput = total / epoch_time
    avg_batch_time = sum(batch_times)/len(batch_times)
    return avg_loss, accuracy, throughput, avg_batch_time

def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return running_loss / len(loader), 100.*correct/total

# ---------------- Benchmark ---------------- #
def run_benchmark(image_paths, labels, batch_size=32, num_workers=2, cache_size=50):
    print("\n" + "="*80)
    print("Running DataLoader Benchmark")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    process = psutil.Process(os.getpid())
    
    loader_configs = [
        ("Baseline (single worker)", 0, False, False),
        ("Multi-worker (4 workers)", 4, False, False),
        ("LRU Cache", 0, True, False),
        ("Fully Preloaded", 0, True, True)
    ]
    
    summary = []

    for name, workers, use_cache, preload in loader_configs:
        print(f"\n--- {name} ---")
        loader = create_dataloader(
            data_paths=image_paths,
            labels=labels,
            batch_size=batch_size,
            num_workers=workers,
            cache_size=cache_size if use_cache else 0,
            augment=False,
            preload=preload,
            prefetch_device=None
        )
        start_mem = process.memory_info().rss / (1024 ** 2)  # MB
        start_time = time.time()
        total_images = 0
        for batch_images, _ in loader:
            total_images += batch_images.size(0)
        elapsed = time.time() - start_time
        end_mem = process.memory_info().rss / (1024 ** 2)
        mem_used = end_mem - start_mem

        throughput = total_images / elapsed
        print(f"Processed {total_images} images in {elapsed:.2f}s | Throughput: {throughput:.1f} imgs/s | Memory used: {mem_used:.1f} MB")
        summary.append((name, throughput, mem_used))

    # Summary table
    print("\n" + "="*80)
    print("DataLoader Benchmark Summary (images/s & memory)")
    print("="*80)
    print(f"{'Configuration':30} {'Throughput':>12} {'Memory(MB)':>12}")
    print("-"*60)
    for name, throughput, mem_used in summary:
        print(f"{name:30} {throughput:12.1f} {mem_used:12.1f}")
    print("="*80)

# ---------------- Complete Training Pipeline ---------------- #
def train_model_pipeline():
    train_paths, train_labels, train_dir = create_synthetic_dataset(num_images=80)
    val_paths, val_labels, val_dir = create_synthetic_dataset(num_images=20)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    train_loader = create_dataloader(
        data_paths=train_paths, labels=train_labels,
        batch_size=16, num_workers=2,
        cache_size=50, augment=True, prefetch_device=device
    )
    val_loader = create_dataloader(
        data_paths=val_paths, labels=val_labels,
        batch_size=16, num_workers=2,
        cache_size=25, augment=False, prefetch_device=device
    )

    model = SimpleConvNet(num_classes=len(set(train_labels))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_acc = 0.0
    training_log = []
    for epoch in range(1, 4):
        print(f"\n--- Epoch {epoch}/3 ---")
        train_loss, train_acc, throughput, avg_batch_time = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Throughput: {throughput:.1f} imgs/s | Avg Batch: {avg_batch_time:.3f}s")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        training_log.append([epoch, train_loss, train_acc, val_loss, val_acc, throughput, avg_batch_time])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✓ Saved best model (val_acc: {val_acc:.2f}%)")

    # Save training log
    with open('training_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 'Throughput', 'Avg Batch Time'])
        writer.writerows(training_log)

    shutil.rmtree(train_dir)
    shutil.rmtree(val_dir)
    print(f"\nTraining Complete! Best validation accuracy: {best_val_acc:.2f}%")
    print("Training log saved to training_log.csv")

# ---------------- Main ---------------- #
if __name__ == "__main__":
    print("High-Performance DataLoader - Benchmark + Training Example")

    # Create synthetic dataset for benchmark
    img_paths, labels, temp_dir = create_synthetic_dataset(num_images=100)

    # Run benchmark
    run_benchmark(img_paths, labels)

    # Run training
    train_model_pipeline()

    # Cleanup benchmark dataset
    shutil.rmtree(temp_dir)

