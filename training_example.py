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

