"""Quick test to verify installation"""
import torch
print(f"✓ PyTorch {torch.__version__}")

from dataloader_module import create_dataloader
print("✓ DataLoader module imported")

import pytest
print("✓ pytest available")

import psutil
print("✓ psutil available")

print("\n✅ All dependencies installed correctly!")
print("\nRun tests with: pytest test_dataloader.py -v")
print("Run benchmarks with: python benchmark_script.py")