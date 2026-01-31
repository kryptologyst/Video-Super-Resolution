#!/usr/bin/env python3
"""
Quick demo script for Video Super-Resolution project.

This script demonstrates the key features of the video super-resolution implementation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import numpy as np
from models.vsr_models import create_model
from data.dataset import SyntheticDataset
from utils.device import get_device, set_seed
from utils.metrics import calculate_psnr, calculate_ssim


def main():
    """Run a quick demonstration of the video super-resolution project."""
    print("🎬 Video Super-Resolution Demo")
    print("=" * 50)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create a simple SRCNN model
    print("\nCreating SRCNN model...")
    model = create_model(
        model_name="srcnn",
        channels=3,
        scale_factor=2
    )
    model.to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Create synthetic test data
    print("\nGenerating synthetic test data...")
    dataset = SyntheticDataset(
        num_samples=1,
        scale_factor=2,
        num_frames=1,
        patch_size=32,
        pattern_type="moving_circle"
    )
    
    lr_frames, hr_frames = dataset[0]
    lr_frame = lr_frames[0:1].to(device)  # Single frame with batch dimension
    hr_frame = hr_frames[0:1].to(device)
    
    print(f"Input shape: {lr_frame.shape}")
    print(f"Target shape: {hr_frame.shape}")
    
    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        pred = model(lr_frame)
    
    print(f"Output shape: {pred.shape}")
    
    # Calculate metrics
    psnr = calculate_psnr(pred, hr_frame)
    ssim = calculate_ssim(pred, hr_frame)
    
    print(f"\nResults:")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")
    
    # Model comparison
    print("\nModel Comparison:")
    print("-" * 30)
    models_info = {
        "SRCNN": {"params": "57K", "speed": "Fast", "quality": "Good"},
        "VDSR": {"params": "665K", "speed": "Medium", "quality": "Better"},
        "EDVR": {"params": "20.6M", "speed": "Slow", "quality": "High"},
        "BasicVSR++": {"params": "6.3M", "speed": "Slow", "quality": "Best"}
    }
    
    for name, info in models_info.items():
        print(f"{name:<12} | {info['params']:<8} | {info['speed']:<8} | {info['quality']}")
    
    print("\n✅ Demo completed successfully!")
    print("\nTo run the full demo:")
    print("1. Training: python -m src.train.train --model srcnn --epochs 10")
    print("2. Evaluation: python -m src.eval.evaluate --checkpoint checkpoints/best.pth --model srcnn")
    print("3. Web Demo: streamlit run demo/streamlit_app.py")
    print("4. Jupyter: jupyter notebook notebooks/demo.ipynb")


if __name__ == "__main__":
    main()
