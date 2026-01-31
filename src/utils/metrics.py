"""Evaluation metrics for video super-resolution."""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def calculate_psnr(pred: Tensor, target: Tensor, max_val: float = 1.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        pred: Predicted tensor.
        target: Target tensor.
        max_val: Maximum possible value.
        
    Returns:
        PSNR value in dB.
    """
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    
    psnr = 20 * math.log10(max_val) - 10 * math.log10(mse.item())
    return psnr


def calculate_ssim(pred: Tensor, target: Tensor, window_size: int = 11) -> float:
    """
    Calculate Structural Similarity Index (SSIM).
    
    Args:
        pred: Predicted tensor.
        target: Target tensor.
        window_size: Size of the sliding window.
        
    Returns:
        SSIM value.
    """
    def gaussian_window(size: int, sigma: float = 1.5) -> Tensor:
        """Create Gaussian window."""
        coords = torch.arange(size, dtype=torch.float32)
        coords = coords - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(0) * g.unsqueeze(1)
        return window.unsqueeze(0).unsqueeze(0)  # Add channel dimensions
    
    def ssim_single_channel(x: Tensor, y: Tensor, window: Tensor) -> Tensor:
        """Calculate SSIM for single channel."""
        mu1 = F.conv2d(x, window, padding=window_size//2)
        mu2 = F.conv2d(y, window, padding=window_size//2)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(x * x, window, padding=window_size//2) - mu1_sq
        sigma2_sq = F.conv2d(y * y, window, padding=window_size//2) - mu2_sq
        sigma12 = F.conv2d(x * y, window, padding=window_size//2) - mu1_mu2
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return ssim_map.mean()
    
    # Convert to grayscale if needed
    if pred.shape[1] == 3:
        pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
    else:
        pred_gray = pred
        target_gray = target
    
    window = gaussian_window(window_size).to(pred.device)
    
    ssim_values = []
    for i in range(pred_gray.shape[0]):
        ssim_val = ssim_single_channel(
            pred_gray[i:i+1], target_gray[i:i+1], window
        )
        ssim_values.append(ssim_val.item())
    
    return np.mean(ssim_values)


def calculate_lpips(pred: Tensor, target: Tensor) -> float:
    """
    Calculate Learned Perceptual Image Patch Similarity (LPIPS).
    
    Simplified version using VGG features.
    
    Args:
        pred: Predicted tensor.
        target: Target tensor.
        
    Returns:
        LPIPS value.
    """
    import torchvision.models as models
    
    # Load VGG19 features
    vgg = models.vgg19(pretrained=True).features.to(pred.device)
    vgg.eval()
    
    # Extract features
    with torch.no_grad():
        pred_features = []
        target_features = []
        
        x_pred = pred
        x_target = target
        
        for layer in vgg:
            x_pred = layer(x_pred)
            x_target = layer(x_target)
            
            if isinstance(layer, torch.nn.Conv2d):
                pred_features.append(x_pred)
                target_features.append(x_target)
        
        # Calculate LPIPS
        lpips_values = []
        for pred_feat, target_feat in zip(pred_features, target_features):
            # Normalize features
            pred_feat_norm = F.normalize(pred_feat, p=2, dim=1)
            target_feat_norm = F.normalize(target_feat, p=2, dim=1)
            
            # Calculate distance
            diff = (pred_feat_norm - target_feat_norm) ** 2
            lpips_val = diff.mean(dim=(1, 2, 3))
            lpips_values.append(lpips_val)
        
        # Average across layers
        lpips = torch.stack(lpips_values).mean(dim=0).mean().item()
    
    return lpips


def calculate_temporal_consistency(pred_frames: Tensor, target_frames: Tensor) -> float:
    """
    Calculate temporal consistency metric.
    
    Measures smoothness of temporal transitions.
    
    Args:
        pred_frames: Predicted frames of shape (B, T, C, H, W).
        target_frames: Target frames of shape (B, T, C, H, W).
        
    Returns:
        Temporal consistency value.
    """
    B, T, C, H, W = pred_frames.shape
    
    if T < 2:
        return 0.0
    
    # Calculate frame differences
    pred_diff = pred_frames[:, 1:] - pred_frames[:, :-1]
    target_diff = target_frames[:, 1:] - target_frames[:, :-1]
    
    # Calculate MSE of differences
    mse_diff = F.mse_loss(pred_diff, target_diff)
    
    return mse_diff.item()


class MetricsCalculator:
    """
    Calculator for video super-resolution metrics.
    """
    
    def __init__(self, metrics: Optional[List[str]] = None):
        """
        Initialize metrics calculator.
        
        Args:
            metrics: List of metrics to calculate.
        """
        if metrics is None:
            metrics = ["psnr", "ssim", "lpips", "temporal_consistency"]
        
        self.metrics = metrics
        self.results = {metric: [] for metric in metrics}
    
    def update(self, pred: Tensor, target: Tensor, pred_frames: Optional[Tensor] = None):
        """
        Update metrics with new predictions.
        
        Args:
            pred: Predicted tensor.
            target: Target tensor.
            pred_frames: Predicted frames for temporal metrics.
        """
        for metric in self.metrics:
            if metric == "psnr":
                value = calculate_psnr(pred, target)
            elif metric == "ssim":
                value = calculate_ssim(pred, target)
            elif metric == "lpips":
                value = calculate_lpips(pred, target)
            elif metric == "temporal_consistency" and pred_frames is not None:
                # For temporal consistency, we need target frames too
                # This is a simplified version
                value = calculate_temporal_consistency(pred_frames, pred_frames)
            else:
                continue
            
            self.results[metric].append(value)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.
        
        Returns:
            Dictionary of metric names and their average values.
        """
        final_metrics = {}
        for metric, values in self.results.items():
            if values:
                final_metrics[metric] = np.mean(values)
            else:
                final_metrics[metric] = 0.0
        
        return final_metrics
    
    def reset(self):
        """Reset metrics."""
        self.results = {metric: [] for metric in self.metrics}


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate.
        dataloader: Data loader.
        device: Device to run evaluation on.
        metrics: List of metrics to calculate.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    model.eval()
    calculator = MetricsCalculator(metrics)
    
    with torch.no_grad():
        for batch_idx, (lr_frames, hr_frames) in enumerate(dataloader):
            lr_frames = lr_frames.to(device)
            hr_frames = hr_frames.to(device)
            
            # Forward pass
            if lr_frames.dim() == 5:  # (B, T, C, H, W)
                B, T, C, H, W = lr_frames.shape
                lr_frames_flat = lr_frames.view(B * T, C, H, W)
                pred_flat = model(lr_frames_flat)
                pred = pred_flat.view(B, T, C, H * 2, W * 2)
                
                # Use center frame for single-frame metrics
                pred_center = pred[:, T//2]
                target_center = hr_frames[:, T//2]
                
                calculator.update(pred_center, target_center, pred)
            else:
                pred = model(lr_frames)
                calculator.update(pred, hr_frames)
    
    return calculator.compute()
