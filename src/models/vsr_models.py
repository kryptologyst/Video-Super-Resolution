"""Video super-resolution models."""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SRCNN(nn.Module):
    """
    Super-Resolution Convolutional Neural Network (SRCNN).
    
    Original paper: "Image Super-Resolution Using Deep Convolutional Networks"
    https://arxiv.org/abs/1501.00092
    """
    
    def __init__(
        self,
        channels: int = 3,
        scale_factor: int = 2,
        kernel_size: Tuple[int, int, int] = (9, 1, 5),
    ):
        """
        Initialize SRCNN model.
        
        Args:
            channels: Number of input channels.
            scale_factor: Upscaling factor.
            kernel_size: Kernel sizes for the three convolutional layers.
        """
        super().__init__()
        self.scale_factor = scale_factor
        
        k1, k2, k3 = kernel_size
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=k1, padding=k1//2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=k2, padding=k2//2)
        self.conv3 = nn.Conv2d(32, channels, kernel_size=k3, padding=k3//2)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            Upscaled tensor of shape (B, C, H*scale, W*scale).
        """
        # Upsample first
        x = F.interpolate(
            x, 
            scale_factor=self.scale_factor, 
            mode='bicubic', 
            align_corners=False
        )
        
        # Apply SRCNN layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        
        return x


class ResidualBlock(nn.Module):
    """Residual block with batch normalization."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out + residual


class VDSR(nn.Module):
    """
    Very Deep Super-Resolution (VDSR) network.
    
    Original paper: "Accurate Image Super-Resolution Using Very Deep Convolutional Networks"
    https://arxiv.org/abs/1511.04587
    """
    
    def __init__(
        self,
        channels: int = 3,
        scale_factor: int = 2,
        num_layers: int = 20,
    ):
        """
        Initialize VDSR model.
        
        Args:
            channels: Number of input channels.
            scale_factor: Upscaling factor.
            num_layers: Number of convolutional layers.
        """
        super().__init__()
        self.scale_factor = scale_factor
        
        # First layer
        self.conv1 = nn.Conv2d(channels, 64, 3, padding=1)
        
        # Residual layers
        self.res_layers = nn.ModuleList([
            nn.Conv2d(64, 64, 3, padding=1) for _ in range(num_layers - 2)
        ])
        
        # Last layer
        self.conv_last = nn.Conv2d(64, channels, 3, padding=1)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            Upscaled tensor of shape (B, C, H*scale, W*scale).
        """
        # Upsample first
        x_up = F.interpolate(
            x, 
            scale_factor=self.scale_factor, 
            mode='bicubic', 
            align_corners=False
        )
        
        # Store for residual connection
        residual = x_up
        
        # First layer
        out = F.relu(self.conv1(x_up))
        
        # Residual layers
        for layer in self.res_layers:
            out = F.relu(layer(out))
        
        # Last layer
        out = self.conv_last(out)
        
        # Add residual
        return out + residual


class EDVRBlock(nn.Module):
    """EDVR temporal attention block."""
    
    def __init__(self, channels: int, num_frames: int = 5):
        super().__init__()
        self.num_frames = num_frames
        self.temporal_attn = nn.Sequential(
            nn.Conv2d(channels * num_frames, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C*T, H, W).
            
        Returns:
            Attended tensor of shape (B, C, H, W).
        """
        attn = self.temporal_attn(x)
        # Reshape and apply attention
        B, C_T, H, W = x.shape
        C = C_T // self.num_frames
        x_reshaped = x.view(B, self.num_frames, C, H, W)
        attn_reshaped = attn.view(B, 1, C, H, W)
        
        attended = (x_reshaped * attn_reshaped).sum(dim=1)
        return attended


class EDVR(nn.Module):
    """
    Enhanced Deformable Video Restoration (EDVR) network.
    
    Adapted for super-resolution from the original EDVR paper.
    """
    
    def __init__(
        self,
        channels: int = 3,
        scale_factor: int = 2,
        num_frames: int = 5,
        mid_channels: int = 64,
    ):
        """
        Initialize EDVR model.
        
        Args:
            channels: Number of input channels.
            scale_factor: Upscaling factor.
            num_frames: Number of input frames.
            mid_channels: Number of intermediate channels.
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.num_frames = num_frames
        
        # Feature extraction
        self.feat_extract = nn.Conv2d(channels, mid_channels, 3, padding=1)
        
        # Temporal attention
        self.temporal_attn = EDVRBlock(mid_channels, num_frames)
        
        # Reconstruction
        self.recon_layers = nn.Sequential(
            ResidualBlock(mid_channels),
            ResidualBlock(mid_channels),
            ResidualBlock(mid_channels),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
        )
        
        # Upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(mid_channels, channels, 3, padding=1),
        )
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, T, C, H, W).
            
        Returns:
            Upscaled tensor of shape (B, C, H*scale, W*scale).
        """
        B, T, C, H, W = x.shape
        
        # Extract features for each frame
        feat_frames = []
        for t in range(T):
            feat = self.feat_extract(x[:, t])
            feat_frames.append(feat)
        
        # Concatenate temporal features
        feat_concat = torch.cat(feat_frames, dim=1)  # (B, C*T, H, W)
        
        # Apply temporal attention
        feat_attended = self.temporal_attn(feat_concat)
        
        # Reconstruction
        feat_recon = self.recon_layers(feat_attended)
        
        # Upsample
        output = self.upsample(feat_recon)
        
        return output


class BasicVSRPlusPlus(nn.Module):
    """
    BasicVSR++ network for video super-resolution.
    
    Simplified version of the original BasicVSR++ architecture.
    """
    
    def __init__(
        self,
        channels: int = 3,
        scale_factor: int = 2,
        num_frames: int = 5,
        mid_channels: int = 64,
    ):
        """
        Initialize BasicVSR++ model.
        
        Args:
            channels: Number of input channels.
            scale_factor: Upscaling factor.
            num_frames: Number of input frames.
            mid_channels: Number of intermediate channels.
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.num_frames = num_frames
        
        # Feature extraction
        self.feat_extract = nn.Conv2d(channels, mid_channels, 3, padding=1)
        
        # Bidirectional propagation
        self.propagation = nn.ModuleList([
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
            for _ in range(num_frames - 1)
        ])
        
        # Reconstruction
        self.recon = nn.Sequential(
            ResidualBlock(mid_channels),
            ResidualBlock(mid_channels),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
        )
        
        # Upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(mid_channels, channels, 3, padding=1),
        )
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, T, C, H, W).
            
        Returns:
            Upscaled tensor of shape (B, C, H*scale, W*scale).
        """
        B, T, C, H, W = x.shape
        
        # Extract features
        feat_frames = []
        for t in range(T):
            feat = self.feat_extract(x[:, t])
            feat_frames.append(feat)
        
        # Bidirectional propagation
        propagated_feats = []
        for t in range(T):
            feat = feat_frames[t]
            
            # Forward propagation
            if t > 0:
                feat = feat + self.propagation[t-1](propagated_feats[t-1])
            
            propagated_feats.append(feat)
        
        # Backward propagation
        for t in range(T-2, -1, -1):
            propagated_feats[t] = propagated_feats[t] + self.propagation[t](propagated_feats[t+1])
        
        # Reconstruction and upsampling for center frame
        center_idx = T // 2
        feat_recon = self.recon(propagated_feats[center_idx])
        output = self.upsample(feat_recon)
        
        return output


def create_model(
    model_name: str,
    channels: int = 3,
    scale_factor: int = 2,
    **kwargs
) -> nn.Module:
    """
    Create a video super-resolution model.
    
    Args:
        model_name: Name of the model to create.
        channels: Number of input channels.
        scale_factor: Upscaling factor.
        **kwargs: Additional model-specific arguments.
        
    Returns:
        nn.Module: The created model.
        
    Raises:
        ValueError: If model_name is not supported.
    """
    models = {
        "srcnn": SRCNN,
        "vdsr": VDSR,
        "edvr": EDVR,
        "basicvsr++": BasicVSRPlusPlus,
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Unsupported model: {model_name}. Available: {list(models.keys())}")
    
    model_class = models[model_name.lower()]
    return model_class(channels=channels, scale_factor=scale_factor, **kwargs)
