"""Loss functions for video super-resolution."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import Tensor


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss (L1 smooth).
    
    More robust to outliers than L2 loss.
    """
    
    def __init__(self, eps: float = 1e-3):
        """
        Initialize Charbonnier loss.
        
        Args:
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.eps = eps
        
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute Charbonnier loss.
        
        Args:
            pred: Predicted tensor.
            target: Target tensor.
            
        Returns:
            Loss value.
        """
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps))


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features.
    
    Measures high-level feature similarity rather than pixel-wise differences.
    """
    
    def __init__(
        self,
        feature_layers: Optional[list] = None,
        weights: Optional[list] = None,
    ):
        """
        Initialize perceptual loss.
        
        Args:
            feature_layers: List of VGG layer indices to use.
            weights: Weights for each layer.
        """
        super().__init__()
        
        if feature_layers is None:
            feature_layers = [4, 9, 18, 27, 36]  # VGG19 layers
        
        if weights is None:
            weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        
        self.feature_layers = feature_layers
        self.weights = weights
        
        # Load VGG19 features
        vgg = torchvision.models.vgg19(pretrained=True).features
        self.vgg_layers = nn.ModuleList()
        
        for i, layer in enumerate(vgg):
            if i in feature_layers:
                self.vgg_layers.append(layer)
        
        # Freeze VGG parameters
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
            
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute perceptual loss.
        
        Args:
            pred: Predicted tensor.
            target: Target tensor.
            
        Returns:
            Loss value.
        """
        pred_features = []
        target_features = []
        
        # Extract features
        for layer in self.vgg_layers:
            pred = layer(pred)
            target = layer(target)
            pred_features.append(pred)
            target_features.append(target)
        
        # Compute loss
        loss = 0.0
        for pred_feat, target_feat, weight in zip(pred_features, target_features, self.weights):
            loss += weight * F.mse_loss(pred_feat, target_feat)
        
        return loss


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency loss for video super-resolution.
    
    Encourages smooth temporal transitions in the output video.
    """
    
    def __init__(self, loss_fn: nn.Module = nn.L1Loss()):
        """
        Initialize temporal consistency loss.
        
        Args:
            loss_fn: Base loss function to use.
        """
        super().__init__()
        self.loss_fn = loss_fn
        
    def forward(self, pred_frames: Tensor, target_frames: Tensor) -> Tensor:
        """
        Compute temporal consistency loss.
        
        Args:
            pred_frames: Predicted frames of shape (B, T, C, H, W).
            target_frames: Target frames of shape (B, T, C, H, W).
            
        Returns:
            Loss value.
        """
        B, T, C, H, W = pred_frames.shape
        
        if T < 2:
            return torch.tensor(0.0, device=pred_frames.device)
        
        # Compute frame differences
        pred_diff = pred_frames[:, 1:] - pred_frames[:, :-1]
        target_diff = target_frames[:, 1:] - target_frames[:, :-1]
        
        return self.loss_fn(pred_diff, target_diff)


class CombinedLoss(nn.Module):
    """
    Combined loss function for video super-resolution.
    
    Combines multiple loss terms with configurable weights.
    """
    
    def __init__(
        self,
        charbonnier_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        temporal_weight: float = 0.1,
        use_perceptual: bool = True,
        use_temporal: bool = True,
    ):
        """
        Initialize combined loss.
        
        Args:
            charbonnier_weight: Weight for Charbonnier loss.
            perceptual_weight: Weight for perceptual loss.
            temporal_weight: Weight for temporal consistency loss.
            use_perceptual: Whether to use perceptual loss.
            use_temporal: Whether to use temporal consistency loss.
        """
        super().__init__()
        
        self.charbonnier_loss = CharbonnierLoss()
        self.charbonnier_weight = charbonnier_weight
        
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()
            self.perceptual_weight = perceptual_weight
        else:
            self.perceptual_loss = None
            self.perceptual_weight = 0.0
            
        if use_temporal:
            self.temporal_loss = TemporalConsistencyLoss()
            self.temporal_weight = temporal_weight
        else:
            self.temporal_loss = None
            self.temporal_weight = 0.0
    
    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        pred_frames: Optional[Tensor] = None,
        target_frames: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute combined loss.
        
        Args:
            pred: Predicted tensor.
            target: Target tensor.
            pred_frames: Predicted frames for temporal loss (B, T, C, H, W).
            target_frames: Target frames for temporal loss (B, T, C, H, W).
            
        Returns:
            Dictionary containing individual and total losses.
        """
        losses = {}
        
        # Charbonnier loss
        charbonnier_loss = self.charbonnier_loss(pred, target)
        losses["charbonnier"] = charbonnier_loss
        
        total_loss = self.charbonnier_weight * charbonnier_loss
        
        # Perceptual loss
        if self.perceptual_loss is not None:
            perceptual_loss = self.perceptual_loss(pred, target)
            losses["perceptual"] = perceptual_loss
            total_loss += self.perceptual_weight * perceptual_loss
        
        # Temporal consistency loss
        if self.temporal_loss is not None and pred_frames is not None and target_frames is not None:
            temporal_loss = self.temporal_loss(pred_frames, target_frames)
            losses["temporal"] = temporal_loss
            total_loss += self.temporal_weight * temporal_loss
        
        losses["total"] = total_loss
        return losses
