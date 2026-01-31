"""Tests for video super-resolution models."""

import pytest
import torch
import numpy as np

from src.models.vsr_models import SRCNN, VDSR, EDVR, BasicVSRPlusPlus, create_model
from src.models.losses import CharbonnierLoss, PerceptualLoss, TemporalConsistencyLoss, CombinedLoss
from src.data.dataset import SyntheticDataset
from src.utils.device import get_device, set_seed
from src.utils.metrics import calculate_psnr, calculate_ssim


class TestModels:
    """Test model implementations."""
    
    def test_srcnn_creation(self):
        """Test SRCNN model creation."""
        model = SRCNN(channels=3, scale_factor=2)
        assert isinstance(model, SRCNN)
        assert model.scale_factor == 2
    
    def test_srcnn_forward(self):
        """Test SRCNN forward pass."""
        model = SRCNN(channels=3, scale_factor=2)
        x = torch.randn(1, 3, 32, 32)
        output = model(x)
        assert output.shape == (1, 3, 64, 64)
    
    def test_vdsr_creation(self):
        """Test VDSR model creation."""
        model = VDSR(channels=3, scale_factor=2)
        assert isinstance(model, VDSR)
        assert model.scale_factor == 2
    
    def test_vdsr_forward(self):
        """Test VDSR forward pass."""
        model = VDSR(channels=3, scale_factor=2)
        x = torch.randn(1, 3, 32, 32)
        output = model(x)
        assert output.shape == (1, 3, 64, 64)
    
    def test_edvr_creation(self):
        """Test EDVR model creation."""
        model = EDVR(channels=3, scale_factor=2, num_frames=5)
        assert isinstance(model, EDVR)
        assert model.scale_factor == 2
        assert model.num_frames == 5
    
    def test_edvr_forward(self):
        """Test EDVR forward pass."""
        model = EDVR(channels=3, scale_factor=2, num_frames=5)
        x = torch.randn(1, 5, 3, 32, 32)
        output = model(x)
        assert output.shape == (1, 3, 64, 64)
    
    def test_basicvsr_plus_plus_creation(self):
        """Test BasicVSR++ model creation."""
        model = BasicVSRPlusPlus(channels=3, scale_factor=2, num_frames=5)
        assert isinstance(model, BasicVSRPlusPlus)
        assert model.scale_factor == 2
        assert model.num_frames == 5
    
    def test_basicvsr_plus_plus_forward(self):
        """Test BasicVSR++ forward pass."""
        model = BasicVSRPlusPlus(channels=3, scale_factor=2, num_frames=5)
        x = torch.randn(1, 5, 3, 32, 32)
        output = model(x)
        assert output.shape == (1, 3, 64, 64)
    
    def test_create_model(self):
        """Test model creation function."""
        models = ['srcnn', 'vdsr', 'edvr', 'basicvsr++']
        for model_name in models:
            model = create_model(model_name, channels=3, scale_factor=2)
            assert model is not None
    
    def test_create_model_invalid(self):
        """Test model creation with invalid name."""
        with pytest.raises(ValueError):
            create_model('invalid_model')


class TestLosses:
    """Test loss function implementations."""
    
    def test_charbonnier_loss(self):
        """Test Charbonnier loss."""
        loss_fn = CharbonnierLoss()
        pred = torch.randn(1, 3, 32, 32)
        target = torch.randn(1, 3, 32, 32)
        loss = loss_fn(pred, target)
        assert loss.item() >= 0
    
    def test_perceptual_loss(self):
        """Test perceptual loss."""
        loss_fn = PerceptualLoss()
        pred = torch.randn(1, 3, 64, 64)
        target = torch.randn(1, 3, 64, 64)
        loss = loss_fn(pred, target)
        assert loss.item() >= 0
    
    def test_temporal_consistency_loss(self):
        """Test temporal consistency loss."""
        loss_fn = TemporalConsistencyLoss()
        pred_frames = torch.randn(1, 5, 3, 32, 32)
        target_frames = torch.randn(1, 5, 3, 32, 32)
        loss = loss_fn(pred_frames, target_frames)
        assert loss.item() >= 0
    
    def test_combined_loss(self):
        """Test combined loss."""
        loss_fn = CombinedLoss()
        pred = torch.randn(1, 3, 32, 32)
        target = torch.randn(1, 3, 32, 32)
        losses = loss_fn(pred, target)
        assert 'total' in losses
        assert losses['total'].item() >= 0


class TestDataset:
    """Test dataset implementations."""
    
    def test_synthetic_dataset_creation(self):
        """Test synthetic dataset creation."""
        dataset = SyntheticDataset(num_samples=10, scale_factor=2)
        assert len(dataset) == 10
    
    def test_synthetic_dataset_getitem(self):
        """Test synthetic dataset item access."""
        dataset = SyntheticDataset(num_samples=10, scale_factor=2)
        lr_frames, hr_frames = dataset[0]
        assert lr_frames.shape[0] == 5  # num_frames
        assert hr_frames.shape[0] == 5  # num_frames
        assert lr_frames.shape[1] == 3  # channels
        assert hr_frames.shape[1] == 3  # channels
    
    def test_synthetic_dataset_patterns(self):
        """Test different synthetic patterns."""
        patterns = ['moving_circle', 'moving_square', 'random_pattern']
        for pattern in patterns:
            dataset = SyntheticDataset(
                num_samples=5, 
                scale_factor=2, 
                pattern_type=pattern
            )
            lr_frames, hr_frames = dataset[0]
            assert lr_frames.shape == hr_frames.shape


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_psnr_calculation(self):
        """Test PSNR calculation."""
        pred = torch.ones(1, 3, 32, 32)
        target = torch.ones(1, 3, 32, 32)
        psnr = calculate_psnr(pred, target)
        assert psnr == float('inf')  # Perfect match
    
    def test_psnr_calculation_noise(self):
        """Test PSNR calculation with noise."""
        pred = torch.randn(1, 3, 32, 32)
        target = torch.randn(1, 3, 32, 32)
        psnr = calculate_psnr(pred, target)
        assert isinstance(psnr, float)
        assert psnr >= 0
    
    def test_ssim_calculation(self):
        """Test SSIM calculation."""
        pred = torch.ones(1, 3, 32, 32)
        target = torch.ones(1, 3, 32, 32)
        ssim = calculate_ssim(pred, target)
        assert ssim == 1.0  # Perfect match
    
    def test_ssim_calculation_noise(self):
        """Test SSIM calculation with noise."""
        pred = torch.randn(1, 3, 32, 32)
        target = torch.randn(1, 3, 32, 32)
        ssim = calculate_ssim(pred, target)
        assert isinstance(ssim, float)
        assert 0 <= ssim <= 1


class TestDevice:
    """Test device utilities."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # This is a basic test - in practice, you'd test reproducibility
        assert True


if __name__ == '__main__':
    pytest.main([__file__])
