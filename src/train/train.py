"""Training script for video super-resolution models."""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.dataset import SyntheticDataset, create_dataloader
from src.models.losses import CombinedLoss
from src.models.vsr_models import create_model
from src.utils.device import get_device, set_seed
from src.utils.metrics import MetricsCalculator


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


class Trainer:
    """Trainer class for video super-resolution models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: torch.device = torch.device('cpu'),
        config: Optional[Dict] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            device: Device to run training on.
            config: Training configuration.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config or {}
        
        # Setup loss function
        self.criterion = CombinedLoss(
            charbonnier_weight=self.config.get('charbonnier_weight', 1.0),
            perceptual_weight=self.config.get('perceptual_weight', 0.1),
            temporal_weight=self.config.get('temporal_weight', 0.1),
            use_perceptual=self.config.get('use_perceptual', True),
            use_temporal=self.config.get('use_temporal', True),
        )
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 1e-4),
        )
        
        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.get('lr_step_size', 30),
            gamma=self.config.get('lr_gamma', 0.5),
        )
        
        # Setup metrics
        self.metrics_calculator = MetricsCalculator()
        
        # Training state
        self.epoch = 0
        self.best_psnr = 0.0
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {'total': 0.0, 'charbonnier': 0.0, 'perceptual': 0.0, 'temporal': 0.0}
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        for batch_idx, (lr_frames, hr_frames) in enumerate(pbar):
            lr_frames = lr_frames.to(self.device)
            hr_frames = hr_frames.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if lr_frames.dim() == 5:  # Multi-frame input
                B, T, C, H, W = lr_frames.shape
                lr_frames_flat = lr_frames.view(B * T, C, H, W)
                pred_flat = self.model(lr_frames_flat)
                pred = pred_flat.view(B, T, C, H * 2, W * 2)
                
                # Use center frame for loss calculation
                pred_center = pred[:, T//2]
                target_center = hr_frames[:, T//2]
                
                losses = self.criterion(pred_center, target_center, pred, hr_frames)
            else:  # Single-frame input
                pred = self.model(lr_frames)
                losses = self.criterion(pred, hr_frames)
            
            # Backward pass
            losses['total'].backward()
            self.optimizer.step()
            
            # Update metrics
            for key, value in losses.items():
                epoch_losses[key] += value.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{losses['total'].item():.4f}",
                'PSNR': f"{epoch_losses.get('psnr', 0):.2f}"
            })
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        val_losses = {'total': 0.0, 'charbonnier': 0.0, 'perceptual': 0.0, 'temporal': 0.0}
        metrics_calculator = MetricsCalculator()
        
        with torch.no_grad():
            for lr_frames, hr_frames in tqdm(self.val_loader, desc='Validation'):
                lr_frames = lr_frames.to(self.device)
                hr_frames = hr_frames.to(self.device)
                
                if lr_frames.dim() == 5:  # Multi-frame input
                    B, T, C, H, W = lr_frames.shape
                    lr_frames_flat = lr_frames.view(B * T, C, H, W)
                    pred_flat = self.model(lr_frames_flat)
                    pred = pred_flat.view(B, T, C, H * 2, W * 2)
                    
                    # Use center frame for loss calculation
                    pred_center = pred[:, T//2]
                    target_center = hr_frames[:, T//2]
                    
                    losses = self.criterion(pred_center, target_center, pred, hr_frames)
                    metrics_calculator.update(pred_center, target_center, pred)
                else:  # Single-frame input
                    pred = self.model(lr_frames)
                    losses = self.criterion(pred, hr_frames)
                    metrics_calculator.update(pred, hr_frames)
                
                # Accumulate losses
                for key, value in losses.items():
                    val_losses[key] += value.item()
        
        # Average losses
        num_batches = len(self.val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches
        
        # Compute metrics
        metrics = metrics_calculator.compute()
        val_losses.update(metrics)
        
        return val_losses
    
    def save_checkpoint(self, checkpoint_dir: Path, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_psnr': self.best_psnr,
            'config': self.config,
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, checkpoint_dir / 'latest.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'best.pth')
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_psnr = checkpoint.get('best_psnr', 0.0)
        
        return checkpoint
    
    def train(
        self,
        num_epochs: int,
        checkpoint_dir: Path,
        log_dir: Path,
        save_freq: int = 10,
    ):
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train.
            checkpoint_dir: Directory to save checkpoints.
            log_dir: Directory to save logs.
            save_freq: Frequency to save checkpoints.
        """
        writer = SummaryWriter(log_dir)
        logger = logging.getLogger(__name__)
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # Training
            train_losses = self.train_epoch()
            
            # Validation
            val_losses = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            for key, value in train_losses.items():
                writer.add_scalar(f'Train/{key}', value, epoch)
            
            for key, value in val_losses.items():
                writer.add_scalar(f'Val/{key}', value, epoch)
            
            writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save checkpoint
            is_best = val_losses.get('psnr', 0) > self.best_psnr
            if is_best:
                self.best_psnr = val_losses.get('psnr', 0)
            
            if epoch % save_freq == 0 or is_best:
                self.save_checkpoint(checkpoint_dir, is_best)
            
            # Log progress
            logger.info(
                f'Epoch {epoch}: Train Loss = {train_losses["total"]:.4f}, '
                f'Val PSNR = {val_losses.get("psnr", 0):.2f}, '
                f'Best PSNR = {self.best_psnr:.2f}'
            )
        
        writer.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train video super-resolution model')
    parser.add_argument('--config', type=str, default='configs/train.yaml',
                       help='Path to config file')
    parser.add_argument('--model', type=str, default='srcnn',
                       choices=['srcnn', 'vdsr', 'edvr', 'basicvsr++'],
                       help='Model architecture')
    parser.add_argument('--scale', type=int, default=2,
                       help='Upscaling factor')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, mps, cpu, auto)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Log directory')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup device
    if args.device == 'auto':
        device = get_device()
    else:
        device = torch.device(args.device)
    
    # Setup logging
    log_dir = Path(args.log_dir)
    logger = setup_logging(log_dir)
    
    logger.info(f'Starting training with device: {device}')
    logger.info(f'Model: {args.model}, Scale: {args.scale}, Epochs: {args.epochs}')
    
    # Create model
    model = create_model(
        model_name=args.model,
        channels=3,
        scale_factor=args.scale,
    )
    
    logger.info(f'Model created with {sum(p.numel() for p in model.parameters())} parameters')
    
    # Create datasets
    train_dataset = SyntheticDataset(
        num_samples=1000,
        scale_factor=args.scale,
        num_frames=5,
        patch_size=64,
    )
    
    val_dataset = SyntheticDataset(
        num_samples=200,
        scale_factor=args.scale,
        num_frames=5,
        patch_size=64,
        augment=False,
    )
    
    # Create data loaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )
    
    # Training configuration
    config = {
        'learning_rate': args.lr,
        'charbonnier_weight': 1.0,
        'perceptual_weight': 0.1,
        'temporal_weight': 0.1,
        'use_perceptual': True,
        'use_temporal': True,
        'weight_decay': 1e-4,
        'lr_step_size': 30,
        'lr_gamma': 0.5,
    }
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
    )
    
    # Start training
    checkpoint_dir = Path(args.checkpoint_dir)
    trainer.train(
        num_epochs=args.epochs,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        save_freq=10,
    )
    
    logger.info('Training completed!')


if __name__ == '__main__':
    main()
