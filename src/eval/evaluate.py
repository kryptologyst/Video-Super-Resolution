"""Evaluation script for video super-resolution models."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import SyntheticDataset, create_dataloader
from src.models.vsr_models import create_model
from src.utils.device import get_device, set_seed
from src.utils.metrics import evaluate_model


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    return logging.getLogger(__name__)


def load_model(
    checkpoint_path: Path,
    model_name: str,
    channels: int = 3,
    scale_factor: int = 2,
    device: torch.device = torch.device('cpu'),
) -> nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        model_name: Name of the model architecture.
        channels: Number of input channels.
        scale_factor: Upscaling factor.
        device: Device to load model on.
        
    Returns:
        Loaded model.
    """
    # Create model
    model = create_model(
        model_name=model_name,
        channels=channels,
        scale_factor=scale_factor,
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


def evaluate_checkpoint(
    checkpoint_path: Path,
    model_name: str,
    scale_factor: int = 2,
    device: torch.device = torch.device('cpu'),
    num_samples: int = 200,
    batch_size: int = 4,
) -> Dict[str, float]:
    """
    Evaluate a model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        model_name: Name of the model architecture.
        scale_factor: Upscaling factor.
        device: Device to run evaluation on.
        num_samples: Number of samples to evaluate.
        batch_size: Batch size for evaluation.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    logger = logging.getLogger(__name__)
    
    # Load model
    logger.info(f'Loading model from {checkpoint_path}')
    model = load_model(checkpoint_path, model_name, scale_factor=scale_factor, device=device)
    
    # Create test dataset
    test_dataset = SyntheticDataset(
        num_samples=num_samples,
        scale_factor=scale_factor,
        num_frames=5,
        patch_size=64,
        augment=False,
    )
    
    # Create data loader
    test_loader = create_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    
    # Evaluate model
    logger.info('Starting evaluation...')
    metrics = evaluate_model(model, test_loader, device)
    
    return metrics


def compare_models(
    checkpoints: Dict[str, Path],
    model_configs: Dict[str, Dict],
    scale_factor: int = 2,
    device: torch.device = torch.device('cpu'),
    num_samples: int = 200,
    batch_size: int = 4,
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models.
    
    Args:
        checkpoints: Dictionary mapping model names to checkpoint paths.
        model_configs: Dictionary mapping model names to configurations.
        scale_factor: Upscaling factor.
        device: Device to run evaluation on.
        num_samples: Number of samples to evaluate.
        batch_size: Batch size for evaluation.
        
    Returns:
        Dictionary mapping model names to their metrics.
    """
    logger = logging.getLogger(__name__)
    results = {}
    
    for model_name, checkpoint_path in checkpoints.items():
        logger.info(f'Evaluating {model_name}...')
        
        config = model_configs.get(model_name, {})
        metrics = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            scale_factor=scale_factor,
            device=device,
            num_samples=num_samples,
            batch_size=batch_size,
        )
        
        results[model_name] = metrics
        
        # Log results
        logger.info(f'{model_name} Results:')
        for metric, value in metrics.items():
            logger.info(f'  {metric}: {value:.4f}')
    
    return results


def print_results_table(results: Dict[str, Dict[str, float]]):
    """Print results in a formatted table."""
    if not results:
        print("No results to display.")
        return
    
    # Get all metrics
    all_metrics = set()
    for model_results in results.values():
        all_metrics.update(model_results.keys())
    
    all_metrics = sorted(list(all_metrics))
    
    # Print header
    print(f"{'Model':<15}", end="")
    for metric in all_metrics:
        print(f"{metric:<12}", end="")
    print()
    
    # Print separator
    print("-" * (15 + 12 * len(all_metrics)))
    
    # Print results
    for model_name, metrics in results.items():
        print(f"{model_name:<15}", end="")
        for metric in all_metrics:
            value = metrics.get(metric, 0.0)
            print(f"{value:<12.4f}", end="")
        print()


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate video super-resolution model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--model', type=str, default='srcnn',
                       choices=['srcnn', 'vdsr', 'edvr', 'basicvsr++'],
                       help='Model architecture')
    parser.add_argument('--scale', type=int, default=2,
                       help='Upscaling factor')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, mps, cpu, auto)')
    parser.add_argument('--num-samples', type=int, default=200,
                       help='Number of samples to evaluate')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for evaluation')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple models')
    parser.add_argument('--checkpoints-dir', type=str, default='checkpoints',
                       help='Directory containing checkpoints for comparison')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Setup device
    if args.device == 'auto':
        device = get_device()
    else:
        device = torch.device(args.device)
    
    logger.info(f'Using device: {device}')
    
    if args.compare:
        # Compare multiple models
        checkpoints_dir = Path(args.checkpoints_dir)
        
        # Find all checkpoint files
        checkpoint_files = list(checkpoints_dir.glob('*.pth'))
        
        if not checkpoint_files:
            logger.error(f'No checkpoint files found in {checkpoints_dir}')
            return
        
        # Create checkpoints dictionary
        checkpoints = {}
        model_configs = {}
        
        for checkpoint_file in checkpoint_files:
            model_name = checkpoint_file.stem
            checkpoints[model_name] = checkpoint_file
            model_configs[model_name] = {'scale_factor': args.scale}
        
        # Compare models
        results = compare_models(
            checkpoints=checkpoints,
            model_configs=model_configs,
            scale_factor=args.scale,
            device=device,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
        )
        
        # Print comparison table
        print("\nModel Comparison Results:")
        print("=" * 50)
        print_results_table(results)
        
    else:
        # Evaluate single model
        checkpoint_path = Path(args.checkpoint)
        
        if not checkpoint_path.exists():
            logger.error(f'Checkpoint file not found: {checkpoint_path}')
            return
        
        # Evaluate model
        metrics = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            model_name=args.model,
            scale_factor=args.scale,
            device=device,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
        )
        
        # Print results
        print(f"\nEvaluation Results for {args.model}:")
        print("=" * 40)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")


if __name__ == '__main__':
    main()
