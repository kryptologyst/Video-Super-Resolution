"""Device utilities for automatic device detection and management."""

import logging
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Get the best available device for computation.
    
    Priority: CUDA -> MPS (Apple Silicon) -> CPU
    
    Args:
        device: Optional device specification. If None, auto-detect.
        
    Returns:
        torch.device: The selected device.
    """
    if device is not None:
        if isinstance(device, str):
            device = torch.device(device)
        return device
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Make deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def get_device_memory_info(device: torch.device) -> dict:
    """
    Get memory information for the specified device.
    
    Args:
        device: The device to get memory info for.
        
    Returns:
        dict: Memory information including total and allocated memory.
    """
    if device.type == "cuda":
        return {
            "total": torch.cuda.get_device_properties(device).total_memory,
            "allocated": torch.cuda.memory_allocated(device),
            "cached": torch.cuda.memory_reserved(device),
        }
    elif device.type == "mps":
        # MPS doesn't provide detailed memory info
        return {"total": None, "allocated": None, "cached": None}
    else:
        # CPU - no GPU memory
        return {"total": None, "allocated": None, "cached": None}
