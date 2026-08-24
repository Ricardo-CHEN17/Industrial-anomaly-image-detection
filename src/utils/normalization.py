# src/utils/normalization.py
"""Min-max normalization utilities for anomaly scores.

This module provides utilities for min-max normalization of anomaly scores.
The main function :func:`normalize` scales values to [0,1] range and centers them
around a threshold. It supports both NumPy arrays and PyTorch tensors.
"""

import numpy as np
import torch


def normalize(
    targets: np.ndarray | np.float32 | torch.Tensor,
    threshold: float | np.ndarray | torch.Tensor,
    min_val: float | np.ndarray | torch.Tensor,
    max_val: float | np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Apply min-max normalization and center values around a threshold.

    This function performs min-max normalization on the input values and shifts them
    such that the threshold value is centered at 0.5. The output is clipped to the
    range [0,1].

    Args:
        targets: Input values to normalize (NumPy array or PyTorch tensor).
        threshold: Threshold value that will be centered at 0.5 after normalization.
        min_val: Minimum value used for normalization scaling.
        max_val: Maximum value used for normalization scaling.

    Returns:
        Normalized values in range [0,1] with threshold centered at 0.5.
        Output type matches input type.

    Raises:
        TypeError: If ``targets`` is neither a NumPy array nor PyTorch tensor.
    """
    normalized = ((targets - threshold) / (max_val - min_val)) + 0.5
    if isinstance(targets, (np.ndarray, np.float32, np.float64)):
        normalized = np.minimum(normalized, 1)
        normalized = np.maximum(normalized, 0)
    elif isinstance(targets, torch.Tensor):
        normalized = torch.minimum(normalized, torch.tensor(1, device=targets.device))
        normalized = torch.maximum(normalized, torch.tensor(0, device=targets.device))
    else:
        msg = f"Targets must be either Tensor or Numpy array. Received {type(targets)}"
        raise TypeError(msg)
    return normalized