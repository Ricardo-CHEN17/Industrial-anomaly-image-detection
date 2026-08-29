import torch
from torch import nn


class LayerScale(nn.Module):
    """Learnable per-channel scaling factor.

    This module introduces a learnable scale parameter ``gamma`` applied
    to the input tensor. It is commonly used in modern transformer
    architectures to improve optimization stability.

    Args:
        dim: Number of feature channels.
        init_values: Initial value for the scale parameter; may be a float
            or a tensor of shape ``(dim,)``.
        inplace: Whether to apply the scaling operation in-place.
        device: Optional torch device for parameter initialization.
        dtype: Optional torch dtype for parameter initialization.
    """

    def __init__(
        self,
        dim: int,
        init_values: float | torch.Tensor = 1e-5,
        inplace: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.init_values = init_values
        self.gamma = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset scale parameters to their initialization values."""
        nn.init.constant_(self.gamma, self.init_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel-wise scaling to the input tensor."""
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
