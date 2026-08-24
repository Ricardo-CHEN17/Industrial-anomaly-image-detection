# src/models/dinomaly/dinov2/layers/__init__.py
"""Layers needed to build DINOv2.

References:
    https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/__init__.py

Classes:
    Attention: Standard multi-head self-attention layer used in Vision Transformers.
    MemEffAttention: Memory-efficient variant of multi-head attention optimized for large inputs.
    DropPath: Implements stochastic depth, randomly dropping residual connections during training.
    LayerScale: Applies learnable per-channel scaling to stabilize deep transformer training.
"""

from .attention import Attention, MemEffAttention
from .drop_path import DropPath
from .layer_scale import LayerScale

__all__ = [
    "Attention",
    "DropPath",
    "LayerScale",
    "MemEffAttention",
]