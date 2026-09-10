# src/data/transforms.py
from __future__ import annotations

from torchvision.transforms.v2 import Compose, Normalize

from src.core.preprocess import PreprocessConfig


def get_dinomaly_transforms(
    preprocess: PreprocessConfig,
) -> Compose:
    """返回 Dinomaly 的归一化变换；几何规则由 ``preprocess`` 唯一决定。"""
    return Compose([
        Normalize(mean=list(preprocess.mean), std=list(preprocess.std)),
    ])
