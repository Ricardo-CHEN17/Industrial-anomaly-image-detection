# src/data/transforms.py
from __future__ import annotations

from torchvision.transforms.v2 import Compose, Normalize

# Dinomaly 默认画布尺寸 (Letterbox)
DEFAULT_IMAGE_SIZE = 392
DEFAULT_CROP_SIZE = 392 # 保留兼容性


def get_dinomaly_transforms(
    image_size: tuple[int, int] | None = None,
    crop_size: int | None = None,
) -> Compose:
    """返回 Dinomaly 的归一化变换（几何缩放与填充已移至 Dataset 中处理）。"""
    return Compose([
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])