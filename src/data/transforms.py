# src/data/transforms.py
from __future__ import annotations

from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, Resize

# Dinomaly 默认图像尺寸与裁剪尺寸
DEFAULT_IMAGE_SIZE = 448
DEFAULT_CROP_SIZE = 392


def get_dinomaly_transforms(
    image_size: tuple[int, int] | None = None,
    crop_size: int | None = None,
) -> Compose:
    """返回 Dinomaly 默认的图像预处理变换。

    Args:
        image_size: 目标尺寸 (height, width)，默认 (448, 448)
        crop_size: 中心裁剪尺寸（正方形），默认 392

    Returns:
        Compose 变换：Resize -> CenterCrop -> Normalize
    """
    crop_size = crop_size or DEFAULT_CROP_SIZE
    image_size = image_size or (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)

    if crop_size > min(image_size):
        raise ValueError(f"Crop size {crop_size} cannot be larger than image size {image_size}")

    return Compose([
        Resize(image_size),
        CenterCrop(crop_size),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])