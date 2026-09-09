from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch

from src.core.manifest import ManifestSample


def _load_image(path: Path) -> np.ndarray:
    # 使用 np.fromfile 和 cv2.imdecode 避免中文路径问题
    try:
        img_array = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as exc:
        raise RuntimeError(f"图像读取失败: {path}: {exc}") from exc

    if img is None:
        raise FileNotFoundError(f"图像读取失败: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class ManifestDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root: Path,
        samples: list[ManifestSample],
        transform: Callable | None = None,
        return_original_size: bool = False,
    ) -> None:
        self.data_root = data_root
        self.samples = samples
        self.transform = transform
        self.return_original_size = return_original_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        image_path = self.data_root / sample.image_path
        image = _load_image(image_path)

        original_size = (image.shape[0], image.shape[1])
        h, w = original_size
        
        # Letterbox 几何一致性变换
        target_size = 392
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        pad_top = (target_size - new_h) // 2
        pad_bottom = target_size - new_h - pad_top
        pad_left = (target_size - new_w) // 2
        pad_right = target_size - new_w - pad_left
        
        # 使用 ImageNet mean (约 124, 116, 104) 填充，避免黑边高频响应
        image = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right, 
            cv2.BORDER_CONSTANT, value=[124, 116, 104]
        )

        if self.transform is not None:
            # torchvision v2 变换（torchvision 0.28）对 numpy 数组是 no-op，
            # 对 (H,W,C) 张量会误判布局，且 Normalize 要求 float 张量；
            # 因此统一转为 (C,H,W) 的 float 张量并缩放到 [0,1] 再应用变换。
            image = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)
            image = self.transform(image)

        item: dict[str, Any] = {
            "image": image,
            "image_name": sample.image_name,
            "category": sample.category,
            "original_size": original_size,
            "padding": (pad_top, pad_bottom, pad_left, pad_right),
        }
        return item
