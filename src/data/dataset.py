from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch

from src.core.manifest import ManifestSample


def _load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
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

        original_size: tuple[int, int] | None = None
        if self.return_original_size:
            original_size = (image.shape[0], image.shape[1])

        if self.transform is not None:
            # torchvision v2 变换（torchvision 0.28）对 numpy 数组是 no-op，
            # 对 (H,W,C) 张量会误判布局，且 Normalize 要求 float 张量；
            # 因此统一转为 (C,H,W) 的 float 张量并缩放到 [0,1] 再应用变换。
            image = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)
            image = self.transform(image)

        item: dict[str, Any] = {
            "image": image,
            "sample_id": sample.sample_id,
            "category": sample.category,
        }
        if original_size is not None:
            item["original_size"] = original_size
        return item
