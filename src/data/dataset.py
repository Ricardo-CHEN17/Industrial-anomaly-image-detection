from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.core.manifest import ManifestSample
from src.core.preprocess import PreprocessConfig


_CV2_INTERPOLATIONS = {
    "linear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
    "area": cv2.INTER_AREA,
    "cubic": cv2.INTER_CUBIC,
}


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
        preprocess: PreprocessConfig,
        transform: Callable | None = None,
        return_original_size: bool = False,
        return_valid_patch_mask: bool = False,
    ) -> None:
        self.data_root = data_root
        self.samples = samples
        self.preprocess = preprocess
        self.transform = transform
        self.return_original_size = return_original_size
        self.return_valid_patch_mask = return_valid_patch_mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        image_path = self.data_root / sample.image_path
        image = _load_image(image_path)

        original_size = (image.shape[0], image.shape[1])
        letterbox = self.preprocess.letterbox_meta(original_size)
        new_h, new_w = letterbox.resized_size
        pad_top, pad_bottom, pad_left, pad_right = letterbox.padding
        image = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=_CV2_INTERPOLATIONS[self.preprocess.resize_interpolation],
        )

        image = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=list(self.preprocess.padding_color_rgb),
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
        }
        if self.return_original_size:
            item["original_size"] = letterbox.original_size
            item["padding"] = letterbox.padding
        if self.return_valid_patch_mask:
            valid_pixels = torch.zeros(
                (1, self.preprocess.canvas_size, self.preprocess.canvas_size),
                dtype=torch.float32,
            )
            valid_pixels[:, pad_top : pad_top + new_h, pad_left : pad_left + new_w] = 1.0
            item["valid_patch_mask"] = F.avg_pool2d(
                valid_pixels,
                kernel_size=self.preprocess.patch_size,
                stride=self.preprocess.patch_size,
            )
        return item
