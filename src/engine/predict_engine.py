from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch

from src.core.args import AppConfig
from src.core.manifest import load_manifest
from src.data.dataset import ManifestDataset
from src.data.transforms import get_dinomaly_transforms
from src.models.builder import load_model_from_dir
from src.utils.image_io import resize_anomaly_map, save_16bit_png
from src.utils.normalization import normalize


def _extract_original_size(value: object) -> tuple[int, int]:
    """从 DataLoader 批次中提取原始图像尺寸 (height, width)。

    兼容 default_collate 在不同 torch 版本下的多种形态：
    ``(tensor, tensor)``、``[tensor, tensor]``、``[(h, w)]``、
    ``tensor([h, w])``、``(h, w)``、``[h, w]``。
    """
    if isinstance(value, torch.Tensor):
        if value.ndim == 2 and value.shape[0] == 1:
            value = value[0]
        return (int(value[0].item()), int(value[1].item()))
    if len(value) == 1 and isinstance(value[0], (tuple, list)):
        value = value[0]
    height = value[0].item() if hasattr(value[0], "item") else value[0]
    width = value[1].item() if hasattr(value[1], "item") else value[1]
    return (int(height), int(width))


def run_inference(config: AppConfig) -> None:
    samples = load_manifest(config.manifest, strict=True)
    dataset = ManifestDataset(
        data_root=config.data_root,
        samples=samples,
        transform=get_dinomaly_transforms(),
        return_original_size=True,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    )

    bundle = load_model_from_dir(config.model_dir, config.device)
    model = bundle.model
    score_min = bundle.score_min
    score_max = bundle.score_max

    config.output_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = config.output_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    results: list[tuple[str, float]] = []
    try:
        with torch.inference_mode():
            for batch in loader:
                sample_id = batch["sample_id"][0]
                image = batch["image"].to(config.device)
                height_orig, width_orig = _extract_original_size(batch["original_size"])

                output = model(image)
                score = float(output.pred_score.detach().cpu().item())
                score = normalize(score, min_val=score_min, max_val=score_max)

                anomaly_map = np.squeeze(output.anomaly_map.detach().cpu().numpy())
                if anomaly_map.ndim != 2:
                    raise RuntimeError(f"anomaly_map 形状非法: {output.anomaly_map.shape}")

                resized = resize_anomaly_map(anomaly_map, (height_orig, width_orig))
                save_16bit_png(resized, maps_dir / f"{sample_id}.png")
                results.append((sample_id, score))
    except Exception as exc:
        print(f"推理失败: {exc}", file=sys.stderr)
        raise

    predictions_path = config.output_dir / "predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "image_score"])
        for sample_id, score in results:
            writer.writerow([sample_id, f"{score:.6f}"])

    if len(results) != len(samples):
        raise RuntimeError(
            f"predictions.csv 样本数 {len(results)} 与 manifest 样本数 {len(samples)} 不一致"
        )
