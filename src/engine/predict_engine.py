from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.core.args import AppConfig
from src.core.manifest import load_manifest
from src.data.dataset import ManifestDataset
from src.data.transforms import get_dinomaly_transforms
from src.models.builder import load_model_from_dir
from src.utils.image_io import save_float32_npy
from src.utils.normalization import normalize


def run_inference(config: AppConfig) -> None:
    samples = load_manifest(config.manifest, strict=True)
    dataset = ManifestDataset(
        data_root=config.data_root,
        samples=samples,
        transform=get_dinomaly_transforms(),
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

    model.eval()
    preds: dict[str, dict[str, str | float]] = defaultdict(dict)
    try:
        with torch.inference_mode():
            for batch in loader:
                image_name = batch["image_name"][0]
                category = batch["category"][0]
                image = batch["image"].to(config.device)

                output = model(image)
                score = float(output.pred_score.detach().cpu().item())
                score = normalize(score, min_val=score_min, max_val=score_max)

                anomaly_map = np.squeeze(output.anomaly_map.detach().cpu().numpy())
                if anomaly_map.ndim != 2:
                    raise RuntimeError(f"anomaly_map 形状非法: {output.anomaly_map.shape}")

                category_dir = config.output_dir / category
                map_rel = Path("pred_maps") / Path(image_name).with_suffix(".npy")
                map_path = category_dir / map_rel
                map_path.parent.mkdir(parents=True, exist_ok=True)
                save_float32_npy(anomaly_map, map_path)

                preds[category][image_name] = {
                    "anomaly_score": score,
                    "anomaly_map": map_rel.as_posix(),
                }
    except Exception as exc:
        print(f"推理失败: {exc}", file=sys.stderr)
        raise

    for category, pred in preds.items():
        category_dir = config.output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        pred_path = category_dir / "pred.json"
        with pred_path.open("w", encoding="utf-8") as f:
            json.dump(pred, f, indent=2, ensure_ascii=False)

    if sum(len(pred) for pred in preds.values()) != len(samples):
        raise RuntimeError(
            f"预测样本数 {sum(len(pred) for pred in preds.values())} 与 manifest 样本数 {len(samples)} 不一致"
        )
