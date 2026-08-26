from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.models.dinomaly.model import build_dinomaly
from src.models.dinomaly.torch_model import DinomalyModel

CHECKPOINT_FILENAME = "shared.pth"
MINMAX_PATH = "auxiliary/thresholds/minmax.json"
PRETRAINED_ENCODER_PATH = "auxiliary/pretrained/dinov2_vitb14_reg4_pretrain.pth"

_BUILD_PARAMS = (
    "encoder_name",
    "bottleneck_dropout",
    "decoder_depth",
    "target_layers",
    "fuse_layer_encoder",
    "fuse_layer_decoder",
    "remove_class_token",
    "use_context_recentering",
    "precision",
    "encoder_pretrained_path",
)


class ModelLoadError(Exception):
    """模型加载过程中的所有错误。"""


@dataclass
class ModelBundle:
    model: torch.nn.Module
    score_min: float
    score_max: float
    categories: list[str] | None = None


def _load_model_manifest(model_dir: Path) -> dict[str, Any]:
    manifest_path = model_dir / "model_manifest.json"
    if not model_dir.exists():
        raise ModelLoadError(f"模型目录不存在: {model_dir}")
    if not manifest_path.exists():
        raise ModelLoadError(f"模型清单不存在: {manifest_path}")
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        raise ModelLoadError(f"模型清单解析失败: {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ModelLoadError(f"模型清单顶层必须是 JSON 对象: {manifest_path}")
    return manifest


def _load_score_range(model_dir: Path) -> tuple[float, float]:
    minmax_path = model_dir / MINMAX_PATH
    if not minmax_path.exists():
        raise ModelLoadError(f"归一化参数文件不存在: {minmax_path}")
    try:
        with minmax_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        raise ModelLoadError(f"归一化参数解析失败: {minmax_path}: {exc}") from exc

    min_val = data.get("min")
    max_val = data.get("max")
    if not isinstance(min_val, (int, float)) or isinstance(min_val, bool):
        raise ModelLoadError(f"minmax 参数缺少合法的 min 字段: {minmax_path}")
    if not isinstance(max_val, (int, float)) or isinstance(max_val, bool):
        raise ModelLoadError(f"minmax 参数缺少合法的 max 字段: {minmax_path}")
    if float(min_val) >= float(max_val):
        raise ModelLoadError(f"minmax 参数 min 必须小于 max: {min_val!r} >= {max_val!r}")
    return float(min_val), float(max_val)


def _build_model(model_dir: Path, config: dict[str, Any]) -> DinomalyModel:
    pretrained_encoder = model_dir / PRETRAINED_ENCODER_PATH
    if not pretrained_encoder.exists():
        raise ModelLoadError(f"预训练编码器权重不存在: {pretrained_encoder}")
    kwargs: dict[str, Any] = {key: config[key] for key in _BUILD_PARAMS if key in config}
    kwargs["encoder_pretrained_path"] = str(pretrained_encoder)
    return build_dinomaly(**kwargs)


def load_model_from_dir(
    model_dir: Path,
    device: str,
    config: dict[str, Any] | None = None,
) -> ModelBundle:
    manifest = _load_model_manifest(model_dir)
    score_min, score_max = _load_score_range(model_dir)

    categories = manifest.get("categories")
    if categories is not None and (
        not isinstance(categories, list) or not all(isinstance(c, str) for c in categories)
    ):
        raise ModelLoadError(f"categories 必须是字符串列表: {categories!r}")

    weight_name = manifest.get("checkpoint") or manifest.get("checkpoint_pattern") or CHECKPOINT_FILENAME
    checkpoint_path = (model_dir / weight_name).resolve()
    if not checkpoint_path.is_relative_to(model_dir.resolve()):
        raise ModelLoadError(f"权重路径越界: {checkpoint_path}")
    if not checkpoint_path.exists():
        raise ModelLoadError(f"模型权重文件不存在: {checkpoint_path}")

    model_cfg = dict(config) if config is not None else {}
    manifest_model = manifest.get("model")
    if manifest_model is not None:
        if not isinstance(manifest_model, dict):
            raise ModelLoadError(f"manifest 的 model 字段必须是 JSON 对象: {manifest_model!r}")
        model_cfg.update(manifest_model)

    try:
        model = _build_model(model_dir, model_cfg)
    except Exception as exc:
        raise ModelLoadError(f"模型构建失败: {exc}") from exc

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as exc:
        raise ModelLoadError(f"模型权重加载失败: {checkpoint_path}: {exc}") from exc

    if not isinstance(checkpoint, dict):
        raise ModelLoadError(f"模型权重格式非法: {checkpoint_path}")
    state_dict = checkpoint.get("state_dict", checkpoint)

    try:
        model.load_state_dict(state_dict)
    except (RuntimeError, ValueError) as exc:
        raise ModelLoadError(f"权重与模型结构不匹配: {exc}") from exc

    model.to(device)
    model.eval()
    return ModelBundle(model=model, score_min=score_min, score_max=score_max, categories=categories)
