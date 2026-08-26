from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import torch

from src.core.args import AppConfig
from src.core.logging import setup_logging
from src.core.manifest import load_manifest
from src.core.seed import set_seed
from src.data.dataset import ManifestDataset
from src.data.transforms import get_dinomaly_transforms
from src.models.dinomaly.components.optimizer import StableAdamW, WarmCosineScheduler
from src.models.dinomaly.model import build_dinomaly
from src.models.dinomaly.training_config import TRAINING_CONFIG

MODEL_FORMAT_VERSION = "omniad-school-model-1.0"
PRETRAINED_ENCODER_FILENAME = "dinov2_vitb14_reg4_pretrain.pth"

_MODEL_PARAMS = (
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


def create_optimizer_and_scheduler(
    model: torch.nn.Module,
    total_steps: int,
) -> tuple[list[torch.optim.Optimizer], list[Any]]:
    optimizer_cfg = TRAINING_CONFIG["optimizer"]
    scheduler_cfg = TRAINING_CONFIG["scheduler"]
    if total_steps <= 0:
        raise ValueError(f"total_steps 必须为正整数: {total_steps}")

    optimizer = StableAdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=optimizer_cfg["lr"],
        betas=optimizer_cfg["betas"],
        weight_decay=optimizer_cfg["weight_decay"],
        amsgrad=optimizer_cfg["amsgrad"],
        eps=optimizer_cfg["eps"],
    )
    scheduler = WarmCosineScheduler(
        optimizer,
        base_value=scheduler_cfg["base_value"],
        final_value=scheduler_cfg["final_value"],
        total_iters=total_steps,
        warmup_iters=scheduler_cfg["warmup_iters"],
    )
    return [optimizer], [scheduler]


def _get_encoder_pretrained_path(config: AppConfig) -> Path:
    configured = config.model_params.get("encoder_pretrained_path")
    if configured:
        return Path(configured)
    return (
        Path(__file__).resolve().parents[2]
        / "model"
        / "auxiliary"
        / "pretrained"
        / PRETRAINED_ENCODER_FILENAME
    )


def _build_model(config: AppConfig) -> torch.nn.Module:
    kwargs: dict[str, Any] = {
        key: config.model_params[key] for key in _MODEL_PARAMS if key in config.model_params
    }
    kwargs["encoder_pretrained_path"] = str(_get_encoder_pretrained_path(config))
    return build_dinomaly(**kwargs)


def _compute_score_stats(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
) -> tuple[float, float]:
    model.eval()
    scores: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in dataloader:
            images = batch["image"].to(device)
            output = model(images)
            scores.append(output.pred_score.detach().cpu())
    if not scores:
        raise RuntimeError("无法计算分数统计：dataloader 为空")
    all_scores = torch.cat(scores)
    return float(all_scores.min().item()), float(all_scores.max().item())


def _save_checkpoint(model: torch.nn.Module, output_dir: Path, config: AppConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": dict(config.model_params),
        "seed": config.seed,
    }
    torch.save(checkpoint, output_dir / "shared.pth")


def _copy_pretrained_encoder(config: AppConfig, output_dir: Path) -> None:
    source = _get_encoder_pretrained_path(config)
    if not source.exists():
        raise FileNotFoundError(f"预训练编码器权重不存在: {source}")
    destination = output_dir / "auxiliary" / "pretrained" / PRETRAINED_ENCODER_FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _save_model_manifest(
    output_dir: Path,
    categories: list[str],
    score_min: float,
    score_max: float,
    config: AppConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_info = dict(config.model_params)
    model_info.pop("encoder_pretrained_path", None)
    manifest = {
        "format_version": MODEL_FORMAT_VERSION,
        "model_mode": "shared",
        "checkpoint": "shared.pth",
        "categories": categories,
        "score_range": [score_min, score_max],
        "model": model_info,
    }
    manifest_path = output_dir / "model_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def run_training(config: AppConfig) -> None:
    set_seed(config.seed)
    logger = setup_logging("INFO")
    logger.info("开始训练 Dinomaly 模型")
    try:
        samples = load_manifest(config.manifest, strict=False)
        if not samples:
            raise RuntimeError("训练 manifest 为空")
        categories = sorted({sample.category for sample in samples})

        dataset = ManifestDataset(
            data_root=config.data_root,
            samples=samples,
            transform=get_dinomaly_transforms(),
            return_original_size=False,
        )
        batch_size = int(config.training_params.get("batch_size", 8))
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            drop_last=True,
        )
        if len(dataloader) == 0:
            raise RuntimeError("训练数据不足，无法构成有效批次")

        encoder_path = _get_encoder_pretrained_path(config)
        if not encoder_path.exists():
            raise FileNotFoundError(f"预训练编码器权重不存在: {encoder_path}")

        model = _build_model(config).to(config.device)

        total_steps = int(
            config.training_params.get("max_steps", TRAINING_CONFIG["trainer"]["max_steps"])
        )
        optimizer_list, scheduler_list = create_optimizer_and_scheduler(model, total_steps)
        optimizer = optimizer_list[0]
        scheduler = scheduler_list[0]
        clip_val = float(TRAINING_CONFIG["trainer"]["gradient_clip_val"])

        model.train()
        current_step = 0
        logger.info(
            f"数据量 {len(samples)}，批次大小 {batch_size}，总步数 {total_steps}，"
            f"设备 {config.device}，类别 {categories}"
        )
        while current_step < total_steps:
            for batch in dataloader:
                if current_step >= total_steps:
                    break
                images = batch["image"].to(config.device)
                loss = model(images, global_step=current_step)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)
                optimizer.step()
                scheduler.step()
                current_step += 1
                if current_step % 10 == 0 or current_step == total_steps:
                    current_lr = scheduler.get_last_lr()[0]
                    logger.info(
                        f"step {current_step}/{total_steps} | loss {loss.item():.6f} | lr {current_lr:.6f}"
                    )

        stats_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=False,
        )
        score_min, score_max = _compute_score_stats(model, stats_loader, config.device)
        thresholds_dir = config.output_dir / "auxiliary" / "thresholds"
        thresholds_dir.mkdir(parents=True, exist_ok=True)
        minmax_path = thresholds_dir / "minmax.json"
        with minmax_path.open("w", encoding="utf-8") as f:
            json.dump({"min": score_min, "max": score_max}, f, indent=2)

        _copy_pretrained_encoder(config, config.output_dir)
        _save_checkpoint(model, config.output_dir, config)
        _save_model_manifest(config.output_dir, categories, score_min, score_max, config)

        logger.info(
            f"训练完成，已保存至 {config.output_dir}（score_range: [{score_min:.6f}, {score_max:.6f}]）"
        )
    except Exception as exc:
        logger.error(f"训练失败: {exc}")
        raise
