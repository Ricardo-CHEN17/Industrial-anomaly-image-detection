# src/engine/train_engine.py
from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.models.dinomaly.components.optimizer import StableAdamW, WarmCosineScheduler
from src.models.dinomaly.training_config import TRAINING_CONFIG

import torch

def create_optimizer_and_scheduler(
    model: torch.nn.Module,
    total_steps: int,
) -> tuple[list[Optimizer], list[LRScheduler]]:
    """创建 Dinomaly 训练所需的优化器和学习率调度器。

    Args:
        model: Dinomaly 模型实例，其可训练参数应为 bottleneck 和 decoder。
        total_steps: 总训练步数（由训练配置或数据量决定）。

    Returns:
        (optimizers, schedulers) 元组，结构与 PyTorch Lightning 类似。
    """
    # 收集可训练参数（模型构建时已冻结其他参数）
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer_config = TRAINING_CONFIG["optimizer"]
    optimizer = StableAdamW([{"params": trainable_params}], **optimizer_config)

    scheduler_config = TRAINING_CONFIG["scheduler"].copy()
    scheduler_config["total_iters"] = total_steps
    lr_scheduler = WarmCosineScheduler(optimizer, **scheduler_config)

    return [optimizer], [lr_scheduler]