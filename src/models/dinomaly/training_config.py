# src/models/dinomaly/training_config.py
from typing import Any

# 默认训练超参数（优化器、调度器、训练器配置）
TRAINING_CONFIG: dict[str, Any] = {
    "optimizer": {
        "lr": 1e-3,
        "betas": (0.9, 0.999),
        "weight_decay": 1e-4,
        "amsgrad": True,
        "eps": 1e-8,
        "clip_threshold": 1.0,
    },
    "scheduler": {
        "base_value": 2e-3,
        "final_value": 2e-4,
        "total_iters": 5000,
        "warmup_iters": 100,
    },
    "trainer": {
        "gradient_clip_val": 0.1,
        "num_sanity_val_steps": 0,
        "max_steps": 5000,
    },
}