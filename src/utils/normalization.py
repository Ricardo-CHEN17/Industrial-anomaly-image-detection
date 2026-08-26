# src/utils/normalization.py
"""Min-max 线性归一化工具。

提供 :func:`normalize`，将异常分数线性映射到 [0,1] 区间：
``(score - min_val) / (max_val - min_val)``，训练集最小分数映射到 0、
最大分数映射到 1，分数越大代表越异常。支持标量、NumPy 数组与 PyTorch 张量。
"""

from __future__ import annotations

import numpy as np
import torch


def normalize(
    score: float | int | np.ndarray | torch.Tensor,
    min_val: float | int | np.ndarray | torch.Tensor,
    max_val: float | int | np.ndarray | torch.Tensor,
) -> float | np.ndarray | torch.Tensor:
    """将异常分数线性归一化到 [0,1] 区间。

    Args:
        score: 待归一化的异常分数（标量、NumPy 数组或 PyTorch 张量）。
        min_val: 训练集正常分数的最小值，映射到 0。
        max_val: 训练集正常分数的最大值，映射到 1。

    Returns:
        归一化后的分数，类型与输入 ``score`` 一致（标量输入返回 float）。

    Raises:
        TypeError: 如果 ``score`` 不是标量、NumPy 数组或 PyTorch 张量。
    """
    if max_val <= min_val:
        return _invalid_range_result(score)

    result = (score - min_val) / (max_val - min_val)

    if isinstance(result, torch.Tensor):
        return torch.clamp(result, 0.0, 1.0)
    if isinstance(result, np.ndarray):
        return np.clip(result, 0.0, 1.0)
    if isinstance(result, (float, int, np.floating, np.integer)):
        return min(max(float(result), 0.0), 1.0)
    raise TypeError(f"score 必须是标量、NumPy 数组或 PyTorch 张量。Received {type(score)}")


def _invalid_range_result(score: float | int | np.ndarray | torch.Tensor) -> float | np.ndarray | torch.Tensor:
    """分数范围无效（max_val <= min_val）时返回与输入类型匹配的 0.5。"""
    if isinstance(score, torch.Tensor):
        return torch.full_like(score, 0.5, dtype=torch.float32)
    if isinstance(score, np.ndarray):
        return np.full_like(score, 0.5, dtype=np.float32)
    return 0.5
