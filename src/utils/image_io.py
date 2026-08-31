from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def save_float32_npy(anomaly_map: np.ndarray, save_path: Path) -> None:
    if anomaly_map.ndim == 3:
        anomaly_map = np.squeeze(anomaly_map)
    if anomaly_map.ndim != 2:
        raise ValueError(f"anomaly_map 必须是 (H, W) 或 (1, H, W)，当前形状: {anomaly_map.shape}")

    clipped = np.clip(anomaly_map, 0.0, 1.0).astype(np.float32)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, clipped)


def save_16bit_png(anomaly_map: np.ndarray, save_path: Path) -> None:
    if anomaly_map.ndim == 3:
        anomaly_map = np.squeeze(anomaly_map)
    if anomaly_map.ndim != 2:
        raise ValueError(f"anomaly_map 必须是 (H, W) 或 (1, H, W)，当前形状: {anomaly_map.shape}")

    clipped = np.clip(anomaly_map, 0.0, 1.0)
    uint16_map = np.round(clipped * 65535.0).astype(np.uint16)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ok = cv2.imwrite(str(save_path), uint16_map)
    except cv2.error as exc:
        raise RuntimeError(f"异常图保存失败: {save_path}: {exc}") from exc
    if not ok:
        raise RuntimeError(f"异常图保存失败: {save_path}")


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"图像读取失败: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_anomaly_map(anomaly_map: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    height, width = target_size
    if height <= 0 or width <= 0:
        raise ValueError(f"target_size 必须为正整数: {target_size}")

    squeeze_axis = anomaly_map.ndim == 3
    if squeeze_axis:
        anomaly_map = np.squeeze(anomaly_map)
    if anomaly_map.ndim != 2:
        raise ValueError(f"anomaly_map 必须是 (H, W) 或 (1, H, W)，当前形状: {anomaly_map.shape}")

    resized = cv2.resize(anomaly_map, (width, height), interpolation=cv2.INTER_LINEAR)

    if squeeze_axis:
        resized = resized[np.newaxis, :, :]
    return resized
