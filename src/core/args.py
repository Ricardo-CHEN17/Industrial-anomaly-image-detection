"""命令行参数解析与配置合并模块。

负责将 configs/default.json 与命令行参数合并为不可变配置对象。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.config import ConfigError, load_config


class ConfigValidationError(ConfigError):
    """配置校验失败时抛出的异常。"""


@dataclass(frozen=True)
class AppConfig:
    """不可变的全局配置对象。"""

    data_root: Path
    manifest: Path
    output_dir: Path
    model_dir: Path | None = None
    device: str = "cpu"
    num_workers: int = 4
    seed: int = 2026
    model_params: dict[str, Any] = field(default_factory=dict)
    training_params: dict[str, Any] = field(default_factory=dict)


def parse_cli_args(train_mode: bool, argv: list[str] | None = None) -> argparse.Namespace:
    """根据模式解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True, help="图像根目录")
    parser.add_argument("--manifest", type=Path, required=True, help="manifest 文件路径")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出目录")
    parser.add_argument("--device", type=str, required=True, help="计算设备")
    parser.add_argument("--num-workers", type=int, required=True, help="数据加载进程数")
    if train_mode:
        parser.add_argument("--seed", type=int, required=True, help="随机种子")
    else:
        parser.add_argument("--model-dir", type=Path, required=True, help="模型目录")
    return parser.parse_args(argv)


def merge_configs(defaults: dict[str, Any], cli_args: argparse.Namespace) -> dict[str, Any]:
    """合并配置字典与命令行参数，命令行非 None 值优先。"""
    merged = dict(defaults)
    for key, value in vars(cli_args).items():
        if value is not None:
            merged[key] = value
    return merged


def validate_config(cfg: dict[str, Any]) -> None:
    """校验配置的必填项与合法性。"""
    required = ("data_root", "manifest", "output_dir")
    for key in required:
        if key not in cfg or cfg.get(key) is None:
            raise ConfigValidationError(f"缺少必填配置项: {key}")

    data_root = Path(cfg["data_root"])
    manifest = Path(cfg["manifest"])
    if not data_root.exists():
        raise ConfigValidationError(f"数据根目录不存在: {data_root}")
    if not manifest.exists():
        raise ConfigValidationError(f"manifest 文件不存在: {manifest}")

    if "num_workers" in cfg:
        num_workers = cfg["num_workers"]
        if not isinstance(num_workers, int) or isinstance(num_workers, bool) or num_workers < 0:
            raise ConfigValidationError(f"num_workers 必须是非负整数: {num_workers!r}")

    if "device" in cfg:
        device = str(cfg["device"])
        if not (
            device == "cpu"
            or device.startswith("cuda:")
            or device == "mps"
            or device.startswith("mps:")
        ):
            raise ConfigValidationError(f"device 非法: {device!r}")

    if "seed" in cfg and cfg["seed"] is not None:
        seed = cfg["seed"]
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ConfigValidationError(f"seed 必须是非负整数: {seed!r}")

    for key in ("model", "training"):
        value = cfg.get(key)
        if value is not None and not isinstance(value, dict):
            raise ConfigValidationError(f"{key} 必须是 JSON 对象: {value!r}")


def build_config(train_mode: bool, argv: list[str] | None = None) -> AppConfig:
    """构建并校验最终的不可变配置对象。"""
    defaults = load_config()
    cli_args = parse_cli_args(train_mode, argv)
    merged = merge_configs(defaults, cli_args)
    validate_config(merged)
    return AppConfig(
        data_root=Path(merged["data_root"]),
        manifest=Path(merged["manifest"]),
        output_dir=Path(merged["output_dir"]),
        model_dir=Path(merged["model_dir"]) if merged.get("model_dir") else None,
        device=str(merged["device"]),
        num_workers=int(merged.get("num_workers", 4)),
        seed=int(merged.get("seed", 2026)),
        model_params=merged.get("model", {}),
        training_params=merged.get("training", {}),
    )
