from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfigError(Exception):
    """配置文件加载或解析失败时抛出的异常。"""


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = (
        config_path
        if config_path is not None
        else Path(__file__).resolve().parents[2] / "configs" / "default.json"
    )
    if not path.exists():
        raise ConfigError(f"配置文件不存在: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"配置文件解析失败: {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"配置文件读取失败: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"配置顶层必须是 JSON 对象: {path}")
    return data
