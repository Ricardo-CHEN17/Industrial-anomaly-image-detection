"""可序列化的图像预处理与 Letterbox 几何配置。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


class PreprocessConfigError(ValueError):
    """预处理配置不合法时抛出。"""


def _as_float_triplet(value: Any, field_name: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise PreprocessConfigError(f"{field_name} 必须是包含 3 个数值的列表")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise PreprocessConfigError(f"{field_name} 必须是数值列表") from exc
    return result  # type: ignore[return-value]


def _as_rgb_triplet(value: Any) -> tuple[int, int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise PreprocessConfigError("padding_color_rgb 必须是包含 3 个整数的列表")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise PreprocessConfigError("padding_color_rgb 必须是整数列表")
    result = tuple(value)
    if any(item < 0 or item > 255 for item in result):
        raise PreprocessConfigError("padding_color_rgb 的每个值必须在 [0, 255]")
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class LetterboxMeta:
    """单张图像的几何反变换所需信息。"""

    original_size: tuple[int, int]
    resized_size: tuple[int, int]
    padding: tuple[int, int, int, int]


@dataclass(frozen=True)
class PreprocessConfig:
    """模型输入与像素异常图反变换所需的全部稳定配置。"""

    canvas_size: int = 392
    patch_size: int = 14
    mode: str = "letterbox"
    resize_interpolation: str = "linear"
    map_interpolation: str = "linear"
    padding_mode: str = "constant"
    padding_color_rgb: tuple[int, int, int] = (124, 116, 104)
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise PreprocessConfigError(f"不支持的 preprocess schema_version: {self.schema_version}")
        if self.mode != "letterbox":
            raise PreprocessConfigError(f"仅支持 mode='letterbox'，实际为 {self.mode!r}")
        if not isinstance(self.canvas_size, int) or isinstance(self.canvas_size, bool) or self.canvas_size <= 0:
            raise PreprocessConfigError("canvas_size 必须是正整数")
        if not isinstance(self.patch_size, int) or isinstance(self.patch_size, bool) or self.patch_size <= 0:
            raise PreprocessConfigError("patch_size 必须是正整数")
        if self.canvas_size % self.patch_size != 0:
            raise PreprocessConfigError(
                f"canvas_size ({self.canvas_size}) 必须能被 patch_size ({self.patch_size}) 整除"
            )
        if self.resize_interpolation not in {"linear", "nearest", "area", "cubic"}:
            raise PreprocessConfigError(f"不支持的 resize_interpolation: {self.resize_interpolation!r}")
        if self.map_interpolation not in {"linear", "nearest", "cubic"}:
            raise PreprocessConfigError(f"不支持的 map_interpolation: {self.map_interpolation!r}")
        if self.padding_mode != "constant":
            raise PreprocessConfigError(f"仅支持 padding_mode='constant'，实际为 {self.padding_mode!r}")
        if len(self.padding_color_rgb) != 3 or any(value < 0 or value > 255 for value in self.padding_color_rgb):
            raise PreprocessConfigError("padding_color_rgb 必须是 [0, 255] 内的 RGB 三元组")
        if len(self.mean) != 3 or len(self.std) != 3:
            raise PreprocessConfigError("mean 与 std 必须是长度为 3 的列表")
        if any(value <= 0 for value in self.std):
            raise PreprocessConfigError("std 的每个值必须大于 0")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreprocessConfig":
        if not isinstance(data, dict):
            raise PreprocessConfigError("preprocess 必须是 JSON 对象")
        required = {
            "schema_version",
            "mode",
            "canvas_size",
            "patch_size",
            "resize_interpolation",
            "map_interpolation",
            "padding_mode",
            "padding_color_rgb",
            "normalization",
        }
        missing = required - set(data)
        if missing:
            raise PreprocessConfigError(f"preprocess 缺少字段: {', '.join(sorted(missing))}")
        normalization = data["normalization"]
        if not isinstance(normalization, dict):
            raise PreprocessConfigError("preprocess.normalization 必须是 JSON 对象")
        try:
            return cls(
                schema_version=data["schema_version"],
                mode=data["mode"],
                canvas_size=data["canvas_size"],
                patch_size=data["patch_size"],
                resize_interpolation=data["resize_interpolation"],
                map_interpolation=data["map_interpolation"],
                padding_mode=data["padding_mode"],
                padding_color_rgb=_as_rgb_triplet(data["padding_color_rgb"]),
                mean=_as_float_triplet(normalization.get("mean"), "normalization.mean"),
                std=_as_float_triplet(normalization.get("std"), "normalization.std"),
            )
        except KeyError as exc:
            raise PreprocessConfigError(f"preprocess 缺少字段: {exc.args[0]}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "canvas_size": self.canvas_size,
            "patch_size": self.patch_size,
            "resize_interpolation": self.resize_interpolation,
            "map_interpolation": self.map_interpolation,
            "padding_mode": self.padding_mode,
            "padding_color_rgb": list(self.padding_color_rgb),
            "normalization": {
                "mean": list(self.mean),
                "std": list(self.std),
            },
        }

    def letterbox_meta(self, original_size: tuple[int, int]) -> LetterboxMeta:
        height, width = original_size
        if height <= 0 or width <= 0:
            raise PreprocessConfigError(f"图像尺寸必须为正数: {original_size}")
        scale = min(self.canvas_size / height, self.canvas_size / width)
        resized_height = min(self.canvas_size, max(1, int(round(height * scale))))
        resized_width = min(self.canvas_size, max(1, int(round(width * scale))))
        pad_top = (self.canvas_size - resized_height) // 2
        pad_bottom = self.canvas_size - resized_height - pad_top
        pad_left = (self.canvas_size - resized_width) // 2
        pad_right = self.canvas_size - resized_width - pad_left
        return LetterboxMeta(
            original_size=original_size,
            resized_size=(resized_height, resized_width),
            padding=(pad_top, pad_bottom, pad_left, pad_right),
        )

    def remove_padding(self, anomaly_map: Any, padding: tuple[int, int, int, int]) -> Any:
        if getattr(anomaly_map, "ndim", None) != 2:
            raise PreprocessConfigError(f"异常图必须是二维数组，实际形状: {getattr(anomaly_map, 'shape', None)}")
        if anomaly_map.shape != (self.canvas_size, self.canvas_size):
            raise PreprocessConfigError(
                f"异常图尺寸应为 {(self.canvas_size, self.canvas_size)}，实际为 {anomaly_map.shape}"
            )
        pad_top, pad_bottom, pad_left, pad_right = padding
        end_h = self.canvas_size - pad_bottom
        end_w = self.canvas_size - pad_right
        if min(pad_top, pad_bottom, pad_left, pad_right) < 0 or pad_top >= end_h or pad_left >= end_w:
            raise PreprocessConfigError(f"padding 非法: {padding}")
        return anomaly_map[pad_top:end_h, pad_left:end_w]
