from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath


@dataclass(frozen=True)
class ManifestSample:
    image_name: str
    category: str
    image_path: Path


class ManifestError(Exception):
    """manifest 文件读取或解析失败时抛出的异常。"""


def load_manifest(manifest_path: Path, strict: bool = True) -> list[ManifestSample]:
    if not manifest_path.exists():
        raise ManifestError(f"manifest 文件不存在: {manifest_path}")

    samples: list[ManifestSample] = []
    seen = set()
    map_targets = set()
    try:
        f = manifest_path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise ManifestError(f"manifest 文件读取失败: {manifest_path}: {exc}") from exc
    with f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ManifestError(f"manifest 文件为空或无表头: {manifest_path}")

        required = {"category", "image_path"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ManifestError(f"manifest 缺少必需列: {', '.join(sorted(missing))}")

        has_image_name = "image_name" in reader.fieldnames
        has_sample_id = "sample_id" in reader.fieldnames
        if strict and not has_image_name:
            raise ManifestError("评测 manifest 缺少必需列: image_name")

        for row in reader:
            if all(value is None or value.strip() == "" for value in row.values()):
                continue

            category = (row.get("category") or "").strip()
            image_path = (row.get("image_path") or "").strip()
            if not category or not image_path:
                raise ManifestError("manifest 行存在空字段: category 或 image_path")

            image_name = (row.get("image_name") or (row.get("sample_id") if not strict and has_sample_id else "") or "").strip()
            if not image_name and strict:
                raise ManifestError("评测 manifest 行存在空字段: image_name")

            for value in (category, image_path, image_name):
                if not value:
                    continue
                if ("\\" in value or ":" in value or PureWindowsPath(value).anchor
                        or ".." in PurePosixPath(value).parts or "\x00" in value):
                    raise ManifestError("manifest 路径必须为安全的 POSIX 相对路径")
            if len(PurePosixPath(category).parts) != 1 or category in {".", ".."}:
                raise ManifestError("category 必须为单个目录名")
            if "ground_truth" in PurePosixPath(image_path).parts:
                raise ManifestError("manifest 不得引用 ground_truth")
            if strict:
                name_path = PurePosixPath(image_name)
                if len(name_path.parts) != 2 or name_path.parts[0] != "test" or not name_path.suffix:
                    raise ManifestError("评测 image_name 必须是 test/ 下带扩展名的 POSIX 相对路径")
            key = (category.casefold(), (image_name or image_path).casefold())
            target = (category.casefold(), str(PurePosixPath(image_name).with_suffix(".npy")).casefold()) if image_name else key
            if key in seen or (strict and target in map_targets):
                raise ManifestError("manifest 包含重复样本或异常图文件名冲突")
            seen.add(key)
            map_targets.add(target)

            samples.append(
                ManifestSample(image_name=image_name, category=category, image_path=Path(image_path))
            )

    return samples


def require_normal_training_samples(samples):
    for sample in samples:
        parts = sample.image_path.parts
        if len(parts) < 3 or tuple(parts[-3:-1]) != ("train", "good"):
            raise ManifestError("训练和校准仅允许 manifest 中的 train/good 正常图像")
