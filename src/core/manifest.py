from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


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
        if not has_image_name and strict:
            raise ManifestError("manifest 缺少必需列: image_name")

        for row in reader:
            if all(value is None or value.strip() == "" for value in row.values()):
                continue

            category = (row.get("category") or "").strip()
            image_path = (row.get("image_path") or "").strip()
            if not category or not image_path:
                raise ManifestError("manifest 行存在空字段: category 或 image_path")

            image_name = (row.get("image_name") or "").strip()
            if not image_name and strict:
                raise ManifestError("manifest 行存在空字段: image_name")

            samples.append(
                ManifestSample(image_name=image_name, category=category, image_path=Path(image_path))
            )

    return samples
