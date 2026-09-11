"""Image-only pooling of the existing model's raw feature-distance maps."""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ImageScoring:
    weights: tuple[float, ...] = (0.5, 0.5)
    resize: int = 256
    top_ratio: float = 0.01
    gaussian_kernel_size: int = 5
    gaussian_sigma: float = 4.0
    region: str = "valid"

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("image_scoring must be an object")
        config = dict(value)
        if "weights" in config:
            try:
                config["weights"] = tuple(float(weight) for weight in config["weights"])
            except (TypeError, ValueError) as exc:
                raise ValueError("image_scoring.weights must be numeric") from exc
        try:
            result = cls(**config)
        except TypeError as exc:
            raise ValueError(f"invalid image_scoring configuration: {exc}") from exc
        if result.region != "valid":
            raise ValueError("image_scoring.region must be valid")
        if not result.weights or any(not math.isfinite(w) or w < 0 for w in result.weights) or sum(result.weights) <= 0:
            raise ValueError("image_scoring.weights must be finite, nonnegative and have a positive sum")
        if isinstance(result.resize, bool) or not isinstance(result.resize, int) or result.resize <= 0:
            raise ValueError("image_scoring.resize must be a positive integer")
        if isinstance(result.top_ratio, bool) or not isinstance(result.top_ratio, (int, float)) or not math.isfinite(result.top_ratio) or not 0 <= result.top_ratio <= 1:
            raise ValueError("image_scoring.top_ratio must be in [0, 1]")
        size = result.gaussian_kernel_size
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0 or size % 2 == 0:
            raise ValueError("image_scoring.gaussian_kernel_size must be positive and odd")
        if not math.isfinite(result.gaussian_sigma) or result.gaussian_sigma <= 0:
            raise ValueError("image_scoring.gaussian_sigma must be positive and finite")
        return result


def batch_padding(collated):
    """Convert default DataLoader collation (four tensors) to per-image tuples."""
    return [tuple(int(side[i]) for side in collated) for i in range(len(collated[0]))]


def image_scores(group_maps, padding, config: ImageScoring):
    """Crop BEFORE resize/blur so unconstrained padding cannot leak into scores.

    No in-place operations: the pixel branch retains its original maps exactly.
    """
    maps = torch.cat(group_maps, dim=1)
    if len(config.weights) != maps.shape[1]:
        raise ValueError("image_scoring.weights must match the feature group count")
    if padding is None or len(padding) != maps.shape[0]:
        raise ValueError("valid image scoring requires padding metadata for every image")
    weights = maps.new_tensor(config.weights)
    fused = (maps * (weights / weights.sum()).view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
    coords = torch.arange(config.gaussian_kernel_size, device=maps.device, dtype=maps.dtype)
    coords = coords - config.gaussian_kernel_size // 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2 * config.gaussian_sigma ** 2))
    kernel = (kernel / kernel.sum()).view(1, 1, *kernel.shape)
    scores = []
    height, width = fused.shape[-2:]
    for i, (top, bottom, left, right) in enumerate(padding):
        if min(top, bottom, left, right) < 0 or top + bottom >= height or left + right >= width:
            raise ValueError("invalid image padding")
        valid = fused[i:i + 1, :, top:height - bottom, left:width - right]
        resized = F.interpolate(valid, size=(config.resize, config.resize), mode="bilinear", align_corners=False)
        smoothed = F.conv2d(resized, kernel, padding=config.gaussian_kernel_size // 2)
        flat = smoothed.flatten(1)
        if config.top_ratio == 0:
            score = flat.max(dim=1).values
        else:
            count = max(1, int(flat.shape[1] * config.top_ratio))
            # Preserve the old image branch's sorted reduction order.
            score = flat.sort(dim=1, descending=True).values[:, :count].mean(dim=1)
        scores.append(score)
    return torch.cat(scores)
