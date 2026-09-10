from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


class CosineHardMiningLoss(torch.nn.Module):
    """Patch-wise cosine reconstruction loss with per-image hard mining.

    The encoder is frozen. Each spatial feature location is scored independently
    against the decoder output. Letterbox padding can be excluded, or partly
    weighted, through ``valid_patch_mask``.
    """

    def __init__(
        self,
        p_final: float = 0.8,
        p_schedule_steps: int = 1000,
        easy_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if not 0.0 <= p_final <= 1.0:
            raise ValueError("p_final must be in [0, 1]")
        if not isinstance(p_schedule_steps, int) or p_schedule_steps <= 0:
            raise ValueError("p_schedule_steps must be a positive integer")
        if not 0.0 <= easy_weight <= 1.0:
            raise ValueError("easy_weight must be in [0, 1]")
        self.p_final = float(p_final)
        self.p_schedule_steps = p_schedule_steps
        self.easy_weight = float(easy_weight)
        self.p = 0.0

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "CosineHardMiningLoss":
        config = dict(config or {})
        loss_type = config.pop("type", "spatial_cosine_hard_mining")
        mask_mode = config.pop("valid_mask_mode", "patch_coverage")
        if loss_type != "spatial_cosine_hard_mining":
            raise ValueError(f"unsupported loss type: {loss_type!r}")
        if mask_mode != "patch_coverage":
            raise ValueError(f"unsupported valid_mask_mode: {mask_mode!r}")
        allowed = {"p_final", "p_schedule_steps", "easy_weight"}
        unexpected = set(config) - allowed
        if unexpected:
            raise ValueError(f"unsupported loss configuration: {sorted(unexpected)}")
        return cls(**config)

    def forward(
        self,
        encoder_features: list[torch.Tensor],
        decoder_features: list[torch.Tensor],
        global_step: int,
        valid_patch_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not encoder_features or len(encoder_features) != len(decoder_features):
            raise ValueError("encoder_features and decoder_features must be non-empty lists of equal length")

        self._update_p_schedule(global_step)
        total_loss = torch.zeros((), device=encoder_features[0].device)
        for encoder_feature, decoder_feature in zip(encoder_features, decoder_features, strict=True):
            if encoder_feature.shape != decoder_feature.shape or encoder_feature.ndim != 4:
                raise ValueError(
                    "encoder and decoder features must have identical [B, C, H, W] shapes; "
                    f"got {tuple(encoder_feature.shape)} and {tuple(decoder_feature.shape)}"
                )
            coverage = self._validate_or_create_mask(valid_patch_mask, encoder_feature)
            distances = 1.0 - F.cosine_similarity(
                encoder_feature.detach(), decoder_feature, dim=1
            )
            weights = self._mining_weights(distances, coverage)
            denominator = weights.sum()
            if denominator <= 0:
                raise ValueError("valid_patch_mask contains no valid patch")
            total_loss = total_loss + (distances * weights).sum() / denominator

        return total_loss / len(encoder_features)

    @staticmethod
    def _validate_or_create_mask(
        valid_patch_mask: torch.Tensor | None,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, height, width = feature.shape
        if valid_patch_mask is None:
            return torch.ones((batch_size, height, width), dtype=feature.dtype, device=feature.device)
        if valid_patch_mask.shape != (batch_size, 1, height, width):
            raise ValueError(
                "valid_patch_mask shape must be [B, 1, H, W] matching feature patches; "
                f"got {tuple(valid_patch_mask.shape)}, expected {(batch_size, 1, height, width)}"
            )
        coverage = valid_patch_mask[:, 0].to(device=feature.device, dtype=feature.dtype)
        if not torch.isfinite(coverage).all() or (coverage < 0).any() or (coverage > 1).any():
            raise ValueError("valid_patch_mask values must be finite and in [0, 1]")
        return coverage

    def _mining_weights(self, distances: torch.Tensor, coverage: torch.Tensor) -> torch.Tensor:
        """Return coverage-aware hard-mining weights, independently per image."""
        hard = torch.zeros_like(distances, dtype=torch.bool)
        flat_distances = distances.reshape(distances.shape[0], -1)
        flat_coverage = coverage.reshape(coverage.shape[0], -1)
        flat_hard = hard.reshape(hard.shape[0], -1)

        for batch_index in range(distances.shape[0]):
            valid_indices = torch.nonzero(flat_coverage[batch_index] > 0, as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                continue
            keep_count = max(1, math.ceil(valid_indices.numel() * (1.0 - self.p)))
            candidate_distances = flat_distances[batch_index, valid_indices]
            hard_indices = valid_indices[torch.topk(candidate_distances, k=keep_count).indices]
            flat_hard[batch_index, hard_indices] = True

        mining = torch.where(hard, torch.ones_like(coverage), torch.full_like(coverage, self.easy_weight))
        return coverage * mining

    def _update_p_schedule(self, global_step: int) -> None:
        self.p = min(self.p_final * global_step / self.p_schedule_steps, self.p_final)
