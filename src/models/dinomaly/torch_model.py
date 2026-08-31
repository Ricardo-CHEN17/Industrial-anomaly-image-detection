from __future__ import annotations

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from timm.models import create_model
from timm.layers.drop import DropPath

from .components.layers import DinomalyMLP, LinearAttention
from .components.loss import CosineHardMiningLoss


# -----------------------------------------------------------------------------
# Custom utility classes (replacing anomalib dependencies)
# -----------------------------------------------------------------------------

@dataclass
class InferenceBatch:
    """Simple container for inference outputs, mimicking anomalib's InferenceBatch."""
    pred_score: torch.Tensor
    anomaly_map: torch.Tensor

    def update(self, **kwargs) -> "InferenceBatch":
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

# 初始化高斯模糊层，用于平滑热力图
class GaussianBlur2d(nn.Module):
    """Fixed-weight Gaussian blur module using depthwise convolution."""
    def __init__(self, sigma: float, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.sigma = sigma
        self.channels = channels
        self.kernel_size = kernel_size
        # Create a 2D Gaussian kernel
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        # Reshape to (channels, 1, kernel_size, kernel_size)
        kernel = kernel.repeat(channels, 1, 1, 1)
        self.register_buffer("kernel", kernel)
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=self.padding,
            groups=channels,
            bias=False,
        )
        # Set the fixed weights
        with torch.no_grad():
            self.conv.weight.copy_(kernel)
        # Freeze
        self.conv.weight.requires_grad = False

    @staticmethod
    def _create_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
        ax = torch.arange(kernel_size).float() - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, kernel_size, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x, self.kernel, padding=self.padding, groups=self.channels
        )


class TimmFeatureExtractor(nn.Module):
    """Lightweight wrapper around timm models to extract intermediate features.

    Args:
        backbone: timm model name (e.g. "vit_base_patch14_reg4_dinov2")
        layers: list of layer names to extract (e.g. ["blocks.2", "blocks.3", ...])
        pretrained_path: path to local pretrained weights file (.pth or .safetensors)
        return_class_token: if True, keep the class token in the output
        requires_grad: whether the encoder parameters require gradients
        output_fmt: "NLC" (we always store as NLC internally)
        norm: if True, apply LayerNorm to outputs (not implemented here)
        dynamic_img_size: if True, allow dynamic input sizes (not used)
    """
    def __init__(
        self,
        backbone: str,
        layers: list[str],
        pretrained_path: str | None = None,
        return_class_token: bool = True,
        requires_grad: bool = False,
        output_fmt: str = "NLC",
        norm: bool = False,
        dynamic_img_size: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.layers = layers
        self.return_class_token = return_class_token
        self.output_fmt = output_fmt
        self.norm = norm
        self.dynamic_img_size = dynamic_img_size

        # Create timm model
        self.model = create_model(
            backbone,
            pretrained=False,
            dynamic_img_size=dynamic_img_size,
        )

        # Load local pretrained weights if provided
        if pretrained_path is not None:
            state_dict = self._load_pretrained_state_dict(pretrained_path)
            self.model.load_state_dict(state_dict, strict=True)
        else:
            raise ValueError("pretrained_path must be provided for offline inference.")

        # Freeze parameters if not requires_grad
        for param in self.model.parameters():
            param.requires_grad = requires_grad

        # Set patch size and register tokens properties
        self.patch_size = self.model.patch_embed.patch_size[0]
        num_prefix = getattr(self.model, "num_prefix_tokens", 1)
        self.num_register_tokens = num_prefix - 1

        # Register forward hooks to extract intermediate features
        self._features: dict[str, torch.Tensor] = {}
        self._hook_handles = []
        for layer_name in self.layers:
            module = self.model.get_submodule(layer_name)
            handle = module.register_forward_hook(
                lambda module, input, output, name=layer_name: self._features.__setitem__(name, output)
            )
            self._hook_handles.append(handle)

    @staticmethod
    def _load_pretrained_state_dict(pretrained_path: str) -> dict[str, torch.Tensor]:
        """Load a pretrained checkpoint and adapt official DINOv2 keys to timm format.

        Official DINOv2 checkpoints use ``register_tokens``/``mask_token`` and include
        the class-token position in ``pos_embed``, while timm's ViT model expects
        ``reg_token`` and a patch-only ``pos_embed``.
        """
        state_dict = torch.load(pretrained_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        if "register_tokens" in state_dict or "mask_token" in state_dict:
            state_dict = dict(state_dict)
            if "register_tokens" in state_dict:
                state_dict["reg_token"] = state_dict.pop("register_tokens")
            state_dict.pop("mask_token", None)
            pos_embed = state_dict.get("pos_embed")
            if pos_embed is not None and pos_embed.shape[1] == 1370:
                state_dict["pos_embed"] = pos_embed[:, 1:]
        return state_dict

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self._features.clear()
        _ = self.model(x)
        # Return features in the order they were requested
        return {name: self._features[name] for name in self.layers}

    def __del__(self):
        # Remove hooks to avoid memory leaks
        for handle in self._hook_handles:
            handle.remove()


# -----------------------------------------------------------------------------
# Original constants and model
# -----------------------------------------------------------------------------

DINO_ARCHITECTURES = {
    "small": {"embed_dim": 384, "num_heads": 6, "target_layers": [2, 3, 4, 5, 6, 7, 8, 9]},
    "base": {"embed_dim": 768, "num_heads": 12, "target_layers": [2, 3, 4, 5, 6, 7, 8, 9]},
    "large": {"embed_dim": 1024, "num_heads": 16, "target_layers": [4, 6, 8, 10, 12, 14, 16, 18]},
    "huge": {"embed_dim": 1280, "num_heads": 20, "target_layers": [3, 9, 12, 15, 18, 21, 24, 27]},
}

def _derive_fuse_layers(n_features: int) -> list[list[int]]:
    """根据特征层数自动切分两组融合层索引。

    与原始 8 层默认值 [[0,1,2,3],[4,5,6,7]] 保持兼容：
    8 层 → [[0,1,2,3],[4,5,6,7]]；6 层 → [[0,1,2],[3,4,5]]。
    """
    if n_features <= 1:
        return [list(range(n_features))]
    mid = (n_features + 1) // 2
    return [list(range(mid)), list(range(mid, n_features))]


def _validate_fuse_layers(
    fuse_layers: list[list[int]], n_features: int, name: str
) -> None:
    """校验融合层索引是否在有效范围内，避免越界导致 list index out of range。"""
    if not fuse_layers:
        raise ValueError(f"{name} 不能为空")
    for group in fuse_layers:
        for idx in group:
            if not isinstance(idx, int) or isinstance(idx, bool) or idx < 0 or idx >= n_features:
                raise ValueError(
                    f"{name} 中的索引 {idx!r} 超出有效范围 [0, {n_features - 1}]"
                    f"（当前特征层数为 {n_features}，请检查 decoder_depth / target_layers 与 fuse 配置是否匹配）"
                )


DEFAULT_RESIZE_SIZE = 256
DEFAULT_GAUSSIAN_KERNEL_SIZE = 5
DEFAULT_GAUSSIAN_SIGMA = 4
DEFAULT_MAX_RATIO = 0.01

TRANSFORMER_CONFIG: dict[str, float | bool] = {
    "mlp_ratio": 4.0,
    "layer_norm_eps": 1e-8,
    "qkv_bias": True,
    "attn_drop": 0.0,
}


class DinomalyModel(nn.Module):
    def __init__(
        self,
        encoder_name: str = "vit_base_patch14_reg4_dinov2",
        bottleneck_dropout: float = 0.2,
        decoder_depth: int = 8,
        target_layers: list[int] | None = None,
        fuse_layer_encoder: list[list[int]] | None = None,
        fuse_layer_decoder: list[list[int]] | None = None,
        remove_class_token: bool = False,
        use_context_recentering: bool = False,
        encoder_pretrained_path: str | None = None,
    ) -> None:
        super().__init__()

        if use_context_recentering and remove_class_token:
            raise ValueError(
                "use_context_recentering=True requires access to the class token "
                "and is incompatible with remove_class_token=True"
            )

        arch_config = self._get_architecture_config(encoder_name, target_layers)
        embed_dim = arch_config["embed_dim"]
        num_heads = arch_config["num_heads"]

        self.target_layers = arch_config["target_layers"] if target_layers is None else target_layers

        self.fuse_layer_encoder = (
            _derive_fuse_layers(len(self.target_layers))
            if fuse_layer_encoder is None
            else fuse_layer_encoder
        )
        self.fuse_layer_decoder = (
            _derive_fuse_layers(decoder_depth)
            if fuse_layer_decoder is None
            else fuse_layer_decoder
        )
        _validate_fuse_layers(
            self.fuse_layer_encoder, len(self.target_layers), "fuse_layer_encoder"
        )
        _validate_fuse_layers(self.fuse_layer_decoder, decoder_depth, "fuse_layer_decoder")
        if len(self.fuse_layer_encoder) != len(self.fuse_layer_decoder):
            raise ValueError(
                f"fuse_layer_encoder 与 fuse_layer_decoder 的分组数必须一致: "
                f"{len(self.fuse_layer_encoder)} vs {len(self.fuse_layer_decoder)}"
            )

        # Create encoder (our custom TimmFeatureExtractor)
        self.encoder = TimmFeatureExtractor(
            backbone=encoder_name,
            layers=[f"blocks.{i}" for i in self.target_layers],
            pretrained_path=encoder_pretrained_path,
            return_class_token=True,
            requires_grad=False,
            output_fmt="NLC",
            norm=False,
            dynamic_img_size=True,
        )

        if decoder_depth <= 1:
            raise ValueError(f"decoder_depth must be greater than 1, got {decoder_depth}")

        bottleneck = [
            DinomalyMLP(
                in_features=embed_dim,
                hidden_features=embed_dim * 4,
                out_features=embed_dim,
                act_layer=nn.GELU,
                drop=bottleneck_dropout,
                bias=False,
                apply_input_dropout=True,
            )
        ]
        self.bottleneck = nn.ModuleList(bottleneck)

        decoder = []
        for _ in range(decoder_depth):
            mlp_ratio_val = TRANSFORMER_CONFIG["mlp_ratio"]
            qkv_bias_val = TRANSFORMER_CONFIG["qkv_bias"]
            layer_norm_eps_val = TRANSFORMER_CONFIG["layer_norm_eps"]
            attn_drop_val = TRANSFORMER_CONFIG["attn_drop"]

            decoder_block = DecoderViTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio_val,
                qkv_bias=qkv_bias_val,
                norm_layer=partial(nn.LayerNorm, eps=layer_norm_eps_val),
                attn_drop=attn_drop_val,
                attn=LinearAttention,
            )
            decoder.append(decoder_block)
        self.decoder = nn.ModuleList(decoder)

        self.remove_class_token = remove_class_token
        self.use_context_recentering = use_context_recentering

        self.gaussian_blur = GaussianBlur2d(
            sigma=DEFAULT_GAUSSIAN_SIGMA,
            channels=1,
            kernel_size=DEFAULT_GAUSSIAN_KERNEL_SIZE,
        )

        self.loss_fn = CosineHardMiningLoss()

    def get_encoder_decoder_outputs(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        h_patches = x.shape[2] // self.encoder.patch_size
        w_patches = x.shape[3] // self.encoder.patch_size

        features = self.encoder(x)
        encoder_features = [features[f"blocks.{i}"] for i in self.target_layers]
        decoder_features = []

        if self.remove_class_token:
            encoder_features = [
                e[:, 1 + self.encoder.num_register_tokens :, :] for e in encoder_features
            ]
        elif self.use_context_recentering:
            recentered = []
            for e in encoder_features:
                cls_token = e[:, 0:1, :]
                patch_start = 1 + self.encoder.num_register_tokens
                patches = e[:, patch_start:, :] - cls_token
                recentered.append(patches)
            encoder_features = recentered

        x = self._fuse_feature(encoder_features)
        for block in self.bottleneck:
            x = block(x)

        for block in self.decoder:
            x = block(x, attn_mask=None)
            decoder_features.append(x)
        decoder_features = decoder_features[::-1]

        en = [self._fuse_feature([encoder_features[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self._fuse_feature([decoder_features[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        en = self._process_features_for_spatial_output(en, h_patches, w_patches)
        de = self._process_features_for_spatial_output(de, h_patches, w_patches)
        return en, de

    def forward(self, batch: torch.Tensor, global_step: int | None = None) -> torch.Tensor | InferenceBatch:
        dtype = next(self.encoder.parameters()).dtype
        batch = batch.type(dtype)
        en, de = self.get_encoder_decoder_outputs(batch)
        image_size = (batch.shape[2], batch.shape[3])

        if self.training:
            if global_step is None:
                raise ValueError("global_step must be provided during training")
            return self.loss_fn(encoder_features=en, decoder_features=de, global_step=global_step)

        anomaly_map, _ = self.calculate_anomaly_maps(en, de, out_size=image_size)
        anomaly_map_resized = anomaly_map.clone()

        if DEFAULT_RESIZE_SIZE is not None:
            anomaly_map = F.interpolate(anomaly_map, size=DEFAULT_RESIZE_SIZE, mode="bilinear", align_corners=False)

        anomaly_map = self.gaussian_blur(anomaly_map)

        if DEFAULT_MAX_RATIO == 0:
            sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
        else:
            anomaly_map_flat = anomaly_map.flatten(1)
            k = int(anomaly_map_flat.shape[1] * DEFAULT_MAX_RATIO)
            sp_score = torch.sort(anomaly_map_flat, dim=1, descending=True)[0][:, :k].mean(dim=1)
        pred_score = sp_score

        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map_resized)

    @staticmethod
    def calculate_anomaly_maps(
        source_feature_maps: list[torch.Tensor],
        target_feature_maps: list[torch.Tensor],
        out_size: int | tuple[int, int] = 392,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if not isinstance(out_size, tuple):
            out_size = (out_size, out_size)

        anomaly_map_list = []
        for i in range(len(target_feature_maps)):
            fs = source_feature_maps[i]
            ft = target_feature_maps[i]
            a_map = 1 - F.cosine_similarity(fs, ft)
            a_map = torch.unsqueeze(a_map, dim=1)
            a_map = F.interpolate(a_map, size=out_size, mode="bilinear", align_corners=True)
            anomaly_map_list.append(a_map)
        anomaly_map = torch.cat(anomaly_map_list, dim=1).mean(dim=1, keepdim=True)
        return anomaly_map, anomaly_map_list

    @staticmethod
    def _fuse_feature(feat_list: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(feat_list, dim=1).mean(dim=1)

    @staticmethod
    def _get_architecture_config(encoder_name: str, target_layers: list[int] | None) -> dict:
        for arch_name, config in DINO_ARCHITECTURES.items():
            if arch_name in encoder_name:
                result = config.copy()
                if target_layers is not None:
                    result["target_layers"] = target_layers
                return result
        raise ValueError(
            f"Architecture not supported. Encoder name must contain one of {list(DINO_ARCHITECTURES.keys())}"
        )

    def _process_features_for_spatial_output(
        self,
        features: list[torch.Tensor],
        h_patches: int,
        w_patches: int,
    ) -> list[torch.Tensor]:
        # Remove class token and register tokens if not already removed.
        # We compute the expected number of patch tokens and take the last N tokens.
        if not self.remove_class_token and not self.use_context_recentering:
            n_patches = h_patches * w_patches
            features = [f[:, -n_patches:, :] for f in features]   # <-- 修改这里

        # Reshape to spatial dimensions
        batch_size = features[0].shape[0]
        return [f.permute(0, 2, 1).reshape([batch_size, -1, h_patches, w_patches]).contiguous() for f in features]

class DecoderViTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float | None = None,
        qkv_bias: bool | None = None,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        attn: type[nn.Module] = LinearAttention,
    ) -> None:
        super().__init__()

        mlp_ratio_config = TRANSFORMER_CONFIG["mlp_ratio"]
        qkv_bias_config = TRANSFORMER_CONFIG["qkv_bias"]
        attn_drop_config = TRANSFORMER_CONFIG["attn_drop"]

        mlp_ratio = mlp_ratio if mlp_ratio is not None else mlp_ratio_config
        qkv_bias = qkv_bias if qkv_bias is not None else qkv_bias_config
        attn_drop = attn_drop if attn_drop is not None else attn_drop_config

        self.norm1 = norm_layer(dim)
        self.attn = attn(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = DinomalyMLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
            drop=drop,
            apply_input_dropout=False,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if attn_mask is not None:
            y, attn = self.attn(self.norm1(x), attn_mask=attn_mask)
        else:
            y, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        return x