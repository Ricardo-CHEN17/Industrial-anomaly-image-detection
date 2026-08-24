# src/models/dinomaly/model.py
from __future__ import annotations

import torch
from torch.nn.init import trunc_normal_

from .torch_model import DinomalyModel


def _initialize_trainable_modules(trainable_modules: torch.nn.ModuleList) -> None:
    """对可训练模块进行截断正态初始化，与 anomalib 原始实现一致。"""
    for m in trainable_modules.modules():
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)


def build_dinomaly(
    encoder_name: str = "vit_base_patch14_reg4_dinov2",
    bottleneck_dropout: float = 0.2,
    decoder_depth: int = 8,
    target_layers: list[int] | None = None,
    fuse_layer_encoder: list[list[int]] | None = None,
    fuse_layer_decoder: list[list[int]] | None = None,
    remove_class_token: bool = False,
    use_context_recentering: bool = False,
    precision: str = "float32",
    encoder_pretrained_path: str | None = None,
) -> DinomalyModel:
    """构建 Dinomaly 模型，并完成参数冻结/解冻及初始化。

    训练时只更新 bottleneck 和 decoder 的参数，编码器保持冻结。
    """
    model = DinomalyModel(
        encoder_name=encoder_name,
        bottleneck_dropout=bottleneck_dropout,
        decoder_depth=decoder_depth,
        target_layers=target_layers,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
        remove_class_token=remove_class_token,
        use_context_recentering=use_context_recentering,
        encoder_pretrained_path=encoder_pretrained_path,
    )

    if precision == "float16":
        model = model.to(torch.bfloat16)
    elif precision == "float32":
        model = model.float()
    else:
        raise ValueError(f"Unsupported precision type: {precision}")

    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    # 解冻 bottleneck 和 decoder
    for param in model.bottleneck.parameters():
        param.requires_grad = True
    for param in model.decoder.parameters():
        param.requires_grad = True

    trainable_modules = torch.nn.ModuleList([model.bottleneck, model.decoder])
    _initialize_trainable_modules(trainable_modules)

    return model