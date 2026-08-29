# Third-Party Licenses

本项目在开发过程中参考或使用了以下第三方开源代码。我们遵守其原始许可证要求，并在下方列出声明。

---

## 1. anomalib

- **来源**: https://github.com/open-edge-platform/anomalib
- **许可证**: Apache License 2.0
- **使用内容**: Dinomaly 模型的核心实现，包括 `torch_model.py`、`components/layers.py`、`components/loss.py`、`components/optimizer.py` 以及部分 DINOv2 层实现。
- **修改说明**: 移除了 PyTorch Lightning 依赖，将模型改为纯 `nn.Module`；移除了 anomalib 内部的 Batch 数据结构和预处理模块；调整了文件结构和导入路径以适配比赛接口。

## 2. DINOv2

- **来源**: https://github.com/facebookresearch/dinov2
- **许可证**: Apache License 2.0
- **使用内容**: 注意力层（`attention.py`）、DropPath（`drop_path.py`）、LayerScale（`layer_scale.py`）等底层组件。
- **修改说明**: 仅保留 Dinomaly 所需的几个组件，删除了其他未使用的层。这些文件在 anomalib 中也有类似实现，但源头可追溯至 DINOv2 官方代码。

## 其他说明

- 所有第三方代码仅用于学术竞赛目的。
- 本项目保留对自行编写部分的版权，第三方代码版权归原作者所有。
- 如对本声明有任何疑问，请联系项目维护者。