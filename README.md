# Industrial Anomaly Image Detection

本项目面向 Omni-AD 工业图像异常检测任务，使用冻结的 DINOv2 `vit_base_patch14_reg4` 编码器和可训练的特征瓶颈、Transformer 解码器，从正常样本中学习可重建的视觉特征。推理时，模型通过编码特征与重建特征之间的余弦距离，同时生成图像级异常分数和像素级连续异常图。

项目遵循 `omniad-school-1.1` 接口，采用一个共享模型覆盖 30 个类别。训练和推理阶段均从本地加载依赖、预训练权重和模型文件，不需要访问网络。

**参赛队伍：** 无言以队

## 主要特性

- **单一共享模型：** 一个模型处理训练 manifest 中出现的全部类别，`model_mode` 为 `shared`。
- **无监督训练：** 训练数据必须来自各类别的 `train/good` 目录，模型训练过程不读取异常标签或缺陷掩膜。
- **冻结 DINOv2 主干：** 编码器保持冻结，仅训练 bottleneck 与 decoder。
- **保持宽高比：** 图像按比例缩放并 letterbox 填充至 `560 × 560`，推理后移除填充并将异常图恢复至原图尺寸。
- **图像与像素分支独立处理：** 像素分支生成连续定位图；图像分支在有效图像区域内进行特征组融合、平滑及 top 1% 汇聚。
- **难例挖掘：** 训练损失根据特征重建难度降低易重建 patch 的梯度贡献，并利用有效区域覆盖率排除纯填充 patch。
- **离线运行：** 模型不会在训练或推理期间下载权重。
- **中文路径兼容：** 使用 `np.fromfile` 与 `cv2.imdecode` 读取图像，可处理 Windows 下的中文路径。
- **规范化输出：** 每张图像输出有限的 `[0,1]` 图像级分数，以及与原图尺寸一致的二维 `float32` `.npy` 异常图。

## 项目分工

### 叶浩权

- 研究模型优化方案，分析预处理、特征融合、异常图生成、图像分数汇聚、难例挖掘和分数校准等环节对图像级及像素级指标的影响。
- 将优化方案落实到模型代码与配置中，完成相关参数调整、功能改造和版本验证。
- 编写模型测试文件与测试用例，检查数据读取、模型加载、训练、推理和预测输出是否符合接口要求。
- 编写本地评估脚本，实现 Image F1、Image AP、Pixel F1、Pixel AP、Pixel AUROC 和统一加权平均分的计算。

### 陈羿锦

- 完成项目需求分析、总体技术路线制定，以及模型最初版本和整体架构的设计与实现。
- 构建冻结 DINOv2 编码器、特征瓶颈和 Transformer 解码器组成的特征重建模型，实现图像级异常判别与像素级异常定位。
- 设计项目代码结构，完成 manifest 解析、基础图像预处理、模型构建、配置管理、随机种子控制和公共工具模块。
- 设计并实现训练入口、训练循环、优化器和学习率调度、模型保存，以及推理入口、离线模型加载和逐类别预测流程。
- 完成比赛评测接口与预测结果格式适配，负责依赖、预训练权重、模型目录和提交材料的整体整合。
- 指导模型优化方向和实验优先级，参与方案评审，并负责版本组织、任务协调、实验结果汇总、技术报告和最终交付检查。

## 项目结构

```text
Industrial-anomaly-image-detection/
├── configs/
│   └── default.json
├── model/
│   ├── shared.pth
│   ├── model_manifest.json
│   └── auxiliary/
│       ├── pretrained/
│       │   └── dinov2_vitb14_reg4_pretrain.pth
│       └── thresholds/
│           └── minmax.json
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── core/
│   ├── data/
│   ├── engine/
│   ├── models/
│   └── utils/
├── third_party/
│   └── LICENSES.md
├── pretrained_manifest.json
├── requirements.lock
├── report.pdf
├── submission.json
├── LICENSE
└── README.md
```

`model/` 是可直接用于推理的模型包。重新训练时，程序会在指定的输出目录中生成新的完整模型包，不允许直接覆盖项目自带的 `model/`。

## 环境安装

### 1. 创建并激活虚拟环境

在 PowerShell 中执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

如果 PowerShell 阻止激活脚本，可以执行：

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. 安装依赖

```powershell
pip install -r requirements.lock
```

当前锁定的主要版本包括：

| 依赖 | 版本 |
| --- | --- |
| Python | 3.14（生成当前 lock 文件的环境） |
| PyTorch | 2.14.0 |
| torchvision | 0.29.0 |
| NumPy | 2.5.2 |
| OpenCV | 5.0.0.93 |
| timm | 1.0.29 |

使用 NVIDIA GPU 时，需要确保安装的 PyTorch 构建支持本机 CUDA。安装后可运行：

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

第一项输出为 `True` 时，可在训练和推理命令中使用 `--device cuda:0`。

### 3. 检查预训练权重

训练所需的 DINOv2 编码器权重必须位于：

```text
model/auxiliary/pretrained/dinov2_vitb14_reg4_pretrain.pth
```

权重信息记录在 `pretrained_manifest.json` 中：

| 项目 | 内容 |
| --- | --- |
| 来源 | `https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14_reg4_pretrain.pth` |
| SHA256 | `73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71` |

校验命令：

```powershell
Get-FileHash "model\auxiliary\pretrained\dinov2_vitb14_reg4_pretrain.pth" -Algorithm SHA256
```

## Manifest 与数据路径

训练和推理都通过 CSV manifest 读取数据。`image_path` 是相对于 `--data-root` 的 POSIX 相对路径，必须使用 `/`，不能使用盘符、反斜杠、`..` 或绝对路径，也不能引用 `ground_truth`。

### 训练 manifest

按官方接口，训练 manifest 使用以下列：

- `image_name`
- `category`
- `image_path`

`image_name` 应为 `train/good/<文件名>`，每个 `image_path` 也必须指向相应类别的 `train/good` 正常图像，否则训练会直接报错。

示例：

```csv
image_name,category,image_path
train/good/000.png,air_conditioner_filter,air_conditioner_filter/train/good/000.png
```

### 推理 manifest

推理必须包含：

- `image_name`
- `category`
- `image_path`

`image_name` 必须采用 `test/<filename>` 的形式，并作为对应类别 `pred.json` 中的键。示例：

```csv
image_name,category,image_path
test/000.png,air_conditioner_filter,air_conditioner_filter/test/000.png
```

### `--data-root` 的确定方法

程序实际读取的文件路径为：

```text
<data-root>/<image_path>
```

例如，当 `--data-root` 为 `judge_input`、`image_path` 为 `air_conditioner_filter/train/good/000.png` 时，程序读取 `judge_input/air_conditioner_filter/train/good/000.png`。README、配置和源码均不固定数据集的实际位置，正式路径由组委会通过命令行传入。

## 训练

### 官方训练接口

```powershell
python -u src/train.py --data-root judge_input --manifest train_manifest.csv --output-dir runs/retrained_model --device cuda:0 --seed 2026 --num-workers 4
```

训练输出目录必须为空，且不能直接指定项目内置的 `model/`。如果目标目录已经包含文件，请改用一个新的目录名。

### 参数说明

| 参数 | 是否必填 | 说明 |
| --- | --- | --- |
| `--data-root` | 是 | 与 manifest 中 `image_path` 拼接的数据根目录 |
| `--manifest` | 是 | 训练 manifest 文件 |
| `--output-dir` | 是 | 新模型包的保存目录，必须为空 |
| `--device` | 是 | `cpu`、`mps`、`mps:N` 或 `cuda:N` |
| `--num-workers` | 否 | DataLoader 工作进程数，默认值为 4 |
| `--seed` | 是 | 随机种子，当前实验使用 2026 |

### 当前默认训练配置

| 配置 | 当前值 |
| --- | --- |
| 输入尺寸 | 保持宽高比并填充至 `560 × 560` |
| 填充值 | RGB `(124, 116, 104)` |
| 编码器 | `vit_base_patch14_reg4_dinov2`，冻结 |
| 编码器特征层 | 默认 `blocks.2` 至 `blocks.9` |
| bottleneck dropout | 0.3 |
| decoder depth | 6 |
| batch size | 2 |
| 梯度累积 | 4 |
| 最大优化步数 | 4000 |
| 优化器 | StableAdamW，初始学习率 0.002 |
| 学习率调度 | 100 步 warmup，余弦退火至 0.0002 |
| 像素图融合权重 | 0.65 / 0.35 |
| 像素图高斯平滑 | kernel 5，sigma 1.2 |
| 图像分数融合权重 | 0.5 / 0.5 |
| 图像分数汇聚 | 有效区域缩放至 `256 × 256`，平滑后取 top 1% 均值 |

需要修改训练参数时，请编辑 `configs/default.json`。

### 训练输出

```text
<output-dir>/
├── shared.pth
├── model_manifest.json
├── run_config.json
└── auxiliary/
    ├── pretrained/
    │   └── dinov2_vitb14_reg4_pretrain.pth
    └── thresholds/
        └── minmax.json
```

- `shared.pth`：模型参数、模型配置和随机种子。
- `model_manifest.json`：类别、模型结构、预处理和图像评分配置。
- `run_config.json`：本次训练参数、样本统计和训练 manifest 摘要。
- `minmax.json`：图像级分数与像素异常图的归一化参数。

## 推理

### 使用提交模型进行预测

在项目根目录执行：

```powershell
python -u src/predict.py --data-root judge_input --manifest eval_manifest.csv --model-dir model --output-dir runs/predictions --device cuda:0 --num-workers 4
```

`--output-dir` 必须是不存在或内容为空的目录。这项限制用于防止旧预测文件混入新结果。如果需要再次预测，请更换目录名，例如 `output_v2`。

### 使用重新训练的模型进行预测

将 `--model-dir` 改为训练生成的模型目录即可：

```powershell
python -u src/predict.py --data-root judge_input --manifest eval_manifest.csv --model-dir runs/retrained_model --output-dir runs/retrained_predictions --device cuda:0 --num-workers 4
```

### 参数说明

| 参数 | 是否必填 | 说明 |
| --- | --- | --- |
| `--data-root` | 是 | 与 manifest 中 `image_path` 拼接的数据根目录 |
| `--manifest` | 是 | 包含 `image_name`、`category` 和 `image_path` 的评测 manifest |
| `--model-dir` | 是 | 完整模型包目录 |
| `--output-dir` | 是 | 预测输出目录，必须为空 |
| `--device` | 是 | `cpu`、`mps`、`mps:N` 或 `cuda:N` |
| `--num-workers` | 否 | DataLoader 工作进程数，默认值为 4 |

### 预测输出

```text
<output-dir>/
└── <category>/
    ├── pred.json
    └── pred_maps/
        └── test/
            └── <filename>.npy
```

每个类别的 `pred.json` 以 manifest 中原始的 `image_name` 为键：

```json
{
  "test/example.png": {
    "anomaly_score": 0.73,
    "anomaly_map": "pred_maps/test/example.npy"
  }
}
```

输出要求：

- `anomaly_score` 必须是 `[0,1]` 范围内的有限浮点数，数值越大表示越可能异常。
- `anomaly_map` 指向相对于该类别目录的 `.npy` 文件。
- `.npy` 内容必须是二维 `float32` 数组，尺寸与原图一致，所有数值有限且位于 `[0,1]`。
- 输出样本数必须与评测 manifest 完全一致。

## 当前实验结果

以下结果均为 30 个类别的宏平均，分数采用 `[0,1]` 单位：

| 指标 | 当前结果 |
| --- | ---: |
| Image AUROC | 0.8896 |
| Image AP | 0.9331 |
| Image F1-max | 0.9130 |
| Pixel AUROC | 0.9685 |
| Pixel AP | 0.3791 |
| Pixel F1-max | 0.4281 |
| 统一加权平均分 | **0.6724（67.2%）** |

本地版本比较使用以下统一口径：

```text
0.25 × Pixel F1
+ 0.25 × Pixel AP
+ 0.15 × Image F1
+ 0.15 × Image AP
+ 0.20 × Pixel AUROC
```

正式比赛分还会受组委会 `metric_best` 归一化规则影响，本表用于本地版本比较。

## 离线运行与复现

- `submission.json` 中的 `network_required` 为 `false`。
- 编码器权重从模型目录本地加载，运行时不会调用在线模型仓库。
- 依赖版本固定在 `requirements.lock`。
- 默认实验随机种子为 2026，并写入训练输出；不同 GPU、驱动或底层算子仍可能造成细微数值差异。
- 如需在无网络环境安装依赖，应提前准备 wheelhouse，然后执行：

```powershell
pip install -r requirements.lock --no-index --find-links wheelhouse
```

## 常见问题

### 1. `评测 manifest 缺少必需列: image_name`

预测必须使用含 `image_name` 的评测 manifest，并确认表头包含：

```csv
image_name,category,image_path
```

训练 manifest 不能直接代替评测 manifest；评测清单中的 `image_name` 必须采用 `test/<文件名>` 格式。

### 2. 图像路径不存在或读取失败

检查相对路径 `<data-root>/<image_path>` 是否存在，并确认 CSV 内使用 `/` 作为路径分隔符。数据移动后只需调整命令行中的 `--data-root`，不应修改源码或配置来硬编码新位置。

### 3. `预测输出目录非空`

推理不会向非空目录写入结果。请指定一个新的输出目录，防止不同模型或不同轮次的预测文件混合。

### 4. 预训练编码器权重不存在

确认以下文件存在，并核对 `pretrained_manifest.json` 中的 SHA256：

```text
model/auxiliary/pretrained/dinov2_vitb14_reg4_pretrain.pth
```

### 5. CUDA 显存不足

- 在 `configs/default.json` 中继续减小 `training.batch_size`。
- 适当降低 `--num-workers` 只能减少数据加载进程，不会直接降低模型显存占用。
- 必要时使用 `--device cpu`，但训练和推理速度会显著下降。

### 6. 模型包不完整或配置不兼容

`--model-dir` 必须同时包含 `shared.pth`、`model_manifest.json`、预训练编码器权重和 `minmax.json`。旧模型若缺少当前预处理或图像评分配置，应使用当前代码重新训练生成完整模型包。

## 提交前检查

- 压缩包根目录应直接包含 `submission.json`、`README.md`、`requirements.lock`、`report.pdf`、`src/`、`configs/` 和 `model/`，不能再套一层同名目录。
- `model/` 必须包含断网推理需要的全部权重、配置和辅助文件；`pretrained_manifest.json` 不能代替实际权重文件。
- 不要把数据集、`ground_truth`、私有标签、已有预测结果、`.venv`、`__pycache__`、`.git` 或 IDE 缓存打入提交包。
- README、技术报告、源码和配置中不得出现本机盘符、用户目录、硬编码数据位置或院校名称与标识。
- `requirements.lock` 只能记录精确依赖版本，训练和推理代码不得动态执行安装命令。
- 应分别使用提交包内的 `model/` 和 `train.py` 新生成的模型目录完成一次 `predict.py` 验证。
- 每个类别的 `pred.json` 键、异常图文件和 manifest 样本必须一一对应，不得缺失、重复或新增样本。
- 需要在组委会环境中确认端到端单张推理时间不超过 100 ms，且推理峰值显存不超过 24 GB。
- Docker 镜像应按组委会后续发布的环境规范单独构建并验证。

## 许可与第三方代码

本项目用于学术研究与竞赛。项目引用 anomalib 和 DINOv2 的相关实现，两者均采用 Apache License 2.0。完整说明见 [third_party/LICENSES.md](third_party/LICENSES.md)。
