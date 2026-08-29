# Industrial Anomaly Image Detection

A fully offline, competition-ready implementation of **Dinomaly** for industrial anomaly detection, built on a frozen **DINOv2** (`vit_base_patch14_reg4`) encoder with a learnable bottleneck and decoder. The repository is prepared for the **OmniAD School** competition (`omniad-school-1.0` spec) and requires **no network access** during training or inference.

**Author:** Yijin Chen

## Features

- **Single shared model** trained across 30 industrial categories (`model_mode: shared`).
- **DINOv2-based** reconstruction architecture (encoder frozen, bottleneck + decoder trained only).
- **Offline-first**: all pretrained weights are bundled locally; nothing is downloaded at runtime.
- **Robust I/O**: images are read via `np.fromfile` + `cv2.imdecode`, so non-ASCII (e.g. Chinese) paths work correctly.
- **16-bit anomaly maps** plus normalized per-image anomaly scores in `[0, 1]`.
- Reproducible results via a fixed random seed (`2026`) and deterministic configuration.

## Project Structure

```
.
├── configs/
│   └── default.json            # Default model & training hyper-parameters
├── model/
│   ├── shared.pth              # Trained shared checkpoint
│   ├── model_manifest.json     # Model manifest (categories, score range, config)
│   └── auxiliary/
│       ├── pretrained/
│       │   └── dinov2_vitb14_reg4_pretrain.pth   # Frozen encoder weights
│       └── thresholds/
│           └── minmax.json     # Score normalization range [min, max]
├── src/
│   ├── train.py                # Competition training entry point
│   ├── predict.py              # Competition inference entry point
│   ├── core/                   # Config, args, manifest, seed, logging
│   ├── data/                   # Dataset & image transforms
│   ├── engine/                 # Training & inference engines
│   ├── models/                 # Dinomaly model implementation
│   └── utils/                  # Image I/O and score normalization
├── third_party/
│   └── LICENSES.md             # Third-party license notices
├── pretrained_manifest.json    # Source URL + sha256 of pretrained weights
├── submission.json             # Competition submission metadata
└── requirements.lock           # Pinned dependency list
```

## Installation

### 1. Create a virtual environment (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, temporarily relax the execution policy first:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. Install pinned dependencies from `requirements.lock`

```powershell
pip install -r requirements.lock
```

The lock file pins exact versions (e.g. `torch==2.13.0`, `torchvision==0.28.0`, `opencv-python==5.0.0.93`) so training results are reproducible.

> **Note:** `requirements.lock` was resolved with Python 3.14. If your Python version differs, install the packages manually and pin the same versions to match the lock file.

### 2b. (Optional) Install the CUDA build of PyTorch if you have an NVIDIA GPU

The wheels in `requirements.lock` are CPU builds by default. If your machine has an NVIDIA GPU with CUDA installed, install the **matching CUDA build** of `torch`/`torchvision` from the official PyTorch index for much faster training:

1. Check your installed CUDA version:

   ```powershell
   nvidia-smi
   ```

   Look at the CUDA version in the top-right corner of the `nvidia-smi` output.

2. Install the corresponding wheel variant from PyTorch's official index. Replace `cu124` with your CUDA version:

   ```powershell
   pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu124
   ```

   Commonly available index suffixes: `cu118` (CUDA 11.8), `cu121` (CUDA 12.1), `cu124` (CUDA 12.4), `cu126` (CUDA 12.6), and so on. Pick the one that matches the CUDA version reported by `nvidia-smi`.

3. Install the remaining pinned dependencies (pip will skip `torch`/`torchvision` since they are already satisfied by the CUDA wheels):

   ```powershell
   pip install -r requirements.lock
   ```

4. Verify that CUDA is available to PyTorch:

   ```powershell
   python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
   ```

   It should print `True` together with your CUDA version. Then pass `--device cuda:0` in the training/inference commands below.

### 3. Prepare the pretrained encoder weights (offline)

The training entry point **requires** the DINOv2 encoder checkpoint at:

```
model/auxiliary/pretrained/dinov2_vitb14_reg4_pretrain.pth
```

This file must be present **before** training. It is downloaded manually and bundled with the repository (see `pretrained_manifest.json`):

| Item | Value |
| --- | --- |
| Source URL | `https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth` |
| SHA256 | `0cefd5cc021528a63aa7c8e758e3800c06f48d18790d440970bfadcae1203ecd` |

Download it on a networked machine, place it at the path above, and verify the checksum (PowerShell):

```powershell
Get-FileHash model\auxiliary\pretrained\dinov2_vitb14_reg4_pretrain.pth -Algorithm SHA256
```

The output `Hash` value must equal `0cefd5cc021528a63aa7c8e758e3800c06f48d18790d440970bfadcae1203ecd`.

For a fully offline environment, install dependencies without network access:

```powershell
pip install -r requirements.lock --no-index --find-links C:\path\to\offline\wheelhouse
```

`submission.json` declares `"network_required": false`, so the competition environment will not have internet access — everything (dependencies and weights) must be prepared in advance.

## Data Format

Both training and inference consume a **manifest CSV** plus an image root directory.

Required columns:

- `category` — the anomaly category label
- `image_path` — path to the image, **relative to `--data-root`**

For inference (`strict=True`), the manifest must also contain:

- `sample_id` — unique sample identifier (used for output file names)

Example `manifest.csv`:

```csv
sample_id,category,image_path
0001,battery_piece,normal/battery_piece/0001.png
0002,battery_piece,normal/battery_piece/0002.png
```

Example directory layout:

```
<data-root>/
└── normal/
    └── battery_piece/
        └── 0001.png
```

## Training

### Command (PowerShell)

```powershell
python src/train.py `
  --data-root C:\path\to\data_root `
  --manifest C:\path\to\manifest.csv `
  --output-dir C:\path\to\output_dir `
  --device cuda:0 `
  --num-workers 4 `
  --seed 2026
```

| Argument | Description |
| --- | --- |
| `--data-root` | Root directory of the images (required) |
| `--manifest` | Training manifest CSV with `category` + `image_path` columns (required) |
| `--output-dir` | Directory where the trained model is saved (required) |
| `--device` | `cpu`, `mps`, or `cuda:N` (required) |
| `--num-workers` | Number of DataLoader worker processes (required) |
| `--seed` | Random seed for reproducibility (required) |

> All arguments are required by the CLI parser, so always pass them explicitly.

### Output

Training writes a complete model bundle to `--output-dir`:

```
output_dir/
├── shared.pth                       # Trained checkpoint
├── model_manifest.json              # Model manifest (categories, score range, config)
└── auxiliary/
    ├── pretrained/
    │   └── dinov2_vitb14_reg4_pretrain.pth
    └── thresholds/
        └── minmax.json              # Score range used for normalization
```

### Configuration

Hyper-parameters are read from `configs/default.json`:

- **Model**: DINOv2 base encoder, `decoder_depth=6`, `bottleneck_dropout=0.3`
- **Training**: `batch_size=8`, `max_steps=4000`, AdamW (`lr=2e-3`), warmup + cosine decay
- **Data**: images resized to `448×448`, center-cropped to `392×392`, ImageNet normalization

Adjust `batch_size` and `max_steps` in this file if you hit GPU memory limits or need shorter training.

## Inference

### Command (PowerShell)

```powershell
python src/predict.py `
  --data-root C:\path\to\data_root `
  --manifest C:\path\to\test_manifest.csv `
  --output-dir C:\path\to\predictions `
  --device cuda:0 `
  --num-workers 4 `
  --model-dir C:\path\to\model_dir
```

| Argument | Description |
| --- | --- |
| `--data-root` | Root directory of the images (required) |
| `--manifest` | Test manifest CSV with `sample_id` + `category` + `image_path` columns (required) |
| `--output-dir` | Directory where predictions are written (required) |
| `--device` | `cpu`, `mps`, or `cuda:N` (required) |
| `--num-workers` | Number of DataLoader worker processes (required) |
| `--model-dir` | Directory containing the trained model bundle (`model_manifest.json` + `shared.pth` + `auxiliary/...`) (required) |

The bundled `model/` directory is a ready-to-use model directory for this argument.

### Output

```
predictions/
├── maps/
│   └── <sample_id>.png          # 16-bit anomaly maps (resized to original size)
└── predictions.csv              # sample_id, image_score
```

- `image_score` is normalized to `[0, 1]` using the `min`/`max` stored in `minmax.json`. Higher score = more anomalous.
- Anomaly maps are 16-bit PNGs with values in `[0, 65535]`.

## Offline Running

The pipeline is fully offline. To guarantee zero network access:

1. **Dependencies**: bundle a local wheelhouse and install with `pip install -r requirements.lock --no-index --find-links <wheelhouse>`.
2. **Pretrained weights**: ensure `model/auxiliary/pretrained/dinov2_vitb14_reg4_pretrain.pth` is already in place before training.
3. **Disable any network calls**: set environment variables such as `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` as a safety net (no Hugging Face models are used by default).
4. **No download at runtime**: the code never queries the network; `submission.json` sets `"network_required": false`.

## Reproducibility

- Fixed seed `2026` (set via `src/core/seed.py` and stored in `submission.json`).
- All dependencies pinned in `requirements.lock`.
- Score range is computed on the training set and saved to `minmax.json`, then reused at inference for consistent normalization.

## FAQ

### 1. Out of GPU memory (CUDA OOM)

- Reduce `batch_size` in `configs/default.json` (e.g. from `8` to `4` or `2`).
- Reduce `--num-workers` (e.g. to `2` or `0`).
- Lower `max_steps` if the training loop does not fit your time budget.
- If you still run out of memory, switch to `--device cpu` (slower but works) or `--device mps` on Apple Silicon.

### 2. Chinese (non-ASCII) paths fail to load images

- All image loading uses `np.fromfile` + `cv2.imdecode`, which handle non-ASCII paths on Windows.
- Do **not** use `cv2.imread` directly — it fails on Chinese paths.
- If you see `图像读取失败` (image read failed), double-check that `--data-root`/`image_path` contain no typos and that the file exists.

### 3. `manifest 缺少必需列` (manifest is missing required columns)

- Training requires `category` and `image_path`.
- Inference additionally requires `sample_id`. Add it to your test manifest.

### 4. `预训练编码器权重不存在` (pretrained encoder weights not found)

- Training requires `model/auxiliary/pretrained/dinov2_vitb14_reg4_pretrain.pth`.
- Re-download it per `pretrained_manifest.json` and verify the SHA256 checksum.
- If you set `encoder_pretrained_path` in `configs/default.json`, make sure that path exists.

### 5. `device 非法` (invalid device)

- Allowed values: `cpu`, `mps`, `cuda:N` (e.g. `cuda:0`). Pass one of these explicitly.

### 6. Dependency installation is slow or blocked

- Use a mirror or local wheelhouse:
  ```powershell
  pip install -r requirements.lock --no-index --find-links C:\path\to\offline\wheelhouse
  ```
- The lock file pins exact versions; installing them from a pre-downloaded wheelhouse is fully offline.

### 7. `torch`/`torchvision` version mismatch

- `requirements.lock` pins `torch==2.13.0` and `torchvision==0.28.0`. They must be installed together from the same index. If the lock was resolved on Python 3.14, match that Python version or reinstall the same wheel versions on your interpreter.
- If you installed the CUDA build (see step 2b in Installation), make sure you did **not** accidentally reinstall the CPU wheels afterwards. Check with `python -c "import torch; print(torch.cuda.is_available())"` — it should print `True`.

### 8. `predictions.csv` sample count mismatch

- Inference raises an error if the number of predictions differs from the manifest. Ensure the manifest does not contain duplicate `sample_id` values and every row is valid.

## License

This project is for academic/competition use. It references third-party code from [anomalib](https://github.com/open-edge-platform/anomalib) and [DINOv2](https://github.com/facebookresearch/dinov2) (both Apache License 2.0). See [third_party/LICENSES.md](third_party/LICENSES.md) for full attribution.

---

**Author:** Yijin Chen
