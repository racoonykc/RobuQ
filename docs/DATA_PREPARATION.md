# Environment, Data, and Asset Preparation

The DiT example follows the original DiT environment setup, with one RobuQ-specific addition: `fast-hadamard-transform` is built from source inside the active PyTorch/CUDA environment.

## 0. Environment

Create the original DiT conda environment:

```bash
conda env create -f examples/DiT/environment.yml
conda activate DiT
```

Install the Python packages used by RobuQ training and evaluation:

```bash
pip install wandb scipy tensorflow tqdm huggingface_hub
```

Install `fast-hadamard-transform` from source. Direct `pip install fast-hadamard-transform` can fail when PyTorch and CUDA build versions do not exactly match the wheel. The recommended setup is:

```bash
mkdir -p third_party
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git third_party/fast-hadamard-transform
cd third_party/fast-hadamard-transform

# Build with the torch/CUDA from the currently activated environment.
pip install -e . --no-build-isolation --config-settings editable_mode=compat
cd ../..
```

If the build fails, check the active PyTorch/CUDA configuration:

```bash
python - <<'CHECK'
import torch
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('cuda_available:', torch.cuda.is_available())
CHECK
nvcc --version
```

The active PyTorch CUDA version and the local CUDA toolkit/compiler should be compatible.

The DiT example expects ImageNet, the official DiT-XL/2 checkpoint, a Stable Diffusion VAE, and optional ADM evaluation assets. RobuQ QAT checkpoints are released separately from git; see `docs/CHECKPOINTS.md`. The default paths below are relative to the repository root, but all launch scripts allow overrides through environment variables.

## 1. ImageNet-1K

Prepare ImageNet train split in torchvision `ImageFolder` format:

```text
data/imagenet/train/
  n01440764/
    ILSVRC2012_val_00000293.JPEG
    ...
  n01443537/
    ...
```

The default training path is:

```text
data/imagenet/train
```

Override it when needed:

```bash
DATA_PATH=/path/to/imagenet/train bash examples/DiT/entrypoints/train_w158a4_8gpu.sh
```

The fixed evaluation loss used during QAT is built from a deterministic subset of the same ImageFolder dataset unless `--eval-data-path` is supplied.

## 2. DiT-XL/2 256x256 Checkpoint

Download the official DiT-XL/2 ImageNet-256 checkpoint from the DiT release and place it at:

```text
weights/DiT-XL-2-256x256.pt
```

Direct URL:

```text
https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt
```

Example:

```bash
mkdir -p weights
wget -O weights/DiT-XL-2-256x256.pt \
  https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt
```

This is the full-precision base model used to initialize RobuQ QAT. The scripts use `WEIGHTS_DIR=weights` by default. To use another directory:

```bash
WEIGHTS_DIR=/path/to/weights bash examples/DiT/entrypoints/train_w158a4_8gpu.sh
```


## 3. RobuQ QAT Checkpoints

RobuQ checkpoints are not tracked by git. Packed inference checkpoints are released as GitHub Release assets; see `docs/CHECKPOINTS.md` for direct links. Download one into `weights/robuq/` or train locally with the entrypoints in `examples/DiT/entrypoints/`, then pass the checkpoint explicitly:

```bash
mkdir -p weights/robuq
wget -O weights/robuq/robuq_dit_xl2_imagenet256_w158a4_packed.pt \
  https://github.com/racoonykc/RobuQ/releases/download/robuq-dit-checkpoints-v1/robuq_dit_xl2_imagenet256_w158a4_packed.pt
python examples/DiT/sample_qat.py --ckpt weights/robuq/robuq_dit_xl2_imagenet256_w158a4_packed.pt
CKPT=weights/robuq/robuq_dit_xl2_imagenet256_w158a4_packed.pt bash examples/DiT/entrypoints/eval_qat_fid50k_adm_8gpu.sh
```

## 4. VAE Decoder

Sampling and evaluation use the Stable Diffusion VAE MSE variant by default. Put a local diffusers-style directory at:

```text
weights/sd-vae-ft-mse/
  config.json
  diffusion_pytorch_model.bin
  ...
```

The VAE can be prepared with:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="stabilityai/sd-vae-ft-mse",
    local_dir="weights/sd-vae-ft-mse",
    local_dir_use_symlinks=False,
)
PY
```

For offline clusters, this directory can be downloaded on a networked machine and copied into `weights/sd-vae-ft-mse`.

## 5. ADM / Guided-Diffusion Evaluation Assets

`eval_adm_metrics.py` wraps the ADM/guided-diffusion TensorFlow evaluator to compute FID, sFID, and Inception Score from `samples.npz`.

Expected default files:

```text
third_party/guided-diffusion/evaluations/evaluator.py
weights/adm/VIRTUAL_imagenet256_labeled.npz
weights/adm/classify_image_graph_def.pb
```

Suggested setup:

```bash
mkdir -p third_party weights/adm
git clone https://github.com/openai/guided-diffusion.git third_party/guided-diffusion
```

Download or copy the ADM ImageNet-256 reference npz and Inception graph into `weights/adm/`. If your assets live elsewhere, override paths:

```bash
CKPT=/path/to/ckpt.pt \
ADM_EVALUATOR=/path/to/guided-diffusion/evaluations/evaluator.py \
ADM_REF=/path/to/VIRTUAL_imagenet256_labeled.npz \
ADM_GRAPH=/path/to/classify_image_graph_def.pb \
bash examples/DiT/entrypoints/eval_qat_fid50k_adm_8gpu.sh
```

## 6. Recommended Directory Check

Before running training/evaluation, the following should exist:

```bash
test -f weights/DiT-XL-2-256x256.pt
test -d weights/sd-vae-ft-mse
test -d data/imagenet/train
```

For RobuQ checkpoint sampling/evaluation, also provide `--ckpt weights/robuq/<checkpoint>.pt` or set `CKPT=weights/robuq/<checkpoint>.pt`.

For ADM metrics:

```bash
test -f weights/adm/VIRTUAL_imagenet256_labeled.npz
test -f weights/adm/classify_image_graph_def.pb
test -f third_party/guided-diffusion/evaluations/evaluator.py
```
