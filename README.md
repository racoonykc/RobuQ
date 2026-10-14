# RobuQ: Pushing DiTs to W1.58A2 via Robust Activation Quantization

**Official PyTorch implementation of RobuQ**, a quantization-aware training (QAT) framework for Diffusion Transformers (DiTs).

[Kaicheng Yang](https://github.com/racoonykc), [Xun Zhang](https://github.com/xunzhang1128), [Haotong Qin](https://github.com/htqin), [Yucheng Lin](https://github.com/catyugu), [Kaisen Yang](https://github.com/yks23), [Xianglong Yan](https://github.com/XIANGLONGYAN), and [Yulun Zhang](https://github.com/yulunzhang)

[[Paper](https://arxiv.org/abs/2509.23582)] [[Data Preparation](docs/DATA_PREPARATION.md)] [[DiT Example](examples/DiT)] [[Checkpoints](docs/CHECKPOINTS.md)]

## News

- Initial public release: RobuQ quantization modules, DiT-XL/2 ImageNet-256 QAT scripts, AMPN allocation tools, and FID-50K evaluation utilities.
- RobuQ packed inference checkpoints are distributed through the GitHub Release assets linked below.

## Overview

Diffusion Transformers have strong generative quality but are expensive to deploy. RobuQ targets ultra-low-bit DiT quantization under QAT. It first builds a strong W1.58A4 baseline with ternary weights, all-layer Hadamard mixing, and a rank-16 SVD low-rank branch. It then introduces a robust activation quantizer and AMPN, an activation-only mixed-precision pipeline that assigns layer-wise activation bits while keeping selected DiT weights ternary.

This repository currently focuses on DiT-XL/2 ImageNet-256 reproduction. Embedding layers, conditioning layers, attention score matmuls, and the final projection are kept at higher precision where the paper found them sensitive or negligible in cost.

## Highlights

- **Ternary DiT weights.** RobuQ supports W1.58 quantization for DiT transformer block linear layers.
- **Robust activation quantization.** The public implementation uses the clean RobuQ `v1` activation quantizer with a stable STE path.
- **Hadamard and SVD components.** `RobuQLinear` includes Hadamard rotation and an optional rank-16 SVD residual branch.
- **AMPN mixed activation precision.** The DiT example includes QAT-based sensitivity collection and dynamic-programming allocation for W1.58A2 and W1.58A3.
- **FID-50K evaluation.** Distributed sampling and ADM/guided-diffusion metric evaluation are provided for FID, sFID, and Inception Score.

## Results

FID-50K results below are from the 8-GPU QAT setting in the paper: DiT-XL/2 on ImageNet-1K 256x256, batch size 256, learning rate 1e-4, 100K QAT steps, DDPM sampler with 50 steps, classifier-free guidance scale 1.5, and 50K generated samples.

| Setting | W/A | FID↓ | sFID↓ | IS↑ | Checkpoint |
|---|---:|---:|---:|---:|---|
| FP DiT-XL/2 | 32/32 | 3.71 | 7.85 | 242.21 | [official DiT](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt) |
| RobuQ (w/o AMP) | 1.58/4 | 4.55 | 7.26 | 194.96 | [download](https://github.com/racoonykc/RobuQ/releases/download/robuq-dit-checkpoints-v1/robuq_dit_xl2_imagenet256_w158a4_packed.pt) |
| RobuQ | 1.58/3 | 5.04 | 7.85 | 187.27 | [download](https://github.com/racoonykc/RobuQ/releases/download/robuq-dit-checkpoints-v1/robuq_dit_xl2_imagenet256_w158a3_ampn_packed.pt) |
| RobuQ | 1.58/2 | 8.47 | 10.24 | 145.90 | [download](https://github.com/racoonykc/RobuQ/releases/download/robuq-dit-checkpoints-v1/robuq_dit_xl2_imagenet256_w158a2_ampn_packed.pt) |

## Visualizations

![RobuQ-W1.58A3 256x256 images from quantized DiT-XL/2 on ImageNet-1K](figs/A3_visualization.svg)

![RobuQ-W1.58A2 256x256 images from quantized DiT-XL/2 on ImageNet-1K](figs/A2_visualization.svg)

![RobuQ consistently outperforms previous ultra-low-bit DiT quantization baselines](figs/compare.svg)

## Repository Structure

```text
RobuQ/
  quant/
    replace.py
    quantized_modules/
      robuq.py
      hadamard_utils.py
  examples/DiT/
    train_qat.py
    sample_qat.py
    eval_qat_fid.py
    eval_adm_metrics.py
    collect_ampn_metrics.py
    allocate_ampn.py
    entrypoints/
  docs/
    DATA_PREPARATION.md
    CHECKPOINTS.md
```

## Installation

RobuQ follows the original DiT environment setup and adds dependencies for fast Hadamard transforms and ADM evaluation.

```bash
conda env create -f examples/DiT/environment.yml
conda activate DiT
pip install wandb scipy tensorflow tqdm huggingface_hub
```

Install `fast-hadamard-transform` from source inside the activated environment:

```bash
mkdir -p third_party
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git third_party/fast-hadamard-transform
cd third_party/fast-hadamard-transform
pip install -e . --no-build-isolation --config-settings editable_mode=compat
cd ../..
```

Direct `pip install fast-hadamard-transform` can fail when PyTorch and CUDA versions do not match the available wheel. Building from source uses the active environment's PyTorch/CUDA configuration.

## Data and Checkpoints

Large datasets, pretrained checkpoints, VAE weights, RobuQ checkpoints, and ADM evaluation assets are not tracked by git. Prepare the following assets before running experiments:

```text
data/imagenet/train/                 # ImageNet-1K train split in ImageFolder format
weights/DiT-XL-2-256x256.pt          # official DiT-XL/2 ImageNet-256 checkpoint
weights/sd-vae-ft-mse/               # local diffusers VAE directory
weights/adm/                         # ADM reference npz and Inception graph
third_party/guided-diffusion/         # ADM/guided-diffusion evaluator
```

See [docs/DATA_PREPARATION.md](docs/DATA_PREPARATION.md) for setup instructions and [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md) for RobuQ checkpoint download and release details.

## DiT-XL/2 Reproduction

All commands below are launched from the repository root. The default scripts assume 8 GPUs and ImageNet-256.

### W1.58A4 Baseline

```bash
bash examples/DiT/entrypoints/train_w158a4_8gpu.sh
```

### AMPN Metric Collection and Allocation

```bash
bash examples/DiT/entrypoints/collect_ampn_metrics_8gpu.sh
bash examples/DiT/entrypoints/allocate_ampn_a2.sh
bash examples/DiT/entrypoints/allocate_ampn_a3.sh
```

### W1.58A2 / W1.58A3 Mixed-Precision QAT

```bash
bash examples/DiT/entrypoints/train_ampn_a2_8gpu.sh
bash examples/DiT/entrypoints/train_ampn_a3_8gpu.sh
```

### FID-50K Evaluation

```bash
CKPT=weights/robuq/robuq_dit_xl2_imagenet256_w158a4_packed.pt bash examples/DiT/entrypoints/eval_qat_fid50k_adm_8gpu.sh
```

The evaluation entrypoint first uses `eval_qat_fid.py --mode sample` to generate samples and build `samples.npz`, then calls `eval_adm_metrics.py` to compute FID, sFID, and Inception Score with the ADM/guided-diffusion evaluator.

## Quantization Interface

The reusable entry point is `replace_linear_with_robuq`:

```python
from quant import replace_linear_with_robuq
```

The DiT example applies RobuQ to transformer block linear layers by default. Patch embedding, conditioning embeddings, and the final projection remain full precision unless explicitly modified.

The public activation quantizer name is:

```text
quant_impl = v1
```

## Citation

```bibtex
@inproceedings{yang2026robuq,
  title={RobuQ: Pushing DiTs to W1.58A2 via Robust Activation Quantization},
  author={Yang, Kaicheng and Zhang, Xun and Qin, Haotong and Lin, Yucheng and Yang, Kaisen and Yan, Xianglong and Zhang, Yulun},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  year={2026}
}
```

## Acknowledgements

This repository builds on [DiT](https://github.com/facebookresearch/DiT), [fast-hadamard-transform](https://github.com/Dao-AILab/fast-hadamard-transform), and ADM/guided-diffusion evaluation utilities. We retain attribution comments in files adapted from DiT and OpenAI diffusion repositories.
