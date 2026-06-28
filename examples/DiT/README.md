# RobuQ DiT Example

This example contains the DiT-XL/2 ImageNet-256 RobuQ recipe used for the public release. It follows the original DiT codebase for model definition, sampling, and diffusion utilities, then adds RobuQ QAT, AMPN allocation, and ADM/FID-50K evaluation wrappers.

## Main Scripts

- `train_qat.py`: QAT for W1.58A4 and AMPN mixed activation precision.
- `sample_qat.py`: sample from a QAT checkpoint.
- `collect_ampn_metrics.py`: collect layer sensitivity metrics for AMPN.
- `allocate_ampn.py`: solve A2/A3 mixed activation configurations from collected metrics.
- `eval_qat_fid.py`: distributed sampling and `samples.npz` generation.
- `eval_adm_metrics.py`: ADM/guided-diffusion FID, sFID, and Inception Score.

## Default Quantization Recipe

The public recipe uses:

```text
w_bits = 1.58
a_bits = 4 for the baseline
quant_impl = v1
hadamard = true
svd = true
svd_rank = 16
```

AMPN keeps W1.58 weights for all selected linear layers and allocates activation bit-width per layer to target average A2 or A3. The released packed A2/A3 checkpoints embed their AMPN configs. For local training, generate `results_ampn/dit_xl2_w158_metrics_t1000/ampn_a2_config.json` or `ampn_a3_config.json` with the metric collection and allocation commands below.

## Entrypoints

Run from the repository root:

```bash
bash examples/DiT/entrypoints/train_w158a4_8gpu.sh
bash examples/DiT/entrypoints/collect_ampn_metrics_8gpu.sh
bash examples/DiT/entrypoints/allocate_ampn_a2.sh
bash examples/DiT/entrypoints/allocate_ampn_a3.sh
bash examples/DiT/entrypoints/train_ampn_a2_8gpu.sh
bash examples/DiT/entrypoints/train_ampn_a3_8gpu.sh
```

Sample or evaluate a checkpoint:

```bash
python examples/DiT/sample_qat.py --ckpt weights/robuq/robuq_dit_xl2_imagenet256_w158a4_packed.pt
CKPT=weights/robuq/robuq_dit_xl2_imagenet256_w158a4_packed.pt bash examples/DiT/entrypoints/eval_qat_fid50k_adm_8gpu.sh
```

See `../../docs/DATA_PREPARATION.md` for data, VAE, ADM metric assets, and full-precision DiT checkpoint setup. See `../../docs/CHECKPOINTS.md` for RobuQ checkpoint download and release details.
