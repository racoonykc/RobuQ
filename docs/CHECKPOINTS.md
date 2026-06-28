# Checkpoints

RobuQ checkpoints are intentionally kept outside git. Packed inference checkpoints are distributed as GitHub Release assets under:

```text
https://github.com/racoonykc/RobuQ/releases/tag/robuq-dit-checkpoints-v1
```

Direct download links:

| Setting | File |
|---|---|
| W1.58A4 | [robuq_dit_xl2_imagenet256_w158a4_packed.pt](https://github.com/racoonykc/RobuQ/releases/download/robuq-dit-checkpoints-v1/robuq_dit_xl2_imagenet256_w158a4_packed.pt) |
| W1.58A3 AMPN | [robuq_dit_xl2_imagenet256_w158a3_ampn_packed.pt](https://github.com/racoonykc/RobuQ/releases/download/robuq-dit-checkpoints-v1/robuq_dit_xl2_imagenet256_w158a3_ampn_packed.pt) |
| W1.58A2 AMPN | [robuq_dit_xl2_imagenet256_w158a2_ampn_packed.pt](https://github.com/racoonykc/RobuQ/releases/download/robuq-dit-checkpoints-v1/robuq_dit_xl2_imagenet256_w158a2_ampn_packed.pt) |

The packed checkpoint files are about 232 MiB each. They were exported from the 100K-step QAT checkpoints and verified to be bitwise-equivalent to the quantized DiT forward path.

Example download:

```bash
mkdir -p weights/robuq
wget -O weights/robuq/robuq_dit_xl2_imagenet256_w158a4_packed.pt \
  https://github.com/racoonykc/RobuQ/releases/download/robuq-dit-checkpoints-v1/robuq_dit_xl2_imagenet256_w158a4_packed.pt
```

Sample or evaluate a downloaded checkpoint:

```bash
python examples/DiT/sample_qat.py --ckpt weights/robuq/robuq_dit_xl2_imagenet256_w158a4_packed.pt
CKPT=weights/robuq/robuq_dit_xl2_imagenet256_w158a4_packed.pt bash examples/DiT/entrypoints/eval_qat_fid50k_adm_8gpu.sh
```

The README table reports the paper's 8-GPU QAT FID-50K setting: DiT-XL/2 on ImageNet-1K 256x256, batch size 256, learning rate 1e-4, 100K QAT steps, DDPM sampler with 50 steps, cfg=1.5, and 50K samples.

## Reproducing Checkpoints

To reproduce the QAT checkpoints locally, use the training entrypoints:

```bash
bash examples/DiT/entrypoints/train_w158a4_8gpu.sh
bash examples/DiT/entrypoints/collect_ampn_metrics_8gpu.sh
bash examples/DiT/entrypoints/allocate_ampn_a2.sh
bash examples/DiT/entrypoints/allocate_ampn_a3.sh
bash examples/DiT/entrypoints/train_ampn_a2_8gpu.sh
bash examples/DiT/entrypoints/train_ampn_a3_8gpu.sh
```

## Packed Release Format

Training checkpoints include optimizer state and full-precision RobuQ weights, so they are much larger than needed for inference. Export release checkpoints with the packed format before uploading:

```bash
python examples/DiT/export_packed_checkpoint.py \
  --ckpt /path/to/training/checkpoints/0100000.pt \
  --output /path/to/robuq_dit_xl2_imagenet256_w158a4_packed.pt \
  --dtype float32 \
  --scale-dtype float32
```

For AMPN A2/A3 checkpoints, pass the corresponding mix config so it is embedded in the release file:

```bash
python examples/DiT/export_packed_checkpoint.py \
  --ckpt /path/to/ampn_a2/checkpoints/0100000.pt \
  --mix-config /path/to/IP1_ampn_a2_config.json \
  --output /path/to/robuq_dit_xl2_imagenet256_w158a2_ampn_packed.pt \
  --dtype float32 \
  --scale-dtype float32
```

The packed checkpoint uses `format = robuq_packed_v1`. W1.58 linear weights are stored as uint2 ternary codes plus per-channel scale tensors. On load, `sample_qat.py` and `eval_qat_fid.py` reconstruct the quantized weights into the existing RobuQLinear modules and keep activation Hadamard/quantization online. This avoids storing optimizer state and avoids storing full-precision copies of the ternary main weights.
