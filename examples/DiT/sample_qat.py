# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Sample images from a QAT RobuQ DiT checkpoint.
"""
import argparse
from pathlib import Path
import sys

import torch

def configure_precision(force_fp32: bool):
    use_tf32 = not force_fp32
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    if force_fp32 and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from models import DiT_models
from quant_utils import replace_dit_linear_layers
from ampn_utils import apply_mix_config, load_mix_config, normalize_mix_config
from quant.packed import apply_packed_weights, is_packed_checkpoint

def default_weights_dir():
    return REPO_ROOT / "weights"


def parse_w_bits(value):
    if isinstance(value, float):
        return value
    normalized = str(value).lower().replace("w", "")
    if normalized in {"1.58", "1_58", "158"}:
        return 1.58
    return int(normalized)


def get_rng_state(device):
    state = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def set_rng_state(state, device):
    torch.set_rng_state(state["cpu"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device)


def checkpoint_args(checkpoint):
    args = checkpoint.get("args", {})
    return args if isinstance(args, dict) else vars(args)


def resolve_arg(cli_value, ckpt_args, key, default):
    return cli_value if cli_value is not None else ckpt_args.get(key, default)


def default_output_path(args, ckpt_step):
    ckpt_name = Path(args.ckpt).stem
    model_name = args.model.replace("/", "-")
    precision = f"w{args.w_bits}a{args.a_bits}"
    return Path("outputs") / f"{model_name}-qat-{precision}-{ckpt_name}-steps{args.num_sampling_steps}-seed{args.seed}.png"


def load_qat_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected a training checkpoint dict at {path}.")
    return checkpoint


def main(args):
    configure_precision(args.force_fp32)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = load_qat_checkpoint(args.ckpt, device)
    ckpt_args = checkpoint_args(checkpoint)
    ckpt_step = checkpoint.get("train_steps", Path(args.ckpt).stem)

    args.model = resolve_arg(args.model, ckpt_args, "model", "DiT-XL/2")
    args.image_size = int(resolve_arg(args.image_size, ckpt_args, "image_size", 256))
    args.num_classes = int(resolve_arg(args.num_classes, ckpt_args, "num_classes", 1000))
    args.w_bits = parse_w_bits(resolve_arg(args.w_bits, ckpt_args, "w_bits", 1.58))
    args.a_bits = int(resolve_arg(args.a_bits, ckpt_args, "a_bits", 4))
    args.hadamard = bool(resolve_arg(args.hadamard, ckpt_args, "hadamard", True))
    args.svd = bool(resolve_arg(args.svd, ckpt_args, "svd", True))
    args.svd_rank = int(resolve_arg(args.svd_rank, ckpt_args, "svd_rank", 16))
    args.svd_before_rotation = bool(resolve_arg(args.svd_before_rotation, ckpt_args, "svd_before_rotation", False))
    args.quant_impl = resolve_arg(args.quant_impl, ckpt_args, "quant_impl", "v1")
    args.mix_config = resolve_arg(args.mix_config, ckpt_args, "mix_config", None)
    packed_checkpoint = is_packed_checkpoint(checkpoint)

    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    ).to(device)

    rng_state = get_rng_state(device)
    quant_report = replace_dit_linear_layers(
        model,
        a_bits=args.a_bits,
        w_bits=args.w_bits,
        if_hadamard=args.hadamard,
        if_svd=args.svd,
        svd_rank=args.svd_rank,
        svd_before_rotation=args.svd_before_rotation,
        quant_impl=args.quant_impl,
        verbose=args.verbose_replace,
    )
    set_rng_state(rng_state, device)
    print(f"Quantized DiT linear layers: {quant_report['num_replaced']}")
    if checkpoint.get("mix_config_data") is not None:
        applied = apply_mix_config(model, normalize_mix_config(checkpoint["mix_config_data"]), verbose=args.verbose_replace)
        print(f"Applied embedded AMPN mix config: {len(applied)} layers")
    elif args.mix_config:
        applied = apply_mix_config(model, load_mix_config(args.mix_config), verbose=args.verbose_replace)
        print(f"Applied AMPN mix config from {args.mix_config}: {len(applied)} layers")

    state_key = "model" if packed_checkpoint else ("ema" if args.use_ema and "ema" in checkpoint else "model")
    if state_key not in checkpoint:
        raise KeyError(f"Checkpoint does not contain '{state_key}'. Available keys: {list(checkpoint.keys())}")
    missing, unexpected = model.load_state_dict(checkpoint[state_key], strict=False)
    if packed_checkpoint:
        packed_report = apply_packed_weights(model, checkpoint["packed_linears"], dtype=model.x_embedder.proj.weight.dtype)
        print(f"Loaded packed RobuQ weights: {len(packed_report['loaded'])} layers")
    if missing or unexpected:
        print(f"Loaded with missing={len(missing)}, unexpected={len(unexpected)}")
    print(f"Loaded {state_key} from {args.ckpt} (step={ckpt_step})")

    model.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps))

    weights_dir = Path(args.weights_dir)
    default_vae_path = weights_dir / f"sd-vae-ft-{args.vae}"
    vae_path = args.vae_path or (str(default_vae_path) if default_vae_path.is_dir() else f"stabilityai/sd-vae-ft-{args.vae}")
    vae_kwargs = {"local_files_only": True} if Path(str(vae_path)).is_dir() else {}
    vae = AutoencoderKL.from_pretrained(vae_path, **vae_kwargs).to(device)
    vae.eval()

    class_labels = args.class_labels or [207, 360, 387, 974, 88, 979, 417, 279]
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    z = torch.cat([z, z], 0)
    y_null = torch.tensor([args.num_classes] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    samples = diffusion.p_sample_loop(
        model.forward_with_cfg,
        z.shape,
        z,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=True,
        device=device,
    )
    samples, _ = samples.chunk(2, dim=0)
    samples = vae.decode(samples / 0.18215).sample

    output = Path(args.output) if args.output else default_output_path(args, ckpt_step)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image(samples, output, nrow=min(4, n), normalize=True, value_range=(-1, 1))
    print(f"Saved samples to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to a RobuQ QAT checkpoint.")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default=None)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights-dir", type=str, default=str(default_weights_dir()))
    parser.add_argument("--vae-path", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--w-bits", type=str, default=None)
    parser.add_argument("--a-bits", type=int, default=None)
    parser.add_argument("--hadamard", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--svd", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--svd-rank", type=int, default=None)
    parser.add_argument("--svd-before-rotation", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--quant-impl", type=str, default=None, choices=[None, "v1"])
    parser.add_argument("--force-fp32", action="store_true", help="Disable TF32 and use the highest available float32 precision.")
    parser.add_argument("--mix-config", type=str, default=None)
    parser.add_argument("--verbose-replace", action="store_true")
    parser.add_argument("--class-labels", type=int, nargs="*", default=None)
    args = parser.parse_args()
    main(args)
