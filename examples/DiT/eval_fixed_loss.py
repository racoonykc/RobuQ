"""Evaluate DiT loss on a fixed deterministic subset."""
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

from diffusers.models import AutoencoderKL
from diffusion import create_diffusion
from download import find_model
from models import DiT_models
from quant_utils import replace_dit_linear_layers
from train_qat import build_fixed_eval_cache, compute_fixed_eval_loss, default_data_path, default_weights_dir
from ampn_utils import apply_mix_config, load_mix_config


def parse_w_bits(value):
    if isinstance(value, float):
        return value
    normalized = str(value).lower().replace("w", "")
    if normalized in {"1.58", "1_58", "158"}:
        return 1.58
    return int(normalized)


def load_checkpoint_args(checkpoint):
    args = checkpoint.get("args", {})
    return args if isinstance(args, dict) else vars(args)


def resolve_pretrained(args):
    if args.pretrained_ckpt:
        return args.pretrained_ckpt
    local = Path(args.weights_dir) / f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    return str(local) if local.is_file() else f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"


def build_model(args, device):
    model = DiT_models[args.model](input_size=args.image_size // 8, num_classes=args.num_classes).to(device)
    if args.qat_ckpt:
        checkpoint = torch.load(args.qat_ckpt, map_location=device, weights_only=False)
        ckpt_args = load_checkpoint_args(checkpoint)
        args.w_bits = parse_w_bits(args.w_bits if args.w_bits is not None else ckpt_args.get("w_bits", 1.58))
        args.a_bits = int(args.a_bits if args.a_bits is not None else ckpt_args.get("a_bits", 4))
        args.hadamard = bool(args.hadamard if args.hadamard is not None else ckpt_args.get("hadamard", True))
        args.svd = bool(args.svd if args.svd is not None else ckpt_args.get("svd", True))
        args.svd_rank = int(args.svd_rank if args.svd_rank is not None else ckpt_args.get("svd_rank", 16))
        args.svd_before_rotation = bool(
            args.svd_before_rotation if args.svd_before_rotation is not None else ckpt_args.get("svd_before_rotation", False)
        )
        args.mix_config = args.mix_config or ckpt_args.get("mix_config")
        replace_dit_linear_layers(
            model,
            a_bits=args.a_bits,
            w_bits=args.w_bits,
            if_hadamard=args.hadamard,
            if_svd=args.svd,
            svd_rank=args.svd_rank,
            svd_before_rotation=args.svd_before_rotation,
            verbose=args.verbose_replace,
        )
        if args.mix_config:
            apply_mix_config(model, load_mix_config(args.mix_config), verbose=args.verbose_replace)
        state_key = "ema" if args.use_ema and "ema" in checkpoint else "model"
        model.load_state_dict(checkpoint[state_key], strict=False)
        print(f"Loaded {state_key} from QAT checkpoint {args.qat_ckpt}")
    else:
        if args.from_scratch:
            print("Using random initialization for fixed-loss smoke/debug run.")
        else:
            model.load_state_dict(find_model(resolve_pretrained(args)))
        if args.quantize:
            replace_dit_linear_layers(
                model,
                a_bits=args.a_bits,
                w_bits=parse_w_bits(args.w_bits),
                if_hadamard=args.hadamard,
                if_svd=args.svd,
                svd_rank=args.svd_rank,
                svd_before_rotation=args.svd_before_rotation,
                verbose=args.verbose_replace,
            )
            if args.mix_config:
                apply_mix_config(model, load_mix_config(args.mix_config), verbose=args.verbose_replace)
    return model


def main(args):
    configure_precision(args.force_fp32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, device)
    model.eval()
    diffusion = create_diffusion(timestep_respacing="")

    weights_dir = Path(args.weights_dir)
    default_vae_path = weights_dir / f"sd-vae-ft-{args.vae}"
    vae_path = args.vae_path or (str(default_vae_path) if default_vae_path.is_dir() else f"stabilityai/sd-vae-ft-{args.vae}")
    vae = AutoencoderKL.from_pretrained(vae_path).to(device)
    vae.eval()

    cache = build_fixed_eval_cache(args, vae, device, logger=type("Logger", (), {"info": print})())
    loss = compute_fixed_eval_loss(model, cache, diffusion, args.eval_batch_size)
    print(f"fixed_eval_loss={loss:.8f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=default_data_path())
    parser.add_argument("--eval-data-path", type=str, default=None)
    parser.add_argument("--eval-num-samples", type=int, default=1000)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--eval-num-workers", type=int, default=4)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--vae-path", type=str, default=None)
    parser.add_argument("--weights-dir", type=str, default=str(default_weights_dir()))
    parser.add_argument("--pretrained-ckpt", type=str, default=None)
    parser.add_argument("--qat-ckpt", type=str, default=None)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--force-fp32", action="store_true", help="Disable TF32 and use the highest available float32 precision.")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--w-bits", type=str, default="1.58")
    parser.add_argument("--a-bits", type=int, default=4)
    parser.add_argument("--hadamard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--svd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--svd-rank", type=int, default=16)
    parser.add_argument("--svd-before-rotation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mix-config", type=str, default=None)
    parser.add_argument("--verbose-replace", action="store_true")
    main(parser.parse_args())
