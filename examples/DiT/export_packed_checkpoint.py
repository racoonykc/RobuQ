#!/usr/bin/env python3
"""Export a RobuQ training checkpoint to a compact packed inference checkpoint."""
import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant.packed import PACKED_FORMAT, cast_floating_tensors



def checkpoint_args(checkpoint):
    args = checkpoint.get("args", {})
    return args if isinstance(args, dict) else vars(args)


def parse_w_bits(value):
    if isinstance(value, float):
        return value
    normalized = str(value).lower().replace("w", "")
    if normalized in {"1.58", "1_58", "158"}:
        return 1.58
    return int(normalized)


def resolve_arg(cli_value, ckpt_args, key, default):
    return cli_value if cli_value is not None else ckpt_args.get(key, default)

def dtype_from_name(name):
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype {name}")


def main(args):
    from models import DiT_models
    from quant_utils import replace_dit_linear_layers
    from ampn_utils import apply_mix_config, load_mix_config
    from quant.quantized_modules.robuq import RobuQLinear

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint_args(ckpt)
    model_name = resolve_arg(args.model, ckpt_args, "model", "DiT-XL/2")
    image_size = int(resolve_arg(args.image_size, ckpt_args, "image_size", 256))
    num_classes = int(resolve_arg(args.num_classes, ckpt_args, "num_classes", 1000))
    w_bits = parse_w_bits(resolve_arg(args.w_bits, ckpt_args, "w_bits", 1.58))
    a_bits = int(resolve_arg(args.a_bits, ckpt_args, "a_bits", 4))
    hadamard = bool(resolve_arg(args.hadamard, ckpt_args, "hadamard", True))
    svd = bool(resolve_arg(args.svd, ckpt_args, "svd", True))
    svd_rank = int(resolve_arg(args.svd_rank, ckpt_args, "svd_rank", 16))
    svd_before_rotation = bool(resolve_arg(args.svd_before_rotation, ckpt_args, "svd_before_rotation", False))
    quant_impl = resolve_arg(args.quant_impl, ckpt_args, "quant_impl", "v1")
    mix_config = args.mix_config or ckpt_args.get("mix_config")

    model = DiT_models[model_name](input_size=image_size // 8, num_classes=num_classes)
    replace_dit_linear_layers(
        model,
        a_bits=a_bits,
        w_bits=w_bits,
        if_hadamard=hadamard,
        if_svd=svd,
        svd_rank=svd_rank,
        svd_before_rotation=svd_before_rotation,
        quant_impl=quant_impl,
        init_svd=False,
        verbose=False,
    )
    if mix_config:
        apply_mix_config(model, load_mix_config(mix_config), verbose=False)

    state_key = args.state_key
    if state_key == "auto":
        state_key = "ema" if args.use_ema and "ema" in ckpt else "model"
    state = ckpt[state_key]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded {state_key}: missing={len(missing)}, unexpected={len(unexpected)}")

    release_dtype = dtype_from_name(args.dtype)
    scale_dtype = dtype_from_name(args.scale_dtype)
    packed_linears = {}
    remove_keys = set()
    for name, module in model.named_modules():
        if isinstance(module, RobuQLinear) and module.w_bits == 1.58:
            packed_linears[name] = module.export_packed_weight(scale_dtype=scale_dtype)
            remove_keys.add(f"{name}.weight")

    model_state = model.state_dict()
    model_state = {k: v.detach().cpu() for k, v in model_state.items() if k not in remove_keys}
    if args.dtype != "float32":
        model_state = cast_floating_tensors(model_state, release_dtype)

    out = {
        "format": PACKED_FORMAT,
        "model": model_state,
        "packed_linears": packed_linears,
        "args": {
            **ckpt_args,
            "model": model_name,
            "image_size": image_size,
            "num_classes": num_classes,
            "w_bits": w_bits,
            "a_bits": a_bits,
            "hadamard": hadamard,
            "svd": svd,
            "svd_rank": svd_rank,
            "svd_before_rotation": svd_before_rotation,
            "quant_impl": quant_impl,
            "mix_config": mix_config,
            "packed_format": PACKED_FORMAT,
            "release_dtype": args.dtype,
            "scale_dtype": args.scale_dtype,
        },
        "source_state_key": state_key,
        "train_steps": ckpt.get("train_steps"),
        "epoch": ckpt.get("epoch"),
    }
    if mix_config:
        import json
        with open(mix_config, "r") as f:
            out["mix_config_data"] = json.load(f)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output)
    in_size = Path(args.ckpt).stat().st_size / (1024**3)
    out_size = output.stat().st_size / (1024**3)
    print(f"Packed {len(packed_linears)} RobuQLinear weights")
    print(f"input:  {in_size:.3f} GiB")
    print(f"output: {out_size:.3f} GiB -> {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state-key", default="auto", choices=["auto", "ema", "model"])
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--scale-dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--w-bits", type=str, default=None)
    parser.add_argument("--a-bits", type=int, default=None)
    parser.add_argument("--hadamard", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--svd", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--svd-rank", type=int, default=None)
    parser.add_argument("--svd-before-rotation", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--quant-impl", type=str, default=None, choices=[None, "v1"])
    parser.add_argument("--mix-config", type=str, default=None)
    main(parser.parse_args())
