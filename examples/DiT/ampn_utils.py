"""Utilities for AMPN mixed-activation precision experiments."""
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant.replace import replace_linear_with_robuq


AMPN_OPS = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")
ADALN_OP = "adaLN_modulation.1"
ALL_AMPN_OPS = AMPN_OPS + (ADALN_OP,)

OP_COSTS = {
    "attn.qkv": 1.248,
    "attn.proj": 0.416,
    "mlp.fc1": 1.664,
    "mlp.fc2": 1.664,
    ADALN_OP: 0.009,
}


def canonical_op_name(op_name):
    if op_name in {"adaln", "adaLN_modulation[1]", "adaLN_modulation.1"}:
        return ADALN_OP
    return op_name


def module_name(block_idx, op_name):
    return f"blocks.{block_idx}.{canonical_op_name(op_name)}"


def get_submodule(root, name):
    module = root
    for part in name.split("."):
        module = getattr(module, part)
    return module


def set_submodule(root, name, new_module):
    parent_name, child_name = name.rsplit(".", 1)
    parent = get_submodule(root, parent_name)
    setattr(parent, child_name, new_module)


def replace_one_dit_linear(
    model,
    block_idx,
    op_name,
    a_bits,
    w_bits=1.58,
    if_hadamard=True,
    if_svd=True,
    svd_rank=16,
    svd_before_rotation=False,
    quant_impl="v1",
):
    name = module_name(block_idx, op_name)
    linear = get_submodule(model, name)
    new_module = replace_linear_with_robuq(
        linear,
        a_bits=a_bits,
        w_bits=w_bits,
        if_hadamard=if_hadamard,
        if_svd=if_svd,
        svd_rank=svd_rank,
        LN_in=False,
        LN_out=False,
        svd_before_rotation=svd_before_rotation,
        quant_impl=quant_impl,
    )
    new_module = new_module.to(device=linear.weight.device, dtype=linear.weight.dtype)
    set_submodule(model, name, new_module)
    return name


def normalize_mix_config(config):
    by_block = config.get("by_block", config)
    normalized = {}
    for block_key, ops in by_block.items():
        block_idx = int(str(block_key).replace("Block", ""))
        normalized[block_idx] = {}
        for op_name, value in ops.items():
            canonical = canonical_op_name(op_name)
            if isinstance(value, dict):
                bitwidth = int(value.get("bitwidth", value.get("bits")))
            else:
                bitwidth = int(value)
            normalized[block_idx][canonical] = bitwidth
    return normalized


def load_mix_config(path):
    with open(path, "r") as f:
        return normalize_mix_config(json.load(f))


def apply_mix_config(model, mix_config, verbose=True):
    applied = []
    for block_idx, ops in sorted(mix_config.items()):
        for op_name, bitwidth in sorted(ops.items()):
            name = module_name(block_idx, op_name)
            module = get_submodule(model, name)
            if not hasattr(module, "refresh_bits"):
                raise TypeError(f"{name} is {module.__class__.__name__}, expected quantized module with refresh_bits().")
            module.refresh_bits(a_bits=bitwidth)
            applied.append((name, bitwidth))
            if verbose:
                print(f"mix {name}: a_bits={bitwidth}")
    return applied


def flatten_solution(solution):
    rows = []
    for block_idx, ops in sorted(normalize_mix_config(solution).items()):
        for op_name, bitwidth in sorted(ops.items()):
            rows.append({"block_idx": block_idx, "op_name": op_name, "activation_bits": bitwidth})
    return rows


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
