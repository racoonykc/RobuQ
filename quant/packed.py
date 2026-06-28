"""Packed RobuQ checkpoint helpers.

The packed format stores W1.58 linear weights as uint2 codes. Codes map
0 -> -1, 1 -> 0, 2 -> +1, with code 3 unused. Four values are stored in
one uint8. Scales are stored separately and dequantized when loading for
standard PyTorch inference.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


PACKED_FORMAT = "robuq_packed_v1"


def pack_ternary_uint2(discrete: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Pack a tensor with values in {-1, 0, 1} into uint2 codes."""
    codes = (discrete.to(torch.int16) + 1).clamp_(0, 2).to(torch.uint8).flatten()
    original_numel = int(codes.numel())
    pad = (-original_numel) % 4
    if pad:
        codes = F.pad(codes, (0, pad), value=1)
    codes = codes.view(-1, 4)
    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=codes.device)
    packed = torch.sum((codes << shifts).to(torch.uint8), dim=1).to(torch.uint8)
    return packed.cpu().contiguous(), original_numel


def unpack_ternary_uint2(packed: torch.Tensor, shape: tuple[int, ...], original_numel: int | None = None) -> torch.Tensor:
    """Unpack uint2 codes back to {-1, 0, 1} int8 tensor."""
    flat = packed.to(torch.uint8).flatten()
    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=flat.device)
    codes = ((flat[:, None] >> shifts) & 0x03).flatten()
    expected = int(torch.tensor(shape).prod().item())
    if original_numel is None:
        original_numel = expected
    if int(original_numel) < expected:
        raise ValueError(f"original_numel={original_numel} is smaller than requested shape numel={expected}.")
    codes = codes[:expected]
    return (codes.to(torch.int8) - 1).view(*shape)


def _counts_for_groups(in_features: int, num_groups: int, group_size: int, *, device, dtype) -> torch.Tensor:
    counts = torch.full((num_groups,), group_size, device=device, dtype=dtype)
    pad_features = num_groups * group_size - in_features
    if pad_features:
        counts[-1] = group_size - pad_features
    return counts


def ternary_discrete_and_scale(weight: torch.Tensor, group_size: int | None = None, epsilon: float = 1e-8):
    """Return ternary codes and BitNet scales matching RobuQ training quantization."""
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D linear weight, got shape {tuple(weight.shape)}.")
    if group_size is None:
        scale = weight.abs().mean(dim=1, keepdim=True)
        discrete = torch.round(weight / (scale + epsilon)).clamp(-1, 1).to(torch.int8)
        return discrete, scale.detach()

    group_size = int(group_size)
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")
    out_features, in_features = weight.shape
    num_groups = (in_features + group_size - 1) // group_size
    padded_features = num_groups * group_size
    pad_features = padded_features - in_features
    if pad_features:
        grouped = F.pad(weight, (0, pad_features)).view(out_features, num_groups, group_size)
    else:
        grouped = weight.view(out_features, num_groups, group_size)
    counts = _counts_for_groups(in_features, num_groups, group_size, device=weight.device, dtype=weight.dtype)
    scale = grouped.abs().sum(dim=2, keepdim=True) / counts.view(1, num_groups, 1)
    discrete = torch.round(grouped / (scale + epsilon)).clamp(-1, 1)
    discrete = discrete.view(out_features, padded_features)[:, :in_features].to(torch.int8)
    return discrete, scale.detach()


def dequantize_packed_linear_weight(payload: dict[str, Any], *, device=None, dtype=None) -> torch.Tensor:
    shape = tuple(payload["shape"])
    group_size = payload.get("group_size")
    packed = payload["packed_weight"].to(device=device or payload["packed_weight"].device)
    discrete = unpack_ternary_uint2(packed, shape, payload.get("original_numel")).to(device=packed.device)
    scale = payload["scale"].to(device=packed.device, dtype=dtype or torch.float32)
    discrete_f = discrete.to(dtype=dtype or scale.dtype)
    if group_size is None:
        return discrete_f * scale

    out_features, in_features = shape
    group_size = int(group_size)
    num_groups = (in_features + group_size - 1) // group_size
    padded_features = num_groups * group_size
    pad_features = padded_features - in_features
    if pad_features:
        discrete_f = F.pad(discrete_f, (0, pad_features)).view(out_features, num_groups, group_size)
    else:
        discrete_f = discrete_f.view(out_features, num_groups, group_size)
    weight = discrete_f * scale
    return weight.view(out_features, padded_features)[:, :in_features]


def make_packed_linear_payload(weight: torch.Tensor, group_size: int | None = None, scale_dtype: torch.dtype = torch.float16) -> dict[str, Any]:
    discrete, scale = ternary_discrete_and_scale(weight.detach(), group_size=group_size)
    packed, original_numel = pack_ternary_uint2(discrete.cpu())
    return {
        "packed_weight": packed,
        "original_numel": original_numel,
        "shape": tuple(weight.shape),
        "group_size": group_size,
        "scale": scale.cpu().to(scale_dtype),
        "scale_dtype": str(scale_dtype).replace("torch.", ""),
    }


def apply_packed_weights(model: torch.nn.Module, packed_linears: dict[str, dict[str, Any]], *, dtype=None, strict: bool = True):
    """Load packed weights into RobuQLinear modules by module name."""
    modules = dict(model.named_modules())
    loaded = []
    missing = []
    for name, payload in packed_linears.items():
        module = modules.get(name)
        if module is None or not hasattr(module, "load_packed_weight"):
            missing.append(name)
            continue
        module.load_packed_weight(payload, dtype=dtype)
        loaded.append(name)
    if strict and missing:
        raise KeyError(f"Could not find packed RobuQLinear modules: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    return {"loaded": loaded, "missing": missing}


def cast_floating_tensors(obj: Any, dtype: torch.dtype) -> Any:
    """Recursively cast floating tensors in a checkpoint payload."""
    if torch.is_tensor(obj):
        return obj.to(dtype=dtype) if obj.is_floating_point() else obj
    if isinstance(obj, dict):
        return {k: cast_floating_tensors(v, dtype) for k, v in obj.items()}
    if isinstance(obj, list):
        return [cast_floating_tensors(v, dtype) for v in obj]
    if isinstance(obj, tuple):
        return tuple(cast_floating_tensors(v, dtype) for v in obj)
    return obj


def is_packed_checkpoint(checkpoint: Any) -> bool:
    return isinstance(checkpoint, dict) and checkpoint.get("format") == PACKED_FORMAT
