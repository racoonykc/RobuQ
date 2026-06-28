import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from .hadamard_utils import get_hadK, matmul_hadU, matmul_hadU_cuda
from quant.packed import dequantize_packed_linear_weight, make_packed_linear_payload


THEORETICAL_OPTIMAL_TABLE = {
    1: {"delta": 1.595769, "a_opt": 0.797885, "mse": 0.36338023, "method": "sign_theoretical"},
    2: {"delta": 0.995687, "a_opt": 1.991374, "mse": 0.11884605, "method": "uniform_theoretical"},
    3: {"delta": 0.586019, "a_opt": 2.344078, "mse": 0.03743966, "method": "uniform_theoretical"},
    4: {"delta": 0.335201, "a_opt": 2.681605, "mse": 0.01154288, "method": "uniform_theoretical"},
    5: {"delta": 0.188139, "a_opt": 3.010220, "mse": 0.00349521, "method": "uniform_theoretical"},
    6: {"delta": 0.104063, "a_opt": 3.330017, "mse": 0.00104005, "method": "uniform_theoretical"},
    7: {"delta": 0.056868, "a_opt": 3.639531, "mse": 0.00030433, "method": "uniform_theoretical"},
    8: {"delta": 0.030762, "a_opt": 3.937585, "mse": 0.00008769, "method": "uniform_theoretical"},
}

ROBUQ_ACT_QUANT_VERSION = "v1"


def normalize_quant_impl(quant_impl):
    if quant_impl in {None, "v1", "no_mean", "no_mean_oldste"}:
        return "v1"
    raise ValueError(f"RobuQ public release only supports quant_impl='v1', got {quant_impl!r}.")


@torch.jit.script
def robuq_uniform_quant_no_mean(x: torch.Tensor, delta: torch.Tensor, bits: int, dim: int = -1) -> torch.Tensor:
    scale = torch.abs(x).mean(dim=dim, keepdim=True) * 1.2533141373155001
    scale = scale + 1e-10
    x_norm = x / (scale * delta)
    x_quant = torch.round(x_norm)
    quant_max_val = 2 ** (bits - 1)
    x_quant = torch.clamp(x_quant, -quant_max_val, quant_max_val - 1)
    x_ste = x_norm + (x_quant - x_norm).detach()
    return x_ste * scale * delta


@torch.jit.script
def robuq_1bit_quant_no_mean(x: torch.Tensor, delta: torch.Tensor, dim: int = -1) -> torch.Tensor:
    scale = torch.abs(x).mean(dim=dim, keepdim=True) * 1.2533141373155001
    scale = scale + 1e-10
    x_norm = x / scale
    x_quant = torch.sign(x_norm) * delta * 0.5
    x_ste = x_norm + (x_quant - x_norm).detach()
    return x_ste * scale


class FusedUniformQuantSTE(nn.Module):
    """No-mean activation quantizer used by RobuQ.

    The straight-through estimator is applied in normalized space so that the
    activation scale path remains trainable during QAT.
    """

    def __init__(self, bits, quant_impl="v1"):
        super().__init__()
        quant_impl = normalize_quant_impl(quant_impl)
        if bits not in THEORETICAL_OPTIMAL_TABLE:
            raise ValueError(f"Unsupported bit-width {bits}; expected an integer in [1, 8].")
        self.bits = int(bits)
        self.config = THEORETICAL_OPTIMAL_TABLE[self.bits]
        self.register_buffer("delta_tensor", torch.tensor(self.config["delta"]))

    def forward(self, x):
        delta = self.delta_tensor.to(x.device)
        if self.bits == 1:
            return robuq_1bit_quant_no_mean(x, delta, dim=-1)
        return robuq_uniform_quant_no_mean(x, delta, self.bits, dim=-1)


class BitNetPerChannel(Function):
    """Per-output-channel W1.58 ternary weight quantization."""

    @staticmethod
    def forward(ctx, weight, epsilon=1e-8):
        scale = weight.abs().mean(dim=1, keepdim=True)
        weight_scaled = weight / (scale + epsilon)
        weight_rounded = torch.round(weight_scaled).clamp(-1, 1)
        return weight_rounded * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class BitNetGroupwise(Function):
    """Group-wise W1.58 ternary weight quantization along input features."""

    @staticmethod
    def forward(ctx, weight, group_size=128, epsilon=1e-8):
        if weight.dim() != 2:
            raise ValueError(f"BitNetGroupwise expects a 2D Linear weight, got shape {tuple(weight.shape)}.")
        group_size = int(group_size)
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}.")

        out_features, in_features = weight.shape
        num_groups = (in_features + group_size - 1) // group_size
        padded_features = num_groups * group_size
        pad_features = padded_features - in_features

        if pad_features:
            weight_grouped = F.pad(weight, (0, pad_features)).view(out_features, num_groups, group_size)
        else:
            weight_grouped = weight.view(out_features, num_groups, group_size)

        counts = torch.full((num_groups,), group_size, device=weight.device, dtype=weight.dtype)
        if pad_features:
            counts[-1] = group_size - pad_features
        scale = weight_grouped.abs().sum(dim=2, keepdim=True) / counts.view(1, num_groups, 1)

        weight_scaled = weight_grouped / (scale + epsilon)
        weight_rounded = torch.round(weight_scaled).clamp(-1, 1)
        quantized = weight_rounded * scale
        return quantized.view(out_features, padded_features)[:, :in_features]

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def quantize_linear_weights_bitnet(weight, group_size=None):
    if group_size is None:
        return BitNetPerChannel.apply(weight)
    return BitNetGroupwise.apply(weight, group_size)


def get_quantized_weights_discrete(weight, group_size=None, epsilon=1e-8):
    if group_size is not None:
        if weight.dim() != 2:
            raise ValueError(f"Group-wise BitNet expects a 2D Linear weight, got shape {tuple(weight.shape)}.")
        group_size = int(group_size)
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}.")

        out_features, in_features = weight.shape
        num_groups = (in_features + group_size - 1) // group_size
        padded_features = num_groups * group_size
        pad_features = padded_features - in_features
        if pad_features:
            weight_grouped = F.pad(weight, (0, pad_features)).view(out_features, num_groups, group_size)
        else:
            weight_grouped = weight.view(out_features, num_groups, group_size)

        counts = torch.full((num_groups,), group_size, device=weight.device, dtype=weight.dtype)
        if pad_features:
            counts[-1] = group_size - pad_features
        scale = weight_grouped.abs().sum(dim=2, keepdim=True) / counts.view(1, num_groups, 1)
        discrete = torch.round(weight_grouped / (scale + epsilon)).clamp(-1, 1)
        return discrete.view(out_features, padded_features)[:, :in_features]

    scale = weight.abs().mean(dim=1, keepdim=True)
    return torch.round(weight / (scale + epsilon)).clamp(-1, 1)


class LoRALinear(nn.Module):
    """Low-rank residual branch initialized from the leading SVD factors."""

    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = min(rank, in_features, out_features)
        self.B = nn.Linear(in_features, self.rank, bias=False)
        self.A = nn.Linear(self.rank, out_features, bias=False)
        nn.init.normal_(self.A.weight, mean=0.0, std=1e-4)
        nn.init.normal_(self.B.weight, mean=0.0, std=1e-4)

    def init_from_svd(self, weight_tensor):
        with torch.no_grad():
            try:
                u, s, v = torch.svd(weight_tensor)
            except Exception:
                u, s, vh = torch.linalg.svd(weight_tensor, full_matrices=False)
                v = vh.mH
            u_r = u[:, : self.rank]
            s_r = s[: self.rank]
            v_r = v[:, : self.rank]
            sqrt_s = torch.diag(torch.sqrt(s_r))
            self.A.weight.data = (u_r @ sqrt_s).to(self.A.weight.dtype)
            self.B.weight.data = (sqrt_s @ v_r.t()).to(self.B.weight.dtype)

    def forward(self, x):
        return self.A(self.B(x))

    def get_equiv_weight(self):
        return torch.matmul(self.A.weight, self.B.weight)


class RobuQLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        pretrained_weight=None,
        if_hadamard=False,
        LN_in=True,
        LN_out=False,
        w_bits=1.58,
        a_bits=32,
        if_svd=False,
        svd_rank=16,
        svd_before_rotation=False,
        quant_impl="v1",
        w_group_size=None,
        init_svd=True,
    ):
        super().__init__(in_features, out_features, bias=bias)
        quant_impl = normalize_quant_impl(quant_impl)
        if LN_in and LN_out:
            raise ValueError("LN_in and LN_out cannot both be true.")
        if pretrained_weight is not None:
            self.weight.data = pretrained_weight.clone()

        self.LN_in = LN_in
        self.LN_out = LN_out
        if LN_in:
            self.sub_norm = nn.RMSNorm(in_features, eps=1e-6)
        if LN_out:
            self.sub_norm = nn.RMSNorm(out_features, eps=1e-6)

        self.if_hadamard = if_hadamard
        self.if_svd = if_svd
        self.svd_rank = svd_rank
        self.svd_before_rotation = svd_before_rotation and if_svd

        if self.svd_before_rotation:
            self.lora = LoRALinear(in_features, out_features, rank=svd_rank).to(self.weight.device)
            if init_svd:
                self.lora.init_from_svd(self.weight)
                with torch.no_grad():
                    self.weight.data -= self.lora.get_equiv_weight()
            if if_hadamard:
                self.hadamard_init(in_features)
        else:
            if if_hadamard:
                self.hadamard_init(in_features)
            if if_svd:
                self.lora = LoRALinear(in_features, out_features, rank=svd_rank).to(self.weight.device)
                if init_svd:
                    self.lora.init_from_svd(self.weight)
                    with torch.no_grad():
                        self.weight.data -= self.lora.get_equiv_weight()
            else:
                self.lora = None

        self.w_bits = 1.58 if w_bits == 1.58 else int(w_bits)
        self.w_group_size = w_group_size
        if self.w_bits == 1.58:
            self.weight_quantize = lambda weight: quantize_linear_weights_bitnet(weight, self.w_group_size)
        elif self.w_bits == 32:
            self.weight_quantize = nn.Identity()
        else:
            self.weight_quantize = FusedUniformQuantSTE(self.w_bits)

        self.a_bits = int(a_bits)
        self.quant_impl = "v1"
        if self.a_bits != 32:
            self.activation_quantize = FusedUniformQuantSTE(self.a_bits)
        self.cin = in_features
        self.cout = out_features
        self.use_fp32_equiv = False
        self.use_packed_weight = False

    def get_discrete_weights(self):
        return get_quantized_weights_discrete(self.weight, self.w_group_size)

    def export_packed_weight(self, scale_dtype=torch.float16):
        if self.w_bits != 1.58:
            raise ValueError(f"Packed RobuQ export only supports W1.58 weights, got w_bits={self.w_bits}.")
        return make_packed_linear_payload(self.weight.detach(), group_size=self.w_group_size, scale_dtype=scale_dtype)

    def load_packed_weight(self, payload, dtype=None):
        target_dtype = dtype or self.weight.dtype
        weight = dequantize_packed_linear_weight(payload, device=self.weight.device, dtype=target_dtype)
        with torch.no_grad():
            self.weight.data.copy_(weight.to(device=self.weight.device, dtype=self.weight.dtype))
        self.weight.requires_grad_(False)
        self.use_packed_weight = True

    def refresh_bits(self, a_bits=None, w_bits=None):
        if a_bits is not None:
            self.a_bits = int(a_bits)
            if self.a_bits == 32:
                self.activation_quantize = nn.Identity()
            else:
                self.activation_quantize = FusedUniformQuantSTE(self.a_bits).to(self.weight.device)
        if w_bits is not None:
            self.w_bits = 1.58 if w_bits == 1.58 else int(w_bits)
            if self.w_bits == 1.58:
                self.weight_quantize = lambda weight: quantize_linear_weights_bitnet(weight, self.w_group_size)
            elif self.w_bits == 32:
                self.weight_quantize = nn.Identity()
            else:
                self.weight_quantize = FusedUniformQuantSTE(self.w_bits).to(self.weight.device)

    def enable_fp32_equiv(self, weight, bias=None):
        self.register_buffer("fp32_equiv_weight", weight.detach().clone())
        if bias is None:
            self.fp32_equiv_bias = None
        else:
            self.register_buffer("fp32_equiv_bias", bias.detach().clone())
        self.use_fp32_equiv = True

    def hadamard_init(self, in_features):
        had_k, k = get_hadK(in_features)
        random_signs = torch.randint(0, 2, (in_features,), device=self.weight.device).float() * 2 - 1
        self.register_buffer("hadamard_signs", random_signs)
        with torch.no_grad():
            self.weight.data *= self.hadamard_signs.unsqueeze(0)
            original_device = self.weight.device
            weight_tmp = self.weight.data
            if not weight_tmp.is_cuda and torch.cuda.is_available():
                weight_tmp = matmul_hadU_cuda(weight_tmp.cuda(), had_k, k)
                self.weight.data = weight_tmp.to(original_device)
            elif weight_tmp.is_cuda:
                self.weight.data = matmul_hadU_cuda(self.weight.data, had_k, k)
            else:
                self.weight.data = matmul_hadU(self.weight.data)

    def activation_hadamard(self, x):
        x = x * self.hadamard_signs
        had_k, k = get_hadK(self.cin)
        if x.is_cuda:
            return matmul_hadU_cuda(x, had_k, k)
        return matmul_hadU(x)

    def forward(self, input):
        if self.use_fp32_equiv and self.a_bits == 32:
            bias = self.fp32_equiv_bias if hasattr(self, "fp32_equiv_bias") else None
            return F.linear(input, self.fp32_equiv_weight, bias)

        if self.LN_in:
            input = self.sub_norm(input)
        x_before_rot = input
        x_rot = self.activation_hadamard(input) if self.if_hadamard else input
        x_main = self.activation_quantize(x_rot) if self.a_bits != 32 else x_rot
        main_weight = self.weight if self.use_packed_weight else self.weight_quantize(self.weight)
        output = F.linear(x_main, main_weight, self.bias)

        if self.if_svd and self.lora is not None:
            x_svd = x_before_rot if self.svd_before_rotation else x_rot
            output = output + self.lora(x_svd)
        if self.LN_out:
            output = self.sub_norm(output)
        return output


def init_RobuQLinear_from_Linear(
    linear,
    w_bits=1.58,
    if_hadamard=True,
    LN_in=True,
    LN_out=False,
    a_bits=32,
    if_svd=True,
    svd_rank=16,
    svd_before_rotation=False,
    quant_impl="v1",
    w_group_size=None,
    init_svd=True,
):
    if os.environ.get("ROBUQ_VERBOSE_INIT", "0") == "1":
        print(
            "Initializing RobuQLinear "
            f"w_bits={w_bits}, a_bits={a_bits}, hadamard={if_hadamard}, "
            f"svd={if_svd}, svd_rank={svd_rank}, w_group_size={w_group_size}"
        )
    robuq_linear = RobuQLinear(
        in_features=linear.in_features,
        out_features=linear.out_features,
        bias=linear.bias is not None,
        pretrained_weight=linear.weight.data,
        if_hadamard=if_hadamard,
        LN_in=LN_in,
        LN_out=LN_out,
        w_bits=w_bits,
        a_bits=a_bits,
        if_svd=if_svd,
        svd_rank=svd_rank,
        svd_before_rotation=svd_before_rotation,
        quant_impl=quant_impl,
        w_group_size=w_group_size,
        init_svd=init_svd,
    )
    if linear.bias is not None:
        robuq_linear.bias.data = linear.bias.data.clone()
    if w_bits == 32 and a_bits == 32 and not LN_in and not LN_out:
        bias = linear.bias.data if linear.bias is not None else None
        robuq_linear.enable_fp32_equiv(linear.weight.data, bias)
    return robuq_linear
