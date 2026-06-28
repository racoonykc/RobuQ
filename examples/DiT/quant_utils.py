"""DiT-specific quantization wiring.

The shared quantized modules live in the repo-level quant/ package. This file
keeps DiT module-selection policy local to the DiT example.
"""
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant.replace import replace_linear_layers, replace_linear_with_robuq


DEFAULT_TARGET_KEYWORDS = ("blocks.",)
DEFAULT_SKIP_KEYWORDS = ()


def replace_dit_linear_layers(
    model,
    a_bits=4,
    w_bits=4,
    if_hadamard=True,
    if_svd=False,
    svd_rank=16,
    svd_before_rotation=False,
    quant_impl="v1",
    w_group_size=None,
    init_svd=True,
    target_keywords=DEFAULT_TARGET_KEYWORDS,
    skip_keywords=DEFAULT_SKIP_KEYWORDS,
    verbose=True,
):
    """Apply RobuQLinear to selected DiT linear layers.

    By default, only transformer blocks are quantized. Patch embedding,
    timestep/class embeddings, and the final projection stay in full precision
    until an experiment explicitly opts into quantizing them.
    """
    return replace_linear_layers(
        model,
        replacement_fn=lambda linear: replace_linear_with_robuq(
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
            w_group_size=w_group_size,
            init_svd=init_svd,
        ),
        target_keywords=target_keywords,
        skip_keywords=skip_keywords,
        verbose=verbose,
    )
