from .quantized_modules.robuq import init_RobuQLinear_from_Linear
import torch.nn as nn


def replace_linear_with_robuq(
    linear,
    a_bits=32,
    w_bits=1.58,
    if_hadamard=True,
    if_svd=True,
    svd_rank=16,
    LN_in=False,
    LN_out=False,
    svd_before_rotation=False,
    quant_impl="v1",
    w_group_size=None,
    init_svd=True,
):
    """Replace an ``nn.Linear`` layer with ``RobuQLinear``."""
    return init_RobuQLinear_from_Linear(
        linear,
        w_bits=w_bits,
        a_bits=a_bits,
        if_hadamard=if_hadamard,
        if_svd=if_svd,
        svd_rank=svd_rank,
        LN_in=LN_in,
        LN_out=LN_out,
        svd_before_rotation=svd_before_rotation,
        quant_impl=quant_impl,
        w_group_size=w_group_size,
        init_svd=init_svd,
    )


def _matches(name, keywords):
    if keywords is None:
        return True
    return any(keyword in name for keyword in keywords)


def replace_linear_layers(model, replacement_fn, target_keywords=None, skip_keywords=(), verbose=True):
    """Recursively replace selected ``nn.Linear`` modules.

    Returns a dictionary containing replaced and skipped module names.
    """
    module_dict = dict(model.named_modules())
    replaced = []
    skipped = []

    for name, module in list(model.named_modules()):
        if not name or not isinstance(module, nn.Linear):
            continue
        if not _matches(name, target_keywords):
            skipped.append(name)
            continue
        if any(keyword in name for keyword in skip_keywords):
            skipped.append(name)
            continue

        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model if parent_name == "" else module_dict[parent_name]
        new_module = replacement_fn(module)
        new_module = new_module.to(device=module.weight.device, dtype=module.weight.dtype)
        setattr(parent, child_name, new_module)
        replaced.append(name)
        if verbose:
            print(f"replace {name}: nn.Linear -> {new_module.__class__.__name__}")

    return {
        "num_replaced": len(replaced),
        "replaced": replaced,
        "num_skipped": len(skipped),
        "skipped": skipped,
    }
