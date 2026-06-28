"""Distributed sampling and FID evaluation for quantized DiT checkpoints."""
import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from PIL import Image
from tqdm import tqdm

def configure_precision(force_fp32: bool):
    use_tf32 = not force_fp32
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    if force_fp32 and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion import create_diffusion
from models import DiT_models
from quant_utils import replace_dit_linear_layers
from sample_qat import checkpoint_args, default_weights_dir, load_qat_checkpoint, parse_w_bits, resolve_arg
from ampn_utils import apply_mix_config, load_mix_config, normalize_mix_config
from quant.packed import apply_packed_weights, is_packed_checkpoint


DEFAULT_REAL_STATS = str(
    REPO_ROOT / "data" / "stats" /
    "imagenet-train-msNone-r1x256x256-image-fr12.0-asvFalse-off1-n1268347-"
    "resize256x256-inception-2015-12-05-stat-mean_cov.pkl"
)
DEFAULT_ALPHAFLOW_ROOT = str(REPO_ROOT / "third_party" / "alphaflow")


def is_distributed():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed():
    if is_distributed():
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    else:
        rank, world_size, local_rank = 0, 1, 0
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce_(tensor):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def load_model(args, device, rank):
    checkpoint = None
    ckpt_args = {}
    if args.ckpt and not args.from_scratch:
        checkpoint = load_qat_checkpoint(args.ckpt, device)
        ckpt_args = checkpoint_args(checkpoint)
    packed_checkpoint = is_packed_checkpoint(checkpoint)

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

    model = DiT_models[args.model](
        input_size=args.image_size // 8,
        num_classes=args.num_classes,
    ).to(device)

    quant_report = replace_dit_linear_layers(
        model,
        a_bits=args.a_bits,
        w_bits=args.w_bits,
        if_hadamard=args.hadamard,
        if_svd=args.svd,
        svd_rank=args.svd_rank,
        svd_before_rotation=args.svd_before_rotation,
        quant_impl=args.quant_impl,
        verbose=args.verbose_replace and rank == 0,
    )
    if rank == 0:
        print(
            f"Quantized DiT linear layers: {quant_report['num_replaced']} "
            f"(w_bits={args.w_bits}, a_bits={args.a_bits}, quant_impl={args.quant_impl}, "
            f"mix_config={args.mix_config})"
        )
    if checkpoint is not None:
        if checkpoint.get("mix_config_data") is not None:
            applied = apply_mix_config(model, normalize_mix_config(checkpoint["mix_config_data"]), verbose=args.verbose_replace and rank == 0)
            if rank == 0:
                print(f"Applied embedded AMPN mix config: {len(applied)} layers")
        elif args.mix_config:
            applied = apply_mix_config(model, load_mix_config(args.mix_config), verbose=args.verbose_replace and rank == 0)
            if rank == 0:
                print(f"Applied AMPN mix config: {len(applied)} layers")

        state_key = "model" if packed_checkpoint else ("ema" if args.use_ema and "ema" in checkpoint else "model")
        if state_key not in checkpoint:
            raise KeyError(f"Checkpoint does not contain '{state_key}'. Available keys: {list(checkpoint.keys())}")
        missing, unexpected = model.load_state_dict(checkpoint[state_key], strict=False)
        if packed_checkpoint:
            packed_report = apply_packed_weights(model, checkpoint["packed_linears"], dtype=model.x_embedder.proj.weight.dtype)
        if rank == 0:
            step = checkpoint.get("train_steps", Path(args.ckpt).stem)
            packed_msg = f", packed_layers={len(packed_report['loaded'])}" if packed_checkpoint else ""
            print(f"Loaded {state_key} from {args.ckpt} (step={step}); missing={len(missing)}, unexpected={len(unexpected)}{packed_msg}")
    elif args.from_scratch and rank == 0:
        print("Using random initialization for smoke/debug sampling.")
    else:
        raise ValueError("--ckpt is required unless --from-scratch is set.")

    model.eval()
    return model


def resolve_sample_folder(args):
    if args.sample_folder:
        return Path(args.sample_folder)
    ckpt_name = Path(args.ckpt).stem if args.ckpt else "from_scratch"
    model_name = (args.model or "DiT-XL/2").replace("/", "-")
    mix_name = "mix" if args.mix_config else f"a{args.a_bits or 'ckpt'}"
    return Path(args.results_dir) / f"{model_name}-{ckpt_name}-{mix_name}-steps{args.num_sampling_steps}-cfg{args.cfg_scale}-n{args.num_samples}" / "samples"


def make_batch_noise_and_labels(args, indices, device, latent_size):
    z_items, y_items = [], []
    fixed_labels = args.class_labels or ([] if args.class_label is None else [args.class_label])
    for sample_idx in indices:
        gen = torch.Generator(device=device)
        gen.manual_seed(args.global_seed * 1_000_003 + int(sample_idx))
        z_items.append(torch.randn(4, latent_size, latent_size, device=device, generator=gen))
        if fixed_labels:
            y_items.append(int(fixed_labels[int(sample_idx) % len(fixed_labels)]))
        else:
            y_items.append(int(torch.randint(0, args.num_classes, (1,), device=device, generator=gen).item()))
    return torch.stack(z_items, dim=0), torch.tensor(y_items, device=device, dtype=torch.long)


@torch.no_grad()
def sample_images(args, model, device, rank, world_size):
    sample_folder = resolve_sample_folder(args)
    if rank == 0:
        sample_folder.mkdir(parents=True, exist_ok=True)
        print(f"Saving PNG samples to {sample_folder}")
    barrier()

    weights_dir = Path(args.weights_dir)
    default_vae_path = weights_dir / f"sd-vae-ft-{args.vae}"
    vae_path = args.vae_path or (str(default_vae_path) if default_vae_path.is_dir() else f"stabilityai/sd-vae-ft-{args.vae}")
    vae_kwargs = {"local_files_only": True} if Path(str(vae_path)).is_dir() else {}
    vae = AutoencoderKL.from_pretrained(vae_path, **vae_kwargs).to(device)
    vae.eval()

    diffusion = create_diffusion(str(args.num_sampling_steps))
    latent_size = args.image_size // 8
    using_cfg = args.cfg_scale > 1.0
    global_batch_size = args.per_proc_batch_size * world_size
    total_rounded = int(math.ceil(args.num_samples / global_batch_size) * global_batch_size)
    iterations = total_rounded // global_batch_size
    iterator = range(iterations)
    if rank == 0:
        print(f"Sampling target={args.num_samples}, rounded={total_rounded}, world_size={world_size}")
        iterator = tqdm(iterator, desc="Sampling")

    for batch_idx in iterator:
        base = batch_idx * global_batch_size
        indices = [base + i * world_size + rank for i in range(args.per_proc_batch_size)]
        z, y = make_batch_noise_and_labels(args, indices, device, latent_size)

        if using_cfg:
            z_in = torch.cat([z, z], dim=0)
            y_null = torch.full_like(y, args.num_classes)
            y_in = torch.cat([y, y_null], dim=0)
            model_kwargs = dict(y=y_in, cfg_scale=args.cfg_scale)
            sample_fn = model.forward_with_cfg
        else:
            z_in = z
            model_kwargs = dict(y=y)
            sample_fn = model.forward

        samples = diffusion.p_sample_loop(
            sample_fn,
            z_in.shape,
            z_in,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=device,
        )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)
        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        for sample_idx, sample in zip(indices, samples):
            if sample_idx < args.num_samples:
                Image.fromarray(sample).save(sample_folder / f"{sample_idx:06d}.png")

    barrier()
    if rank == 0 and args.save_npz:
        npz_path = create_npz_from_sample_folder(sample_folder, args.num_samples)
        print(f"Saved NPZ to {npz_path}")
    barrier()
    return sample_folder


def create_npz_from_sample_folder(sample_folder, num_samples):
    samples = []
    for i in tqdm(range(num_samples), desc="Building npz"):
        path = Path(sample_folder) / f"{i:06d}.png"
        if not path.is_file():
            raise FileNotFoundError(f"Missing sample image: {path}")
        samples.append(np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8))
    samples = np.stack(samples, axis=0)
    out = str(sample_folder) + ".npz"
    np.savez(out, arr_0=samples)
    return out


def load_real_mean_cov(real_stats_path):
    with open(real_stats_path, "rb") as f:
        state = pickle.load(f)
    mean = state["raw_mean"] / state["num_items"]
    cov = state["raw_cov"] / state["num_items"] - np.outer(mean, mean)
    return mean, cov, int(state["num_items"])


def compute_frechet_distance(mean_gen, cov_gen, mean_real, cov_real):
    m = np.square(mean_gen - mean_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(cov_gen, cov_real), disp=False)
    return float(np.real(m + np.trace(cov_gen + cov_real - 2 * s)))


def import_alphaflow_detector(args):
    alphaflow_root = Path(args.alphaflow_root)
    os.environ.setdefault("DNNLIB_CACHE_DIR", str(alphaflow_root / ".cache" / "dnnlib"))
    if str(alphaflow_root) not in sys.path:
        sys.path.insert(0, str(alphaflow_root))
    from src.metrics import detectors
    return detectors.construct_inception_detector(args.device, verbose=args.verbose_fid)


def load_image_batch(paths):
    imgs = []
    for path in paths:
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).copy()
        imgs.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(imgs, dim=0)


@torch.no_grad()
def compute_fid_from_folder(args, sample_folder, device, rank, world_size):
    mean_real, cov_real, real_count = load_real_mean_cov(args.real_stats)
    detector = import_alphaflow_detector(args)

    paths = [Path(sample_folder) / f"{i:06d}.png" for i in range(args.num_samples)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} generated samples, first missing: {missing[0]}")
    my_paths = paths[rank::world_size]

    local_count = torch.zeros([], dtype=torch.float64, device=device)
    local_sum = None
    local_outer = None
    iterator = range(0, len(my_paths), args.detector_batch_size)
    if rank == 0:
        print(f"Computing FID with {args.num_samples} generated samples; real_stats_items={real_count}")
        iterator = tqdm(iterator, desc="FID features")
    for start in iterator:
        batch_paths = my_paths[start:start + args.detector_batch_size]
        videos = load_image_batch(batch_paths).unsqueeze(1).to(device, non_blocking=True)
        feats = detector.run(videos, resize_frames=(args.fid_resize, args.fid_resize)).to(torch.float64)
        if local_sum is None:
            local_sum = torch.zeros(feats.shape[1], dtype=torch.float64, device=device)
            local_outer = torch.zeros(feats.shape[1], feats.shape[1], dtype=torch.float64, device=device)
        local_count += feats.shape[0]
        local_sum += feats.sum(dim=0)
        local_outer += feats.t().matmul(feats)

    if local_sum is None:
        raise RuntimeError(f"Rank {rank} received no images for FID.")
    all_reduce_(local_count)
    all_reduce_(local_sum)
    all_reduce_(local_outer)

    if rank != 0:
        return None
    count = int(local_count.item())
    if count != args.num_samples:
        raise RuntimeError(
            f"FID feature count mismatch: got {count}, expected {args.num_samples}. "
            "Do not trust this FID. Re-run with torchrun using the intended number of processes."
        )
    mean_gen = (local_sum / local_count).cpu().numpy()
    raw_outer = (local_outer / local_count).cpu().numpy()
    cov_gen = raw_outer - np.outer(mean_gen, mean_gen)
    fid = compute_frechet_distance(mean_gen, cov_gen, mean_real, cov_real)
    if abs(fid) < 1e-12:
        raise RuntimeError(
            "Computed FID is numerically zero, which is not credible for generated samples. "
            "Check detector/statistics inputs before using this result."
        )
    result = {
        "fid": fid,
        "num_samples": count,
        "real_stats": args.real_stats,
        "real_stats_items": real_count,
        "sample_folder": str(sample_folder),
        "detector": "inception-2015-12-05",
        "fid_resize": args.fid_resize,
    }
    out_path = Path(args.fid_output) if args.fid_output else Path(sample_folder).parent / "fid.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return result


def main(args):
    assert torch.cuda.is_available(), "Distributed evaluation requires CUDA."
    configure_precision(args.force_fp32)
    rank, world_size, local_rank = setup_distributed()
    device = torch.device("cuda", local_rank)
    args.device = device
    torch.set_grad_enabled(False)

    if not (1 <= world_size <= 8):
        raise ValueError(f"Expected 1-8 GPUs, got world_size={world_size}.")
    if rank == 0:
        print(f"Starting eval rank={rank}, world_size={world_size}, mode={args.mode}")

    sample_folder = resolve_sample_folder(args)
    if args.mode in {"sample", "all"}:
        model = load_model(args, device, rank)
        sample_folder = sample_images(args, model, device, rank, world_size)
        del model
        torch.cuda.empty_cache()

    if args.mode in {"fid", "all"}:
        compute_fid_from_folder(args, sample_folder, device, rank, world_size)

    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["sample", "fid", "all"], default="all")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default=None)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--weights-dir", type=str, default=str(default_weights_dir()))
    parser.add_argument("--vae-path", type=str, default=None)

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

    parser.add_argument("--results-dir", type=str, default="results_eval")
    parser.add_argument("--sample-folder", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=50_000)
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--class-label", type=int, default=None)
    parser.add_argument("--class-labels", type=int, nargs="*", default=None)
    parser.add_argument("--save-npz", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--alphaflow-root", type=str, default=DEFAULT_ALPHAFLOW_ROOT)
    parser.add_argument("--real-stats", type=str, default=DEFAULT_REAL_STATS)
    parser.add_argument("--detector-batch-size", type=int, default=64)
    parser.add_argument("--fid-resize", type=int, default=256)
    parser.add_argument("--fid-output", type=str, default=None)
    parser.add_argument("--verbose-fid", action="store_true")
    main(parser.parse_args())
