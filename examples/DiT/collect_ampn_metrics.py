"""Collect AMPN QAT-based sensitivity metrics with embarrassingly parallel GPUs."""
import argparse
import csv
import os
import sys
from copy import deepcopy
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from torch.utils.data import DataLoader
from torchvision import transforms

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion import create_diffusion
from download import find_model
from models import DiT_models
from train_qat import (
    build_fixed_eval_cache,
    build_image_dataset,
    center_crop_arr,
    compute_fixed_eval_loss,
    default_data_path,
    default_weights_dir,
    requires_grad,
)
from ampn_utils import ADALN_OP, AMPN_OPS, OP_COSTS, replace_one_dit_linear


def is_distributed():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup():
    if is_distributed():
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    else:
        rank, world_size, local_rank = 0, 1, 0
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def parse_bits(bits):
    return [int(item) for item in str(bits).split(",") if item]


def build_train_loader(args, rank):
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = build_image_dataset(args.data_path, transform)
    generator = torch.Generator()
    generator.manual_seed(args.train_seed + rank)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        generator=generator,
        persistent_workers=args.num_workers > 0,
    )


def resolve_pretrained(args):
    if args.pretrained_ckpt:
        return args.pretrained_ckpt
    local = Path(args.weights_dir) / f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    return str(local) if local.is_file() else f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"


def train_isolated(model, loader, vae, diffusion, args, device):
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0)
    data_iter = iter(loader)
    for _ in range(args.isolated_train_steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.no_grad():
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)
        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
        loss = diffusion.training_losses(model, x, t, dict(y=y))["loss"].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
    return model


def build_jobs(args):
    ops = list(AMPN_OPS)
    if args.profile_adaln:
        ops.append(ADALN_OP)
    jobs = []
    for block_idx in range(args.total_blocks):
        for op_name in ops:
            for bit in parse_bits(args.activation_bits):
                jobs.append((block_idx, op_name, bit))
    return jobs


def write_rows(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "block_idx",
                "op_name",
                "activation_bits",
                "eval_loss",
                "loss_diff",
                "fp_eval_loss",
                "isolated_train_steps",
                "op_cost",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def merge_csvs(result_dir, world_size):
    rows = []
    for rank in range(world_size):
        part = Path(result_dir) / f"sensitivity_rank{rank}.csv"
        if not part.is_file():
            continue
        with open(part, "r") as f:
            rows.extend(csv.DictReader(f))
    rows.sort(key=lambda r: (int(r["block_idx"]), r["op_name"], int(r["activation_bits"])))
    out = Path(result_dir) / "ampn_sensitivity.csv"
    write_rows(out, rows)
    return out, len(rows)


def main(args):
    assert torch.cuda.is_available(), "AMPN metric collection requires CUDA."
    rank, world_size, local_rank = setup()
    device = torch.device("cuda", local_rank)
    result_dir = Path(args.results_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.global_seed + rank)
    np.random.seed(args.global_seed + rank)

    model = DiT_models[args.model](input_size=args.image_size // 8, num_classes=args.num_classes).to(device)
    if args.from_scratch:
        print("Using random initialization for AMPN smoke/debug run.")
    else:
        model.load_state_dict(find_model(resolve_pretrained(args)))
    model.eval()
    fp_model = deepcopy(model).to(device)

    diffusion = create_diffusion(timestep_respacing="")
    weights_dir = Path(args.weights_dir)
    default_vae_path = weights_dir / f"sd-vae-ft-{args.vae}"
    vae_path = args.vae_path or (str(default_vae_path) if default_vae_path.is_dir() else f"stabilityai/sd-vae-ft-{args.vae}")
    vae = AutoencoderKL.from_pretrained(vae_path).to(device)
    vae.eval()
    requires_grad(vae, False)

    eval_cache = build_fixed_eval_cache(args, vae, device, logger=type("Logger", (), {"info": print})())
    fp_eval_loss = compute_fixed_eval_loss(fp_model, eval_cache, diffusion, args.eval_batch_size)
    if rank == 0:
        print(f"fp_eval_loss={fp_eval_loss:.8f}")
    del fp_model
    torch.cuda.empty_cache()

    train_loader = build_train_loader(args, rank)
    jobs = build_jobs(args)
    my_jobs = [job for i, job in enumerate(jobs) if i % world_size == rank]
    print(f"rank={rank}/{world_size} jobs={len(my_jobs)}/{len(jobs)}")

    rows = []
    start = time()
    for job_idx, (block_idx, op_name, bit) in enumerate(my_jobs):
        current = deepcopy(model).to(device)
        replace_one_dit_linear(
            current,
            block_idx,
            op_name,
            a_bits=bit,
            w_bits=args.w_bits,
            if_hadamard=args.hadamard,
            if_svd=args.svd,
            svd_rank=args.svd_rank,
            svd_before_rotation=args.svd_before_rotation,
            quant_impl=args.quant_impl,
        )
        if not args.train_all:
            requires_grad(current, False)
            target = f"blocks.{block_idx}.{op_name}"
            requires_grad(dict(current.named_modules())[target], True)
        train_isolated(current, train_loader, vae, diffusion, args, device)
        eval_loss = compute_fixed_eval_loss(current, eval_cache, diffusion, args.eval_batch_size)
        loss_diff = eval_loss - fp_eval_loss
        rows.append({
            "block_idx": block_idx,
            "op_name": op_name,
            "activation_bits": bit,
            "eval_loss": eval_loss,
            "loss_diff": loss_diff,
            "fp_eval_loss": fp_eval_loss,
            "isolated_train_steps": args.isolated_train_steps,
            "op_cost": OP_COSTS[op_name],
        })
        print(
            f"rank={rank} job={job_idx + 1}/{len(my_jobs)} block={block_idx} op={op_name} "
            f"a={bit} eval={eval_loss:.6f} diff={loss_diff:.6f}"
        )
        del current
        torch.cuda.empty_cache()

    part_path = result_dir / f"sensitivity_rank{rank}.csv"
    write_rows(part_path, rows)
    print(f"rank={rank} wrote {part_path}; elapsed_min={(time() - start) / 60:.1f}")

    if is_distributed():
        dist.barrier()
    if rank == 0:
        merged, count = merge_csvs(result_dir, world_size)
        print(f"merged {count} rows -> {merged}")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=default_data_path())
    parser.add_argument("--results-dir", type=str, default="results_ampn/dit_xl2_w158_metrics")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--total-blocks", type=int, default=28)
    parser.add_argument("--weights-dir", type=str, default=str(default_weights_dir()))
    parser.add_argument("--pretrained-ckpt", type=str, default=None)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--vae-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--isolated-train-steps", type=int, default=1000)
    parser.add_argument("--activation-bits", type=str, default="1,2,3,4")
    parser.add_argument("--w-bits", type=float, default=1.58)
    parser.add_argument("--hadamard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--svd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--svd-rank", type=int, default=16)
    parser.add_argument("--svd-before-rotation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--quant-impl", type=str, default="v1", choices=["v1"])
    parser.add_argument("--train-all", action="store_true", help="Train all parameters during profiling; default follows AMPN and trains only the target layer.")
    parser.add_argument("--profile-adaln", action="store_true")
    parser.add_argument("--eval-num-samples", type=int, default=1000)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--eval-data-path", type=str, default=None)
    parser.add_argument("--eval-num-workers", type=int, default=4)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--train-seed", type=int, default=123)
    main(parser.parse_args())
