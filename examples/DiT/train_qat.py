# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Quantization-aware training for DiT with RobuQ.

Single GPU:
    python train_qat.py --a-bits 4

1-8 GPUs:
    torchrun --standalone --nproc_per_node=8 train_qat.py \
        --global-batch-size 256 --a-bits 4
"""
import argparse
import logging
import os
import sys
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from pathlib import Path
from time import time

import numpy as np
import torch

import torch.distributed as dist
from diffusers.models import AutoencoderKL
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader, has_file_allowed_extension
from torchvision.utils import save_image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion import create_diffusion
from download import find_model
from models import DiT_models
from quant_utils import replace_dit_linear_layers
from ampn_utils import apply_mix_config, load_mix_config
from quant.quantized_modules.robuq import ROBUQ_ACT_QUANT_VERSION


def configure_precision(force_fp32: bool):
    use_tf32 = not force_fp32
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    if force_fp32 and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


def default_weights_dir():
    return REPO_ROOT / "weights"


def default_data_path():
    return str(REPO_ROOT / "data" / "imagenet" / "train")


def parse_w_bits(value):
    if isinstance(value, float):
        return value
    normalized = str(value).lower().replace("w", "")
    if normalized in {"1.58", "1_58", "158"}:
        return 1.58
    return int(normalized)


def is_distributed():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed():
    if is_distributed():
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def create_logger(logging_dir, rank):
    logger = logging.getLogger("dit_qat")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not is_main_process(rank):
        logger.addHandler(logging.NullHandler())
        return logger

    formatter = logging.Formatter(
        "[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if logging_dir is not None:
        file_handler = logging.FileHandler(f"{logging_dir}/log.txt")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if name in ema_params:
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size),
            resample=Image.BOX,
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size),
        resample=Image.BICUBIC,
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def get_rng_state(device):
    state = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def set_rng_state(state, device):
    torch.set_rng_state(state["cpu"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device)


def init_wandb(args, experiment_dir, rank, logger):
    if not args.wandb or not is_main_process(rank):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is required unless --no-wandb is set.") from exc

    wandb_name = args.wandb_name or Path(experiment_dir).name
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=wandb_name,
        dir=experiment_dir,
        mode=args.wandb_mode,
        config=vars(args),
    )
    logger.info(f"Initialized wandb run: project={args.wandb_project}, name={wandb_name}")
    return run


def wandb_log(wandb_run, payload, step=None):
    if wandb_run is not None:
        wandb_run.log(payload, step=step)


def wandb_log_image(wandb_run, key, image_path, step=None):
    if wandb_run is None:
        return
    import wandb
    wandb_run.log({key: wandb.Image(str(image_path))}, step=step)


def resolve_pretrained_ckpt(args):
    if args.pretrained_ckpt:
        return args.pretrained_ckpt

    weights_dir = Path(args.weights_dir)
    local_ckpt = weights_dir / f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    if args.model == "DiT-XL/2" and local_ckpt.is_file():
        return str(local_ckpt)

    if args.model == "DiT-XL/2":
        return f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"

    return None


def load_pretrained_model(model, args, logger):
    if args.from_scratch:
        logger.info("Training QAT model from random initialization.")
        return

    ckpt_path = resolve_pretrained_ckpt(args)
    if ckpt_path is None:
        raise ValueError("--pretrained-ckpt is required for non-DiT-XL/2 models unless --from-scratch is set.")

    state_dict = find_model(ckpt_path)
    if args.pretrained_key:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint[args.pretrained_key]

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logger.info(f"Loaded pretrained checkpoint with missing={len(missing)}, unexpected={len(unexpected)}.")
    logger.info(f"Loaded pretrained checkpoint: {ckpt_path}")


def resolve_vae_path(args):
    if args.vae_path:
        return args.vae_path
    weights_dir = Path(args.weights_dir)
    default_vae_path = weights_dir / f"sd-vae-ft-{args.vae}"
    if default_vae_path.is_dir():
        return str(default_vae_path)
    return f"stabilityai/sd-vae-ft-{args.vae}"


def load_resume_checkpoint(path, model, ema, opt, device, logger):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logger.info(f"Resume model load: missing={len(missing)}, unexpected={len(unexpected)}.")

    if ema is not None and "ema" in checkpoint:
        ema.load_state_dict(checkpoint["ema"], strict=False)
    if opt is not None and "opt" in checkpoint:
        opt.load_state_dict(checkpoint["opt"])

    return int(checkpoint.get("train_steps", 0)), int(checkpoint.get("epoch", 0))


def create_experiment_dir(args, rank):
    if not is_main_process(rank):
        return None, None

    os.makedirs(args.results_dir, exist_ok=True)
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        model_name = args.model.replace("/", "-")
        q_name = f"w{args.w_bits}a{args.a_bits}"
        experiment_index = len(glob(f"{args.results_dir}/*"))
        experiment_name = f"{experiment_index:03d}-{model_name}-qat-{q_name}"

    experiment_dir = Path(args.results_dir) / experiment_name
    checkpoint_dir = experiment_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return str(experiment_dir), str(checkpoint_dir)


class FlatImageNet(Dataset):
    """ImageNet split with files stored directly under root as nXXXXXXXX_*.jpeg."""

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.loader = default_loader

        image_paths = []
        class_names = set()
        with os.scandir(root) as entries:
            for entry in entries:
                if not entry.is_file():
                    continue
                if not has_file_allowed_extension(entry.name, IMG_EXTENSIONS):
                    continue
                class_name = entry.name.split("_", 1)[0]
                if not class_name:
                    continue
                image_paths.append(entry.path)
                class_names.add(class_name)

        if not image_paths:
            raise FileNotFoundError(f"Couldn't find any image files in flat ImageNet directory {root}.")

        self.classes = sorted(class_names)
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.samples = [
            (path, self.class_to_idx[Path(path).name.split("_", 1)[0]])
            for path in sorted(image_paths)
        ]
        self.targets = [target for _, target in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


def has_class_subdirs(root):
    with os.scandir(root) as entries:
        return any(entry.is_dir() for entry in entries)


def build_image_dataset(root, transform):
    if has_class_subdirs(root):
        return ImageFolder(root, transform=transform)
    return FlatImageNet(root, transform=transform)


def build_dataset(args, rank, world_size):
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = build_image_dataset(args.data_path, transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.global_batch_size // world_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    return dataset, sampler, loader


def build_fixed_eval_cache(args, vae, device, logger):
    eval_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = build_image_dataset(args.eval_data_path or args.data_path, eval_transform)
    eval_size = min(args.eval_num_samples, len(dataset))
    generator = torch.Generator()
    generator.manual_seed(args.eval_seed)
    indices = torch.randperm(len(dataset), generator=generator)[:eval_size].tolist()
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.eval_num_workers,
        pin_memory=True,
        drop_last=False,
    )

    rng_state = get_rng_state(device)
    torch.manual_seed(args.eval_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.eval_seed)

    latents = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            latents.append(vae.encode(x).latent_dist.sample().mul_(0.18215).cpu())
            labels.append(y.cpu())

    set_rng_state(rng_state, device)

    x_start = torch.cat(latents, dim=0).to(device)
    y = torch.cat(labels, dim=0).to(device)
    eval_generator = torch.Generator(device=device)
    eval_generator.manual_seed(args.eval_seed + 1)
    t = torch.randint(0, args.diffusion_steps, (x_start.shape[0],), device=device, generator=eval_generator)
    noise = torch.randn(x_start.shape, device=device, generator=eval_generator)

    logger.info(
        f"Built fixed eval cache: {x_start.shape[0]} samples from {args.eval_data_path or args.data_path}, "
        f"eval_seed={args.eval_seed}."
    )
    return {
        "x_start": x_start,
        "y": y,
        "t": t,
        "noise": noise,
    }


@torch.no_grad()
def compute_fixed_eval_loss(model, eval_cache, diffusion, batch_size):
    was_training = model.training
    model.eval()
    losses = []
    total = eval_cache["x_start"].shape[0]
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        x = eval_cache["x_start"][start:end]
        y = eval_cache["y"][start:end]
        t = eval_cache["t"][start:end]
        noise = eval_cache["noise"][start:end]
        loss_dict = diffusion.training_losses(model, x, t, dict(y=y), noise=noise)
        losses.append(loss_dict["loss"].detach())
    eval_loss = torch.cat(losses).mean().item()
    if was_training:
        model.train()
    return eval_loss


def parse_class_labels(labels):
    if labels is None:
        return [207, 360, 387, 974, 88, 979, 417, 279]
    if isinstance(labels, str):
        return [int(item) for item in labels.split(",") if item.strip()]
    return labels


@torch.no_grad()
def sample_and_log(model, vae, args, experiment_dir, device, step, wandb_run, logger, tag="train"):
    was_training = model.training
    model.eval()

    rng_state = get_rng_state(device)
    torch.manual_seed(args.sample_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.sample_seed)

    latent_size = args.image_size // 8
    sample_diffusion = create_diffusion(str(args.sample_steps))
    class_labels = parse_class_labels(args.sample_class_labels)
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    z = torch.cat([z, z], 0)
    y_null = torch.tensor([args.num_classes] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.sample_cfg_scale)

    samples = sample_diffusion.p_sample_loop(
        model.forward_with_cfg,
        z.shape,
        z,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=False,
        device=device,
    )
    samples, _ = samples.chunk(2, dim=0)
    samples = vae.decode(samples / 0.18215).sample

    output_dir = Path(experiment_dir) / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{tag}_step_{step:07d}.png"
    save_image(samples, output_path, nrow=min(4, n), normalize=True, value_range=(-1, 1))
    logger.info(f"(step={step:07d}) Saved sample grid to {output_path}")
    wandb_log_image(wandb_run, "sample/image", output_path, step=step)

    set_rng_state(rng_state, device)
    if was_training:
        model.train()
    return output_path


def main(args):
    assert torch.cuda.is_available(), "QAT training requires at least one CUDA GPU."

    configure_precision(args.force_fp32)
    args.w_bits = parse_w_bits(args.w_bits)
    assert 1 <= args.a_bits <= 8 or args.a_bits == 32, "--a-bits must be in [1, 8] or 32."

    rank, world_size, local_rank = setup_distributed()
    device = torch.device("cuda", local_rank)
    assert 1 <= world_size <= 8, f"This DiT QAT entry is intended for 1-8 GPUs, got world_size={world_size}."
    assert args.global_batch_size % world_size == 0, "--global-batch-size must be divisible by GPU count."

    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    experiment_dir, checkpoint_dir = create_experiment_dir(args, rank)
    if is_distributed():
        obj = [experiment_dir, checkpoint_dir]
        dist.broadcast_object_list(obj, src=0)
        experiment_dir, checkpoint_dir = obj

    logger = create_logger(experiment_dir, rank)
    logger.info(f"Starting rank={rank}, local_rank={local_rank}, seed={seed}, world_size={world_size}.")
    wandb_run = init_wandb(args, experiment_dir, rank, logger)

    assert args.image_size % 8 == 0, "Image size must be divisible by 8 for the VAE encoder."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    ).to(device)

    load_pretrained_model(model, args, logger)

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
        verbose=args.verbose_replace and is_main_process(rank),
    )
    set_rng_state(rng_state, device)
    logger.info(
        f"QAT quantized linear layers: {quant_report['num_replaced']} "
        f"(w_bits={args.w_bits}, a_bits={args.a_bits}, hadamard={args.hadamard}, "
        f"svd={args.svd}, svd_rank={args.svd_rank}, quant_impl={args.quant_impl}, "
        f"act_quant={ROBUQ_ACT_QUANT_VERSION})"
    )
    if args.mix_config:
        mix_config = load_mix_config(args.mix_config)
        applied_mix = apply_mix_config(model, mix_config, verbose=args.verbose_replace and is_main_process(rank))
        logger.info(f"Applied AMPN mixed activation config from {args.mix_config}: {len(applied_mix)} layers.")

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    update_ema(ema, model, decay=0)

    if args.freeze_non_quantized:
        requires_grad(model, False)
        for module in model.modules():
            if module.__class__.__name__ == "RobuQLinear":
                requires_grad(module, True)
        logger.info("Training only RobuQLinear parameters.")
    else:
        requires_grad(model, True)
        logger.info("Training all DiT parameters through QAT STE.")

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=args.find_unused_parameters,
    ) if is_distributed() else model

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    diffusion = create_diffusion(timestep_respacing="")
    vae_path = resolve_vae_path(args)
    vae_kwargs = {"local_files_only": True} if Path(str(vae_path)).is_dir() else {}
    vae = AutoencoderKL.from_pretrained(vae_path, **vae_kwargs).to(device)
    vae.eval()
    requires_grad(vae, False)

    dataset, sampler, loader = build_dataset(args, rank, world_size)
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    wandb_log(wandb_run, {"dataset/size": len(dataset), "dataset/eval_size": args.eval_num_samples})

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_steps = 0
    start_epoch = 0
    if args.resume:
        raw_model = model.module if isinstance(model, DDP) else model
        train_steps, start_epoch = load_resume_checkpoint(args.resume, raw_model, ema, opt, device, logger)
        logger.info(f"Resumed QAT checkpoint from {args.resume} at step={train_steps}, epoch={start_epoch}.")

    eval_cache = None
    if args.eval_every_steps > 0:
        if is_main_process(rank):
            eval_cache = build_fixed_eval_cache(args, vae, device, logger)
            raw_model = model.module if isinstance(model, DDP) else model
            initial_eval_loss = compute_fixed_eval_loss(raw_model, eval_cache, diffusion, args.eval_batch_size)
            logger.info(f"(step={train_steps:07d}) Fixed Eval Loss: {initial_eval_loss:.6f}")
            wandb_log(wandb_run, {"eval/fixed_loss": initial_eval_loss}, step=train_steps)
        if is_distributed():
            dist.barrier()

    if args.sample_every_steps > 0:
        if is_main_process(rank):
            sample_model = ema if args.sample_use_ema else (model.module if isinstance(model, DDP) else model)
            sample_and_log(sample_model, vae, args, experiment_dir, device, train_steps, wandb_run, logger, tag="initial")
        if is_distributed():
            dist.barrier()

    model.train()
    ema.eval()

    running_loss = 0.0
    log_steps = 0
    start_time = time()

    logger.info(f"Training QAT for {args.epochs} epochs.")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}.")
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            raw_model = model.module if isinstance(model, DDP) else model
            update_ema(ema, raw_model, decay=args.ema_decay)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize(device)
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                if is_distributed():
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss / world_size
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss.item():.4f}, "
                    f"Train Steps/Sec/GPU: {steps_per_sec:.2f}"
                )
                wandb_log(
                    wandb_run,
                    {
                        "train/loss": avg_loss.item(),
                        "train/steps_per_sec_per_gpu": steps_per_sec,
                        "train/lr": opt.param_groups[0]["lr"],
                        "epoch": epoch,
                    },
                    step=train_steps,
                )
                running_loss = 0.0
                log_steps = 0
                start_time = time()

            if args.eval_every_steps > 0 and train_steps % args.eval_every_steps == 0 and train_steps > 0:
                if is_distributed():
                    dist.barrier()
                if is_main_process(rank):
                    eval_loss = compute_fixed_eval_loss(raw_model, eval_cache, diffusion, args.eval_batch_size)
                    logger.info(f"(step={train_steps:07d}) Fixed Eval Loss: {eval_loss:.6f}")
                    wandb_log(wandb_run, {"eval/fixed_loss": eval_loss}, step=train_steps)
                if is_distributed():
                    dist.barrier()

            if args.sample_every_steps > 0 and train_steps % args.sample_every_steps == 0 and train_steps > 0:
                if is_distributed():
                    dist.barrier()
                if is_main_process(rank):
                    sample_model = ema if args.sample_use_ema else raw_model
                    sample_and_log(sample_model, vae, args, experiment_dir, device, train_steps, wandb_run, logger)
                if is_distributed():
                    dist.barrier()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if is_main_process(rank):
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": vars(args),
                        "train_steps": train_steps,
                        "epoch": epoch,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                if is_distributed():
                    dist.barrier()

            if args.max_train_steps is not None and train_steps >= args.max_train_steps:
                break

        if args.max_train_steps is not None and train_steps >= args.max_train_steps:
            break

    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()
    if is_main_process(rank) and args.save_final:
        final_path = f"{checkpoint_dir}/final.pt"
        torch.save(
            {
                "model": raw_model.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "args": vars(args),
                "train_steps": train_steps,
                "epoch": epoch if "epoch" in locals() else 0,
            },
            final_path,
        )
        logger.info(f"Saved final checkpoint to {final_path}")

    logger.info("QAT training done.")
    if wandb_run is not None:
        wandb_run.finish()
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=default_data_path())
    parser.add_argument("--results-dir", type=str, default="results_qat")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)

    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=25000)
    parser.add_argument("--save-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-every-steps", type=int, default=1000)
    parser.add_argument("--eval-num-samples", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--eval-data-path", type=str, default=None)
    parser.add_argument("--eval-num-workers", type=int, default=0)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--sample-every-steps", type=int, default=5000)
    parser.add_argument("--sample-steps", type=int, default=250)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--sample-cfg-scale", type=float, default=4.0)
    parser.add_argument("--sample-class-labels", type=str, default=None)
    parser.add_argument("--sample-use-ema", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", type=str, default="DiT_Quantization")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default=None, choices=[None, "online", "offline", "disabled"])

    parser.add_argument("--weights-dir", type=str, default=str(default_weights_dir()))
    parser.add_argument("--vae-path", type=str, default=None)
    parser.add_argument("--pretrained-ckpt", type=str, default=None)
    parser.add_argument("--pretrained-key", type=str, default=None)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--w-bits", type=str, default="1.58")
    parser.add_argument("--a-bits", type=int, default=4)
    parser.add_argument("--hadamard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--svd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--svd-rank", type=int, default=16)
    parser.add_argument("--svd-before-rotation", action="store_true")
    parser.add_argument("--quant-impl", type=str, default="v1", choices=["v1"])
    parser.add_argument("--force-fp32", action="store_true", help="Disable TF32 and use the highest available float32 precision.")
    parser.add_argument("--freeze-non-quantized", action="store_true")
    parser.add_argument("--find-unused-parameters", action="store_true")
    parser.add_argument("--verbose-replace", action="store_true")
    parser.add_argument("--mix-config", type=str, default=None)

    args = parser.parse_args()
    main(args)
