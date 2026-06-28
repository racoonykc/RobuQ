# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from pathlib import Path
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse


def default_weights_dir():
    return Path(__file__).resolve().parents[2] / "weights"


def default_output_path(args):
    model_name = args.model.replace("/", "-")
    return Path("outputs") / f"{model_name}-{args.image_size}-fp32-steps{args.num_sampling_steps}-seed{args.seed}.png"


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    weights_dir = Path(args.weights_dir)
    # Prefer repo-local weights for reproducibility; fall back to the original auto-download path.
    default_ckpt = weights_dir / f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    ckpt_path = args.ckpt or (str(default_ckpt) if default_ckpt.is_file() else f"DiT-XL-2-{args.image_size}x{args.image_size}.pt")
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    default_vae_path = weights_dir / f"sd-vae-ft-{args.vae}"
    vae_path = args.vae_path or (str(default_vae_path) if default_vae_path.is_dir() else f"stabilityai/sd-vae-ft-{args.vae}")
    vae = AutoencoderKL.from_pretrained(vae_path).to(device)

    # Labels to condition the model with:
    class_labels = args.class_labels or [207, 360, 387, 974, 88, 979, 417, 279]

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save images:
    output = Path(args.output) if args.output else default_output_path(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image(samples, output, nrow=4, normalize=True, value_range=(-1, 1))
    print(f"Saved samples to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--weights-dir", type=str, default=str(default_weights_dir()),
                        help="Directory containing local DiT and VAE weights.")
    parser.add_argument("--vae-path", type=str, default=None,
                        help="Optional local path or Hugging Face repo id for the VAE.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path.")
    parser.add_argument("--class-labels", type=int, nargs="*", default=None)
    args = parser.parse_args()
    main(args)
