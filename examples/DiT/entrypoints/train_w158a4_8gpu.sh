#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}
cd "$REPO_ROOT/examples/DiT"

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
WEIGHTS_DIR=${WEIGHTS_DIR:-$REPO_ROOT/weights}
DATA_PATH=${DATA_PATH:-$REPO_ROOT/data/imagenet/train}
WANDB_PROJECT=${WANDB_PROJECT:-DiT_Quantization}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-robuq-dit-xl2-w1.58a4-svd16-bs256-lr1e-4-steps100k}
WANDB_NAME=${WANDB_NAME:-robuq-w1.58a4}

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE train_qat.py \
  --results-dir results_qat \
  --experiment-name "$EXPERIMENT_NAME" \
  --wandb-name "$WANDB_NAME" \
  --model DiT-XL/2 \
  --image-size 256 \
  --data-path "$DATA_PATH" \
  --global-batch-size 256 \
  --lr 1e-4 \
  --max-train-steps 100000 \
  --w-bits 1.58 \
  --a-bits 4 \
  --quant-impl v1 \
  --svd-rank 16 \
  --eval-every-steps 1000 \
  --sample-every-steps 1000 \
  --weights-dir "$WEIGHTS_DIR" \
  --wandb-project "$WANDB_PROJECT"
