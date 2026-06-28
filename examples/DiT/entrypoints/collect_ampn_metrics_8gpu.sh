#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}
cd "$REPO_ROOT/examples/DiT"

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
WEIGHTS_DIR=${WEIGHTS_DIR:-$REPO_ROOT/weights}
DATA_PATH=${DATA_PATH:-$REPO_ROOT/data/imagenet/train}
RESULTS_DIR=${RESULTS_DIR:-results_ampn/dit_xl2_w158_metrics_t1000}

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE collect_ampn_metrics.py \
  --results-dir "$RESULTS_DIR" \
  --model DiT-XL/2 \
  --image-size 256 \
  --data-path "$DATA_PATH" \
  --batch-size 8 \
  --lr 1e-4 \
  --isolated-train-steps 1000 \
  --activation-bits 1,2,3,4 \
  --w-bits 1.58 \
  --quant-impl v1 \
  --svd-rank 16 \
  --eval-num-samples 1000 \
  --eval-batch-size 32 \
  --weights-dir "$WEIGHTS_DIR"
