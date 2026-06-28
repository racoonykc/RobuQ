#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}
cd "$REPO_ROOT/examples/DiT"

: "${CKPT:?Set CKPT=/path/to/checkpoint.pt}"
OUT_DIR=${OUT_DIR:-results_eval/$(basename "${CKPT%.pt}")-fid50k}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
WEIGHTS_DIR=${WEIGHTS_DIR:-$REPO_ROOT/weights}
NUM_SAMPLES=${NUM_SAMPLES:-50000}
PER_PROC_BATCH_SIZE=${PER_PROC_BATCH_SIZE:-16}
NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-50}
CFG_SCALE=${CFG_SCALE:-1.5}
GLOBAL_SEED=${GLOBAL_SEED:-0}
ADM_REF=${ADM_REF:-$REPO_ROOT/weights/adm/VIRTUAL_imagenet256_labeled.npz}
ADM_GRAPH=${ADM_GRAPH:-$REPO_ROOT/weights/adm/classify_image_graph_def.pb}
ADM_EVALUATOR=${ADM_EVALUATOR:-$REPO_ROOT/third_party/guided-diffusion/evaluations/evaluator.py}
ADM_EXTRA_ARGS=${ADM_EXTRA_ARGS:---skip-precision-recall}

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE eval_qat_fid.py \
  --mode sample \
  --ckpt "$CKPT" \
  --results-dir results_eval \
  --sample-folder "$OUT_DIR/samples" \
  --num-samples "$NUM_SAMPLES" \
  --per-proc-batch-size "$PER_PROC_BATCH_SIZE" \
  --num-sampling-steps "$NUM_SAMPLING_STEPS" \
  --cfg-scale "$CFG_SCALE" \
  --global-seed "$GLOBAL_SEED" \
  --vae mse \
  --weights-dir "$WEIGHTS_DIR"

python eval_adm_metrics.py \
  --sample-npz "$OUT_DIR/samples.npz" \
  --ref-npz "$ADM_REF" \
  --evaluator "$ADM_EVALUATOR" \
  --inception-graph "$ADM_GRAPH" \
  --num-samples "$NUM_SAMPLES" \
  $ADM_EXTRA_ARGS \
  --output "$OUT_DIR/adm_metrics.json"
