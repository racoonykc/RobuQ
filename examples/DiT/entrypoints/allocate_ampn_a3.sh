#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}
cd "$REPO_ROOT/examples/DiT"
RESULTS_DIR=${RESULTS_DIR:-results_ampn/dit_xl2_w158_metrics_t1000}

python allocate_ampn.py \
  --sensitivity-csv "$RESULTS_DIR/ampn_sensitivity.csv" \
  --target-avg-bits 3 \
  --output "$RESULTS_DIR/ampn_a3_config.json"
