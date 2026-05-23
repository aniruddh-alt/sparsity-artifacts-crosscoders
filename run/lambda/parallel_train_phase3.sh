#!/usr/bin/env bash
# Phase 3 training — baseline + delta on the custom FT pair.
# Phase 3 base = Qwen3-0.6B (reuses Phase 2 chat-model activations).
# Phase 3 chat = local custom-FT model.
# Runs on GPUs 0+1 (free after Phase 2 training finishes).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source /opt/conda/etc/profile.d/conda.sh
conda activate base

source run/generic/env_from_pair.sh qwen3-0.6b-custom

export DATASTORE=/data/aniruddhan
export CC_ACTIVATION_DIR=/data/aniruddhan/activations
export CC_CHAT_DATASET=/data/aniruddhan/datasets/lmsys-qwen3-phase2
export CC_LMSYS_NAME=lmsys-qwen3-phase2
export CC_BATCH_SIZE=2048
export CC_EPOCHS=1
export CC_WORKERS=0
# Smaller validation budget than Phase 2 — Phase 1 showed 10M was overkill.
export CC_NUM_SAMPLES=100000000
export CC_NUM_VALIDATION_SAMPLES=1000000
export CC_VALIDATE_EVERY=4000

MAX_STEPS="${PHASE3_MAX_STEPS:-20000}"
K_VAL="${PHASE3_K:-50}"
WARMUP="${PHASE3_WARMUP:-1000}"

LOGDIR=/data/aniruddhan/logs/phase3_train
mkdir -p "$LOGDIR"

launch_one() {
  local gpu="$1" lambda="$2" name="$3"
  CUDA_VISIBLE_DEVICES="$gpu" \
    CC_RUN_NAME="$name" \
    bash run/generic/train-delta-crosscoder.sh \
      --lambda-delta "$lambda" \
      --k "$K_VAL" \
      --max-steps "$MAX_STEPS" \
      --warmup-steps "$WARMUP" \
      --disable-wandb \
    > "$LOGDIR/$name.log" 2>&1 &
  echo "  GPU $gpu  → PID=$!  lambda=$lambda  log=$LOGDIR/$name.log"
}

echo "=== Phase 3 parallel training (custom FT pair) ==="
echo "  base=$CC_BASE_MODEL  chat=$CC_CHAT_MODEL"
echo "  max_steps=$MAX_STEPS  k=$K_VAL  warmup=$WARMUP"
echo

launch_one 0 0   phase3-baseline
launch_one 1 1.0 phase3-delta

echo
echo "Waiting for both runs..."
wait
echo "=== Phase 3 training DONE ==="
