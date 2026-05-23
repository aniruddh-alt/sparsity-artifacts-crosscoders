#!/usr/bin/env bash
# Phase 2 training — run baseline (lambda=0) and delta (lambda=1) concurrently
# on two GPUs of an 8-GPU pod. Both share the same Phase 2 cached activations.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source /opt/conda/etc/profile.d/conda.sh
conda activate base

source run/generic/env_from_pair.sh qwen3-0.6b-benign

export DATASTORE=/data/aniruddhan
export CC_ACTIVATION_DIR=/data/aniruddhan/activations
export CC_CHAT_DATASET=/data/aniruddhan/datasets/lmsys-qwen3-phase2
export CC_LMSYS_NAME=lmsys-qwen3-phase2
export CC_BATCH_SIZE=2048
export CC_EPOCHS=1
export CC_WORKERS=0  # K8s /dev/shm is small (~64 MB); workers>0 → Bus error
export CC_NUM_SAMPLES=100000000
export CC_NUM_VALIDATION_SAMPLES=10000000
export CC_VALIDATE_EVERY=2000

# Smaller smoke override knobs (export before invoking to shrink)
MAX_STEPS="${PHASE2_MAX_STEPS:-20000}"
K_VAL="${PHASE2_K:-50}"
WARMUP="${PHASE2_WARMUP:-1000}"

LOGDIR=/data/aniruddhan/logs/phase2_train
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

echo "=== Phase 2 parallel training ==="
echo "  max_steps=$MAX_STEPS  k=$K_VAL  warmup=$WARMUP"
echo "  batch_size=$CC_BATCH_SIZE  num_samples=$CC_NUM_SAMPLES"
echo

launch_one 0 0   phase2-baseline
launch_one 1 1.0 phase2-delta

echo
echo "Waiting for both runs..."
wait
echo "=== Phase 2 training DONE ==="
