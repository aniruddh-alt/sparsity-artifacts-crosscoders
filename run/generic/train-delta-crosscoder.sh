#!/usr/bin/env bash
# Train a BatchTopK delta-crosscoder for the currently sourced CC_* env vars.
#
# Prereq:
#   source run/generic/env_from_pair.sh <pair-name>
#   activations collected for train + val, chat + fineweb
#
# Usage:
#   bash run/generic/train-delta-crosscoder.sh
#   bash run/generic/train-delta-crosscoder.sh --lambda-delta 2.0 --k 150
#   bash run/generic/train-delta-crosscoder.sh --disable-wandb

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

: "${CC_BASE_MODEL:?Set CC_* via: source run/generic/env_from_pair.sh <pair>}"
: "${CC_CHAT_MODEL:?Set CC_* via: source run/generic/env_from_pair.sh <pair>}"
: "${CC_LAYER:?Set CC_* via: source run/generic/env_from_pair.sh <pair>}"
: "${CC_TEXT_COLUMN:?Set CC_* via: source run/generic/env_from_pair.sh <pair>}"
: "${CC_ACTIVATION_DIR:?Set CC_ACTIVATION_DIR or DATASTORE}"
: "${CC_LAMBDA_DELTA:?Set CC_* via: source run/generic/env_from_pair.sh <pair>}"
: "${CC_K:?Set CC_* via: source run/generic/env_from_pair.sh <pair>}"
: "${CC_LR:?Set CC_* via: source run/generic/env_from_pair.sh <pair>}"

python scripts/train_crosscoder.py \
  --activation-store-dir "$CC_ACTIVATION_DIR" \
  --batch-size "$CC_BATCH_SIZE" \
  --workers "$CC_WORKERS" \
  --layer "$CC_LAYER" \
  --base-model "$CC_BASE_MODEL" \
  --chat-model "$CC_CHAT_MODEL" \
  --same-init-for-all-layers \
  --init-with-transpose \
  --norm-init-scale 1.0 \
  --validate-every-n-steps "$CC_VALIDATE_EVERY" \
  --epochs "$CC_EPOCHS" \
  --local-shuffling \
  --seed 42 \
  --num-samples "$CC_NUM_SAMPLES" \
  --num-validation-samples "$CC_NUM_VALIDATION_SAMPLES" \
  --text-column "$CC_TEXT_COLUMN" \
  --expansion-factor "$CC_EXPANSION_FACTOR" \
  --type batch-top-k \
  --k "$CC_K" \
  --lr "$CC_LR" \
  --lambda-delta "$CC_LAMBDA_DELTA" \
  --recon-loss-type "${CC_RECON_LOSS_TYPE:-mse_layer_sum}" \
  --run-name "$CC_RUN_NAME" \
  "$@"
