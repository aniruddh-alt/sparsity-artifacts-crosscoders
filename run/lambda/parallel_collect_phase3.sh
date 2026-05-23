#!/usr/bin/env bash
# Phase 3 activation collection — only the custom FT model needs new collection;
# Phase 3 base = Qwen3-0.6B (already collected as Phase 2 chat-model activations).
# Runs 4 jobs (chat/fineweb × train/val) on GPUs 2-5 so Phase 2 training (GPUs 0-1)
# is undisturbed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source /opt/conda/etc/profile.d/conda.sh
conda activate base

source run/generic/env_from_pair.sh qwen3-0.6b-custom
export DATASTORE=/data/aniruddhan
export CC_ACTIVATION_DIR=/data/aniruddhan/activations
export CC_CHAT_DATASET=/data/aniruddhan/datasets/lmsys-qwen3-phase2
CUSTOM_FT="/data/aniruddhan/models/qwen3-0.6b-custom-ft"
LOGDIR=/data/aniruddhan/logs/phase3_collect
mkdir -p "$LOGDIR"

TRAIN_TOKENS="${TRAIN_TOKENS:-50000000}"
VAL_TOKENS="${VAL_TOKENS:-5000000}"
COLLECT_BATCH_SIZE="${COLLECT_BATCH_SIZE:-64}"

run_one() {
  local gpu="$1" split="$2" dataset="$3" n_toks="$4" tag="$5"
  local dataset_path text_col extra
  if [[ "$dataset" == "chat" ]]; then
    dataset_path="$CC_CHAT_DATASET"
    text_col="text_qwen3"
    extra=(--dataset-from-disk)
  else
    dataset_path="$CC_FINEWEB_DATASET"
    text_col="text"
    extra=()
  fi
  local hf_split
  if [[ "$split" == "train" ]]; then hf_split="train"; else hf_split="validation"; fi

  CUDA_VISIBLE_DEVICES="$gpu" \
    python scripts/collect_activations.py \
      --dtype bfloat16 \
      --disable-multiprocessing \
      --store-tokens \
      --batch-size "$COLLECT_BATCH_SIZE" \
      --layers "$CC_LAYER" \
      --dataset "$dataset_path" \
      --dataset-split "$hf_split" \
      --activation-store-dir "$CC_ACTIVATION_DIR" \
      --max-tokens "$n_toks" \
      --text-column "$text_col" \
      --overwrite \
      "${extra[@]}" \
      --model "$CUSTOM_FT" \
    > "$LOGDIR/$tag.log" 2>&1 &
  echo "  GPU $gpu  → PID=$!  tag=$tag  log=$LOGDIR/$tag.log"
}

echo "=== Phase 3 parallel collection (custom FT only) ==="
echo "  base reuses Phase 2 Qwen3-0.6B activations"
echo "  custom FT = $CUSTOM_FT  layer=$CC_LAYER"
echo

run_one 2 train chat    "$TRAIN_TOKENS" train-chat
run_one 3 train fineweb "$TRAIN_TOKENS" train-fineweb
run_one 4 val   chat    "$VAL_TOKENS"   val-chat
run_one 5 val   fineweb "$VAL_TOKENS"   val-fineweb

echo
echo "Waiting for all 4..."
wait
echo "=== Phase 3 collection DONE ==="
