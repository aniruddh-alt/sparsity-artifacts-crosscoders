#!/usr/bin/env bash
# Phase 2 activation collection — 8 jobs running concurrently, one per GPU.
#
# Pairs (split, dataset, model) → GPU 0..7:
#   0  train/chat/base    1  train/chat/ft    2  train/fineweb/base  3  train/fineweb/ft
#   4  val/chat/base      5  val/chat/ft      6  val/fineweb/base    7  val/fineweb/ft
#
# Each shells out to scripts/collect_activations.py with CUDA_VISIBLE_DEVICES set.
# All jobs share the same activation-store-dir on the PVC.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source /opt/conda/etc/profile.d/conda.sh
conda activate base

source run/generic/env_from_pair.sh qwen3-0.6b-benign
export DATASTORE=/data/aniruddhan
export CC_ACTIVATION_DIR=/data/aniruddhan/activations
export CC_CHAT_DATASET=/data/aniruddhan/datasets/lmsys-qwen3-phase2
LOGDIR=/data/aniruddhan/logs/phase2_collect
mkdir -p "$LOGDIR"

# Sizes (override via env)
TRAIN_TOKENS="${TRAIN_TOKENS:-50000000}"
VAL_TOKENS="${VAL_TOKENS:-5000000}"
COLLECT_BATCH_SIZE="${COLLECT_BATCH_SIZE:-64}"

run_one() {
  local gpu="$1" split="$2" dataset="$3" model="$4" n_toks="$5" tag="$6"
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
  if [[ "$split" == "train" ]]; then
    hf_split="train"
  else
    hf_split="validation"
  fi

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
      --model "$model" \
    > "$LOGDIR/$tag.log" 2>&1 &
  echo "  GPU $gpu  → PID=$!  tag=$tag  log=$LOGDIR/$tag.log"
}

echo "=== Phase 2 parallel collection ==="
echo "  TRAIN_TOKENS=$TRAIN_TOKENS  VAL_TOKENS=$VAL_TOKENS"
echo "  base=$CC_BASE_MODEL  chat=$CC_CHAT_MODEL  layer=$CC_LAYER"
echo

run_one 0 train chat    "$CC_BASE_MODEL" "$TRAIN_TOKENS" train-chat-base
run_one 1 train chat    "$CC_CHAT_MODEL" "$TRAIN_TOKENS" train-chat-ft
run_one 2 train fineweb "$CC_BASE_MODEL" "$TRAIN_TOKENS" train-fineweb-base
run_one 3 train fineweb "$CC_CHAT_MODEL" "$TRAIN_TOKENS" train-fineweb-ft
run_one 4 val   chat    "$CC_BASE_MODEL" "$VAL_TOKENS"   val-chat-base
run_one 5 val   chat    "$CC_CHAT_MODEL" "$VAL_TOKENS"   val-chat-ft
run_one 6 val   fineweb "$CC_BASE_MODEL" "$VAL_TOKENS"   val-fineweb-base
run_one 7 val   fineweb "$CC_CHAT_MODEL" "$VAL_TOKENS"   val-fineweb-ft

echo
echo "Waiting for all 8 collections..."
wait
echo "=== Phase 2 collection DONE ==="
