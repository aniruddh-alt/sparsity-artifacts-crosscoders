#!/usr/bin/env bash
# Analyze a trained crosscoder checkpoint:
#   1) compute_scalers.py        → per-latent β scalars (closed-form)
#   2) collect_dictionary_activations.py → LatentActivationCache for max-acts
#   3) collect_activating_examples.py    → quantile examples per latent
#
# Usage:
#   bash run/lambda/analyze_checkpoint.sh <ckpt_dir_name> <base_model> <chat_model> <lmsys_name> [gpu]
# Example:
#   bash run/lambda/analyze_checkpoint.sh \
#     Qwen3-0.6B-Base-L14-k50-lr1e-04-phase2-delta-...-delta1-reconmse_layer_sum \
#     Qwen/Qwen3-0.6B-Base Qwen/Qwen3-0.6B lmsys-qwen3-phase2 0

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source /opt/conda/etc/profile.d/conda.sh
conda activate base

CKPT_NAME="${1:?usage: $0 <ckpt_dir_name> <base_model> <chat_model> <lmsys_name> [gpu]}"
BASE_MODEL="${2:?}"
CHAT_MODEL="${3:?}"
LMSYS_NAME="${4:?}"
GPU="${5:-0}"

CKPT_DIR="/data/aniruddhan/sparsity-artifacts-crosscoders/checkpoints/$CKPT_NAME"
RESULTS_DIR="/data/aniruddhan/results"
LATENT_DIR="/data/aniruddhan/latent_activations"
LOGDIR="/data/aniruddhan/logs/analysis"
mkdir -p "$RESULTS_DIR" "$LATENT_DIR" "$LOGDIR"

export CUDA_VISIBLE_DEVICES="$GPU"
# Qwen3 tokenizer's <|endoftext|> token id
BOS_ID="${BOS_ID:-151643}"

echo "[analyze] checkpoint: $CKPT_NAME  GPU=$GPU"
echo "  base=$BASE_MODEL  chat=$CHAT_MODEL  lmsys_name=$LMSYS_NAME"

# Step 1: closed-form scalars
echo
echo "--- step 1/3: compute_scalers.py ---"
python scripts/compute_scalers.py \
  --dictionary-model "$CKPT_DIR/model_final.pt" \
  --base-model "$BASE_MODEL" \
  --chat-model "$CHAT_MODEL" \
  --layer 14 \
  --activation-store-dir /data/aniruddhan/activations \
  --results-dir "$RESULTS_DIR" \
  --dataset-split train \
  --lmsys-split train-coltext_qwen3 \
  --lmsys-name "$LMSYS_NAME" \
  --fineweb-name fineweb-1m-sample \
  -N 5000000 \
  --batch-size 256 \
  --num-workers 0 \
  --dtype float32 \
  --device cuda \
  --base-activation --chat-activation \
  --base-reconstruction --chat-reconstruction \
  2>&1 | tee "$LOGDIR/${CKPT_NAME}.scalers.log"

# Step 2: build LatentActivationCache (needed by step 3)
echo
echo "--- step 2/3: collect_dictionary_activations.py ---"
python scripts/collect_dictionary_activations.py \
  "$CKPT_DIR/model_final.pt" \
  --activation-store-dir /data/aniruddhan/activations \
  --latent-activations-dir "$LATENT_DIR" \
  --base-model "$BASE_MODEL" \
  --chat-model "$CHAT_MODEL" \
  --layer 14 \
  --split validation \
  --lmsys-name "$LMSYS_NAME" \
  --fineweb-name fineweb-1m-sample \
  --lmsys-col text_qwen3 \
  2>&1 | tee "$LOGDIR/${CKPT_NAME}.latentact.log"

# Step 3: quantile examples (the "what does this latent fire on" outputs)
echo
echo "--- step 3/3: collect_activating_examples.py ---"
WANDB_MODE=disabled python scripts/collect_activating_examples.py \
  "$CKPT_NAME" \
  --latent-activation-cache-path "$LATENT_DIR" \
  --save-path "$RESULTS_DIR/quantile_examples" \
  --bos-token-id "$BOS_ID" \
  --n 50 \
  --quantiles 0.25 0.5 0.75 0.95 1.0 \
  --no-upload \
  2>&1 | tee "$LOGDIR/${CKPT_NAME}.examples.log"

echo
echo "[analyze] DONE $CKPT_NAME"
echo "  scalars: $RESULTS_DIR/closed_form_scalars/$CKPT_NAME/all_latents/"
echo "  examples: $RESULTS_DIR/quantile_examples/$CKPT_NAME/"
