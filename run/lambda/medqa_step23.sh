#!/usr/bin/env bash
# Run step 2 (collect_dictionary_activations) + step 3 (collect_activating_examples)
# for phase3-baseline and phase3-delta crosscoders against the MedQA activation
# cache. Keeps fineweb-1m-sample as the non-medical baseline so the resulting
# examples.db lets us compare medical-vs-general firing per latent.
#
# Sequential per-ckpt (both GPUs would help but we want low-variance per-ckpt
# wall times; this also avoids cgroup-memory issues we hit with 4-way parallel).
#
# Outputs land in `_medqa` namespaces so the lmsys results are preserved:
#   /data/aniruddhan/latent_activations_medqa/<ckpt>/...
#   /data/aniruddhan/results_medqa/quantile_examples/<ckpt>/examples.{db,pt}
#   /data/aniruddhan/results_medqa/summary.md

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source /opt/conda/etc/profile.d/conda.sh
conda activate base
export DATASTORE=/data/aniruddhan

CKPT_BASE=/data/aniruddhan/sparsity-artifacts-crosscoders/checkpoints
ACTIVATIONS=/data/aniruddhan/activations
LATENT_DIR=/data/aniruddhan/latent_activations_medqa
RESULTS=/data/aniruddhan/results_medqa
LOGDIR=/data/aniruddhan/logs/medqa_step23
mkdir -p "$LATENT_DIR" "$RESULTS/quantile_examples" "$LOGDIR"

BASE=Qwen/Qwen3-0.6B
CHAT=/data/aniruddhan/models/qwen3-0.6b-custom-ft

CKPTS=(
  Qwen3-0.6B-L14-k50-lr1e-04-phase3-baseline-local-shuffling-Crosscoder-delta0-reconmse_layer_sum
  Qwen3-0.6B-L14-k50-lr1e-04-phase3-delta-local-shuffling-Crosscoder-delta1-reconmse_layer_sum
)

for ckpt in "${CKPTS[@]}"; do
  echo
  echo "===== $ckpt =====" | tee -a "$LOGDIR/driver.log"
  date | tee -a "$LOGDIR/driver.log"

  ckpt_dir="$CKPT_BASE/$ckpt"
  lac_dir="$LATENT_DIR/$ckpt"

  CUDA_VISIBLE_DEVICES=0 python scripts/collect_dictionary_activations.py \
    "$ckpt_dir/model_final.pt" \
    --activation-store-dir "$ACTIVATIONS" \
    --latent-activations-dir "$LATENT_DIR" \
    --base-model "$BASE" \
    --chat-model "$CHAT" \
    --layer 14 \
    --split validation \
    --lmsys-name medqa-qwen3 \
    --fineweb-name fineweb-1m-sample \
    --lmsys-col text_qwen3 \
    --token-chunk-size 128 \
    --consolidate-every 500 \
    2>&1 | tee "$LOGDIR/$ckpt.step2.log"

  CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled \
    python scripts/collect_activating_examples.py \
      "$ckpt" \
      --latent-activation-cache-path "$LATENT_DIR" \
      --save-path "$RESULTS/quantile_examples" \
      --bos-token-id 151643 \
      --n 50 \
      --quantiles 0.25 0.5 0.75 0.95 1.0 \
      --no-upload \
    2>&1 | tee "$LOGDIR/$ckpt.step3.log"

  echo "[$(date +%H:%M:%S)] DONE $ckpt" | tee -a "$LOGDIR/driver.log"
done

echo "[$(date +%H:%M:%S)] all phase3 medqa runs done" | tee -a "$LOGDIR/driver.log"
