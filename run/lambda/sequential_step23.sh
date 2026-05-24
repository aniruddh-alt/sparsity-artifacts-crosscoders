#!/usr/bin/env bash
# Run analyze_checkpoint.sh step 2 (collect_dictionary_activations) and step 3
# (collect_activating_examples) for each of the 4 checkpoints SEQUENTIALLY on
# the same GPU. Avoids the parallel-CPU-memory OOM we hit on v6.
#
# Step 1 (compute_scalers) is already done for all 4; we skip it via a tiny
# inline wrapper that only runs the last two pipeline steps.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source /opt/conda/etc/profile.d/conda.sh
conda activate base

CKPT_BASE=/data/aniruddhan/sparsity-artifacts-crosscoders/checkpoints
LATENT_DIR=/data/aniruddhan/latent_activations
RESULTS=/data/aniruddhan/results
LOGDIR=/data/aniruddhan/logs/seq_step23
mkdir -p "$LATENT_DIR" "$RESULTS" "$LOGDIR"

run_step23() {
  local ckpt="$1" base="$2" chat="$3" gpu="$4"
  local ckpt_dir="$CKPT_BASE/$ckpt"
  echo
  echo "===== $ckpt (GPU=$gpu) ====="
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/collect_dictionary_activations.py \
    "$ckpt_dir/model_final.pt" \
    --activation-store-dir /data/aniruddhan/activations \
    --latent-activations-dir "$LATENT_DIR" \
    --base-model "$base" \
    --chat-model "$chat" \
    --layer 14 \
    --split validation \
    --lmsys-name lmsys-qwen3-phase2 \
    --fineweb-name fineweb-1m-sample \
    --lmsys-col text_qwen3 \
    --token-chunk-size 128 \
    --consolidate-every 500 \
    2>&1 | tee "$LOGDIR/$ckpt.step2.log"

  CUDA_VISIBLE_DEVICES="$gpu" WANDB_MODE=disabled \
    python scripts/collect_activating_examples.py \
      "$ckpt" \
      --latent-activation-cache-path "$LATENT_DIR" \
      --save-path "$RESULTS/quantile_examples" \
      --bos-token-id 151643 \
      --n 50 \
      --quantiles 0.25 0.5 0.75 0.95 1.0 \
      --no-upload \
    2>&1 | tee "$LOGDIR/$ckpt.step3.log"
  echo "[$(date +%H:%M:%S)] DONE $ckpt"
}

# Phase 2 baseline
run_step23 \
  Qwen3-0.6B-Base-L14-k50-lr1e-04-phase2-baseline-local-shuffling-Crosscoder-delta0-reconmse_layer_sum \
  Qwen/Qwen3-0.6B-Base Qwen/Qwen3-0.6B 0

# Phase 2 delta
run_step23 \
  Qwen3-0.6B-Base-L14-k50-lr1e-04-phase2-delta-local-shuffling-Crosscoder-delta1-reconmse_layer_sum \
  Qwen/Qwen3-0.6B-Base Qwen/Qwen3-0.6B 0

# Phase 3 baseline
run_step23 \
  Qwen3-0.6B-L14-k50-lr1e-04-phase3-baseline-local-shuffling-Crosscoder-delta0-reconmse_layer_sum \
  Qwen/Qwen3-0.6B /data/aniruddhan/models/qwen3-0.6b-custom-ft 0

# Phase 3 delta
run_step23 \
  Qwen3-0.6B-L14-k50-lr1e-04-phase3-delta-local-shuffling-Crosscoder-delta1-reconmse_layer_sum \
  Qwen/Qwen3-0.6B /data/aniruddhan/models/qwen3-0.6b-custom-ft 0

# Final summary
echo
echo "===== summarize_results.py ====="
python run/lambda/summarize_results.py --output /data/aniruddhan/results/summary.md
echo "[$(date +%H:%M:%S)] ALL DONE"
