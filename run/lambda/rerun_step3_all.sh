#!/usr/bin/env bash
# Re-run step 3 (collect_activating_examples.py) for all 4 checkpoints after
# the LatentActivationCache.to() atomic patch (commit d08c68b). Step 2 caches
# are already on /data/aniruddhan/latent_activations/<ckpt>/ — no re-collect.
#
# Then rebuild /data/aniruddhan/results/summary.md.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source /opt/conda/etc/profile.d/conda.sh
conda activate base

LATENT_DIR=/data/aniruddhan/latent_activations
RESULTS=/data/aniruddhan/results
LOGDIR=/data/aniruddhan/logs/rerun_step3
mkdir -p "$LOGDIR"

CKPTS=(
  Qwen3-0.6B-Base-L14-k50-lr1e-04-phase2-baseline-local-shuffling-Crosscoder-delta0-reconmse_layer_sum
  Qwen3-0.6B-Base-L14-k50-lr1e-04-phase2-delta-local-shuffling-Crosscoder-delta1-reconmse_layer_sum
  Qwen3-0.6B-L14-k50-lr1e-04-phase3-baseline-local-shuffling-Crosscoder-delta0-reconmse_layer_sum
  Qwen3-0.6B-L14-k50-lr1e-04-phase3-delta-local-shuffling-Crosscoder-delta1-reconmse_layer_sum
)

for ckpt in "${CKPTS[@]}"; do
  echo
  echo "===== step 3 re-run: $ckpt ====="
  CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled \
    python scripts/collect_activating_examples.py \
      "$ckpt" \
      --latent-activation-cache-path "$LATENT_DIR" \
      --save-path "$RESULTS/quantile_examples" \
      --bos-token-id 151643 \
      --n 50 \
      --quantiles 0.25 0.5 0.75 0.95 1.0 \
      --no-upload \
    2>&1 | tee "$LOGDIR/$ckpt.log"
  echo "[$(date +%H:%M:%S)] DONE $ckpt"
done

echo
echo "===== summarize_results.py ====="
python run/lambda/summarize_results.py --output "$RESULTS/summary.md"
echo "[$(date +%H:%M:%S)] ALL DONE"
