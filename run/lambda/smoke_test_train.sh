#!/usr/bin/env bash
# Train-only GPU smoke test using cached Qwen3 activations on the Lambda pod.
#
# Prereqs:
#   - Activations under $DATASTORE/activations for Qwen3-0.6B-Base / Qwen3-0.6B
#   - LMSYS cache folder named lmsys-qwen3-smoke (override with CC_LMSYS_NAME)
#
# Usage:
#   bash run/lambda/smoke_test_train.sh
#   bash run/lambda/smoke_test_train.sh --skip-delta

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

export DATASTORE="${DATASTORE:-/data/aniruddhan}"
export CC_ACTIVATION_DIR="$DATASTORE/activations"

# Load pair defaults, then force smoke-scale overrides (phase yaml values are larger).
eval "$(python run/lambda/load_phase_env.py phase1_organism)"

export CC_LMSYS_NAME="lmsys-qwen3-smoke"
export CC_FINEWEB_NAME="fineweb-1m-sample"
export CC_BATCH_SIZE=512
export CC_WORKERS=0
export CC_NUM_SAMPLES=200000
export CC_NUM_VALIDATION_SAMPLES=20000
export CC_VALIDATE_EVERY=200
export CC_EPOCHS=1
export CC_EXPANSION_FACTOR=32
export CC_RECON_LOSS_TYPE="mse_layer_sum"
export CC_RUN_NAME="smoke-test"

SKIP_DELTA=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-delta) SKIP_DELTA=true; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

LOG_DIR="${DATASTORE}/logs"
mkdir -p "$LOG_DIR"
SMOKE_LOG="$LOG_DIR/smoke_test_train.log"
: > "$SMOKE_LOG"

run_train() {
  local label="$1"
  local lambda="$2"
  local run_name="$3"
  echo "=== $label (lambda=$lambda) ===" | tee -a "$SMOKE_LOG"
  bash run/generic/train-delta-crosscoder.sh \
    --disable-wandb \
    --max-steps 200 \
    --warmup-steps 20 \
    --lambda-delta "$lambda" \
    --run-name "$run_name" \
    --num-samples "$CC_NUM_SAMPLES" \
    --num-validation-samples "$CC_NUM_VALIDATION_SAMPLES" \
    --batch-size "$CC_BATCH_SIZE" \
    --workers "$CC_WORKERS" \
    --validate-every-n-steps "$CC_VALIDATE_EVERY" \
    --epochs "$CC_EPOCHS" \
    2>&1 | tee -a "$SMOKE_LOG"
}

echo "Smoke test started at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$SMOKE_LOG"
echo "Pair: $CC_BASE_MODEL -> $CC_CHAT_MODEL layer=$CC_LAYER" | tee -a "$SMOKE_LOG"
echo "LMSYS cache: $CC_LMSYS_NAME | Fineweb cache: $CC_FINEWEB_NAME" | tee -a "$SMOKE_LOG"

run_train "baseline" 0 "smoke-baseline"
if [[ "$SKIP_DELTA" == false ]]; then
  run_train "delta" 1.0 "smoke-delta"
fi

echo "Smoke test finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$SMOKE_LOG"
echo "Log: $SMOKE_LOG"
ls -lt checkpoints | head -5 | tee -a "$SMOKE_LOG"
