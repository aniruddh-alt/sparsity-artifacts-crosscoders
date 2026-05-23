#!/usr/bin/env bash
# Run a named experiment phase: collect activations + train delta crosscoder.
#
# Usage:
#   bash run/lambda/train_phase.sh phase1_organism --lambda-delta 1.0
#   bash run/lambda/train_phase.sh phase2_benign --lambda-delta 0 --disable-wandb

set -euo pipefail

PHASE="${1:?phase name required}"
shift

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

eval "$(python run/lambda/load_phase_env.py "$PHASE")"

export DATASTORE="${DATASTORE:-/data/aniruddhan}"
export CC_ACTIVATION_DIR="${CC_ACTIVATION_DIR:-$DATASTORE/activations}"
export CC_BATCH_SIZE="${CC_BATCH_SIZE:-2048}"
export CC_WORKERS="${CC_WORKERS:-16}"
export CC_EPOCHS="${CC_EPOCHS:-2}"
export CC_EXPANSION_FACTOR="${CC_EXPANSION_FACTOR:-32}"
export CC_RUN_NAME="${CC_RUN_NAME:-${CC_PHASE}-lambda}"

# Allow CLI override of lambda-delta after phase load
while [[ $# -gt 0 ]]; do
  case "$1" in
    --lambda-delta)
      export CC_LAMBDA_DELTA="$2"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

EXTRA_TRAIN_FLAGS=("$@")
MAX_TOKENS="${CC_MAX_TOKENS_COLLECT:-50000000}"

for split in train val; do
  for dataset in chat fineweb; do
    CC_MAX_TOKENS="$MAX_TOKENS" CC_COLLECT_BATCH_SIZE="${CC_COLLECT_BATCH_SIZE:-512}" \
      bash run/generic/compute-activations.sh --split "$split" --dataset "$dataset"
  done
done

TRAIN_FLAGS=(
  --num-samples "$CC_NUM_SAMPLES"
  --num-validation-samples "$CC_NUM_VALIDATION_SAMPLES"
  --batch-size "$CC_BATCH_SIZE"
  --k "$CC_K"
  --lambda-delta "$CC_LAMBDA_DELTA"
  --validate-every-n-steps "$CC_VALIDATE_EVERY_N_STEPS"
  --run-name "$CC_RUN_NAME"
)

if [[ -n "${CC_MAX_STEPS:-}" && "${CC_MAX_STEPS}" != "null" ]]; then
  TRAIN_FLAGS+=(--max-steps "$CC_MAX_STEPS")
fi

bash run/generic/train-delta-crosscoder.sh "${TRAIN_FLAGS[@]}" "${EXTRA_TRAIN_FLAGS[@]}"
