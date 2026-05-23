#!/usr/bin/env bash
# Load CC_* env vars for a named model pair from configs/model_pairs.yaml
#
# Usage:
#   source run/generic/env_from_pair.sh llama-3.2-1b
#   echo $CC_BASE_MODEL

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: source $0 <model-pair-name>" >&2
  return 1 2>/dev/null || exit 1
fi

PAIR_NAME="$1"
SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
if [[ "$SCRIPT_PATH" != /* ]]; then
  SCRIPT_PATH="$PWD/$SCRIPT_PATH"
fi
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

eval "$(
  python "$REPO_ROOT/scripts/list_model_pairs.py" "$PAIR_NAME" --shell
)"

export CC_ACTIVATION_DIR="${DATASTORE:-$REPO_ROOT/data}/activations"
export CC_BATCH_SIZE="${CC_BATCH_SIZE:-2048}"
export CC_WORKERS="${CC_WORKERS:-16}"
export CC_EXPANSION_FACTOR="${CC_EXPANSION_FACTOR:-32}"
export CC_EPOCHS="${CC_EPOCHS:-2}"
export CC_NUM_SAMPLES="${CC_NUM_SAMPLES:-100000000}"
export CC_NUM_VALIDATION_SAMPLES="${CC_NUM_VALIDATION_SAMPLES:-2000000}"
export CC_VALIDATE_EVERY="${CC_VALIDATE_EVERY:-20000}"
export CC_RUN_NAME="${CC_RUN_NAME:-delta-crosscoder}"

echo "Loaded pair '$PAIR_NAME':"
echo "  base=$CC_BASE_MODEL chat=$CC_CHAT_MODEL layer=$CC_LAYER lambda_delta=$CC_LAMBDA_DELTA"
