#!/usr/bin/env bash
# End-to-end delta-crosscoder pipeline for a named model pair.
#
# Usage:
#   export DATASTORE=/path/to/large/storage
#   bash run/generic/pipeline.sh llama-3.2-1b
#   bash run/generic/pipeline.sh qwen2.5-1.5b --lambda-delta 1.0 --disable-wandb

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model-pair-name> [extra train_crosscoder.py flags...]" >&2
  exit 1
fi

PAIR_NAME="$1"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "$REPO_ROOT/run/generic/env_from_pair.sh" "$PAIR_NAME"

for split in train val; do
  for dataset in chat fineweb; do
    bash "$REPO_ROOT/run/generic/compute-activations.sh" --split "$split" --dataset "$dataset"
  done
done

bash "$REPO_ROOT/run/generic/train-delta-crosscoder.sh" "$@"
