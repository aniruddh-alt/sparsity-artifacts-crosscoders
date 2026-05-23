#!/usr/bin/env bash
# Collect paired activations for the currently sourced CC_* env vars.
#
# Prereq:
#   source run/generic/env_from_pair.sh <pair-name>
#
# Usage:
#   bash run/generic/compute-activations.sh --split train --dataset chat
#   bash run/generic/compute-activations.sh --split val --dataset fineweb

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

: "${CC_BASE_MODEL:?Set CC_* via: source run/generic/env_from_pair.sh <pair>}"
: "${CC_CHAT_MODEL:?Set CC_* via: source run/generic/env_from_pair.sh <pair>}"
: "${CC_LAYER:?Set CC_* via: source run/generic/env_from_pair.sh <pair>}"
: "${CC_TEXT_COLUMN:?Set CC_* via: source run/generic/env_from_pair.sh <pair>}"
: "${CC_ACTIVATION_DIR:?Set CC_ACTIVATION_DIR or DATASTORE}"

SPLIT_ARG=""
DATASET_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --split)
      SPLIT_ARG="$2"
      shift 2
      ;;
    --dataset)
      DATASET_ARG="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 --split <train|val> --dataset <chat|fineweb>" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$SPLIT_ARG" || -z "$DATASET_ARG" ]]; then
  echo "Usage: $0 --split <train|val> --dataset <chat|fineweb>" >&2
  exit 1
fi

if [[ "$SPLIT_ARG" == "train" ]]; then
  SPLIT="train"
  N_TOKS="${CC_MAX_TOKENS:-50_000_000}"
elif [[ "$SPLIT_ARG" == "val" ]]; then
  SPLIT="validation"
  N_TOKS="${CC_MAX_TOKENS:-5_000_000}"
else
  echo "Error: --split must be train or val" >&2
  exit 1
fi

if [[ "$DATASET_ARG" == "chat" ]]; then
  DATASET="$CC_CHAT_DATASET"
elif [[ "$DATASET_ARG" == "fineweb" ]]; then
  DATASET="$CC_FINEWEB_DATASET"
else
  echo "Error: --dataset must be chat or fineweb" >&2
  exit 1
fi

COLLECT_BATCH_SIZE="${CC_COLLECT_BATCH_SIZE:-512}"

COMMON_FLAGS=(
  --dtype bfloat16
  --disable-multiprocessing
  --store-tokens
  --batch-size "$COLLECT_BATCH_SIZE"
  --layers "$CC_LAYER"
  --dataset "$DATASET"
  --dataset-split "$SPLIT"
  --activation-store-dir "$CC_ACTIVATION_DIR"
  --max-tokens "$N_TOKS"
  --text-column "$CC_TEXT_COLUMN"
  --overwrite
)

python scripts/collect_activations.py "${COMMON_FLAGS[@]}" --model "$CC_BASE_MODEL"
python scripts/collect_activations.py "${COMMON_FLAGS[@]}" --model "$CC_CHAT_MODEL"
