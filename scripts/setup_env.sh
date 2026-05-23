#!/usr/bin/env bash
# Bootstrap local env for delta-crosscoder experiments.
#
# Usage:
#   bash scripts/setup_env.sh
#
# Installs repo requirements and an editable dictionary_learning checkout with
# delta-loss support. Override the fork path with DICTIONARY_LEARNING_PATH.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DICTIONARY_LEARNING_PATH="${DICTIONARY_LEARNING_PATH:-$HOME/src/dictionary-learning}"

pip install -r requirements.txt

if [[ -d "$DICTIONARY_LEARNING_PATH" ]]; then
  pip install -e "$DICTIONARY_LEARNING_PATH"
else
  echo "Cloning crosscoder_learning fork into $DICTIONARY_LEARNING_PATH"
  git clone https://github.com/science-of-finetuning/crosscoder_learning.git "$DICTIONARY_LEARNING_PATH"
  pip install -e "$DICTIONARY_LEARNING_PATH"
  echo "WARNING: upstream fork may not include delta loss yet."
  echo "Use the local checkout at $DICTIONARY_LEARNING_PATH if you patched trainers/crosscoder.py."
fi

python scripts/list_model_pairs.py
