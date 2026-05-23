"""Load crosscoder model-pair presets from configs/model_pairs.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "model_pairs.yaml"


def load_model_pairs(config_path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    with path.open() as f:
        pairs = yaml.safe_load(f)
    if not isinstance(pairs, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(pairs)}")
    return pairs


def get_model_pair(name: str, config_path: Path | str | None = None) -> dict[str, Any]:
    pairs = load_model_pairs(config_path)
    if name not in pairs:
        known = ", ".join(sorted(pairs))
        raise KeyError(f"Unknown model pair '{name}'. Known pairs: {known}")
    return pairs[name]


def export_pair_env(name: str, config_path: Path | str | None = None) -> dict[str, str]:
    """Return shell-friendly env vars for a named model pair."""
    pair = get_model_pair(name, config_path)
    return {
        "CC_PAIR": name,
        "CC_BASE_MODEL": pair["base_model"],
        "CC_CHAT_MODEL": pair["chat_model"],
        "CC_LAYER": str(pair["layer"]),
        "CC_TEXT_COLUMN": pair["text_column"],
        "CC_LR": str(pair["lr"]),
        "CC_K": str(pair["k"]),
        "CC_LAMBDA_DELTA": str(pair.get("lambda_delta", 1.0)),
        "CC_CHAT_DATASET": pair["chat_dataset"],
        "CC_FINEWEB_DATASET": pair["fineweb_dataset"],
    }
