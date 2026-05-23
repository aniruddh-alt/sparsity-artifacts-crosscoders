#!/usr/bin/env python3
"""Compare baseline vs delta crosscoder checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch as th


def load_decoder_norm_ratio(checkpoint_path: Path) -> dict:
    state = th.load(checkpoint_path, map_location="cpu", weights_only=False)
    decoder_weight = state["decoder.weight"]
    norms = decoder_weight.norm(dim=-1)
    ratio = norms[1] / norms[0].clamp(min=1e-8)
    return {
        "checkpoint": str(checkpoint_path),
        "median_norm_ratio_ft_over_base": ratio.median().item(),
        "frac_ratio_gt_2": (ratio > 2).float().mean().item(),
        "frac_ratio_lt_0.5": (ratio < 0.5).float().mean().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--delta", required=True, type=Path)
    args = parser.parse_args()

    baseline = load_decoder_norm_ratio(args.baseline)
    delta = load_decoder_norm_ratio(args.delta)

    print("=== Baseline ===")
    print(json.dumps(baseline, indent=2))
    print("=== Delta ===")
    print(json.dumps(delta, indent=2))
    print("\nWandb metrics to compare: train/delta_loss, val/num_specific_latents_l1, val/cl1_frac_variance_explained")


if __name__ == "__main__":
    main()
