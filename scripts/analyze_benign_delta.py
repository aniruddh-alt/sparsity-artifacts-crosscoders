#!/usr/bin/env python3
"""Compute decoder norm ratios for a BatchTopK crosscoder checkpoint."""

from __future__ import annotations

import argparse

import torch as th
from dictionary_learning.dictionary import BatchTopKCrossCoder


def decoder_norm_ratio(checkpoint_path: str) -> th.Tensor:
    crosscoder = BatchTopKCrossCoder.from_pretrained(checkpoint_path)
    norms = crosscoder.decoder.weight.norm(dim=-1)
    return norms[1] / norms[0].clamp(min=1e-8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Path to model_final.pt or checkpoint dir")
    args = parser.parse_args()

    ratio = decoder_norm_ratio(args.checkpoint)
    print("median ratio:", ratio.median().item())
    print("frac ratio > 2:", (ratio > 2).float().mean().item())
    print("frac ratio < 0.5:", (ratio < 0.5).float().mean().item())


if __name__ == "__main__":
    main()
