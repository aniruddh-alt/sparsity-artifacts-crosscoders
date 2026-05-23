#!/usr/bin/env python3
"""Print shell export statements for a named experiment phase."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", help="Phase name from configs/experiment_phases.yaml")
    args = parser.parse_args()

    phases = yaml.safe_load((REPO_ROOT / "configs" / "experiment_phases.yaml").read_text())
    pairs = yaml.safe_load((REPO_ROOT / "configs" / "model_pairs.yaml").read_text())

    if args.phase not in phases:
        known = ", ".join(sorted(phases))
        print(f"Unknown phase '{args.phase}'. Known: {known}", file=sys.stderr)
        raise SystemExit(1)

    phase = phases[args.phase]
    pair_name = phase["pair"]
    if pair_name not in pairs:
        print(f"Unknown pair '{pair_name}'", file=sys.stderr)
        raise SystemExit(1)

    pair = pairs[pair_name]
    exports = {
        "CC_PAIR": pair_name,
        "CC_PHASE": args.phase,
        **{f"CC_{key.upper()}": str(value) for key, value in pair.items()},
        **{
            f"CC_{key.upper()}": str(value)
            for key, value in phase.items()
            if key != "pair"
        },
    }

    for key, value in exports.items():
        escaped = value.replace("'", "'\"'\"'")
        print(f"export {key}='{escaped}'")


if __name__ == "__main__":
    main()
