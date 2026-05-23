#!/usr/bin/env python3
"""List or inspect crosscoder model-pair presets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.model_pairs import export_pair_env, get_model_pair, load_model_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pair", nargs="?", help="Model pair name to inspect")
    parser.add_argument(
        "--shell",
        action="store_true",
        help="Print export statements for bash (used by env_from_pair.sh)",
    )
    args = parser.parse_args()

    if args.pair is None:
        for name in sorted(load_model_pairs()):
            print(name)
        return

    if args.shell:
        for key, value in export_pair_env(args.pair).items():
            escaped = value.replace("'", "'\"'\"'")
            print(f"export {key}='{escaped}'")
        return

    print(json.dumps(get_model_pair(args.pair), indent=2))


if __name__ == "__main__":
    main()
