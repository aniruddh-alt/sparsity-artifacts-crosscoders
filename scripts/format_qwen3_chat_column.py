#!/usr/bin/env python3
"""Add text_qwen3 column to a HF dataset split using the Qwen3 chat template."""

from __future__ import annotations

import argparse

from datasets import load_dataset
from transformers import AutoTokenizer


def format_example(example, tokenizer, messages_col: str = "messages"):
    text = tokenizer.apply_chat_template(
        example[messages_col],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return {"text_qwen3": text}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--messages-col", default="messages")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    dataset = load_dataset(args.dataset, split=args.split)
    dataset = dataset.map(
        lambda example: format_example(example, tokenizer, args.messages_col),
        desc="format_qwen3",
    )
    dataset.save_to_disk(args.output)
    print(f"Saved {len(dataset)} rows to {args.output}")


if __name__ == "__main__":
    main()
