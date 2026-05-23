#!/usr/bin/env python3
"""Format the LMSYS chat dataset with the Qwen3 chat template, save to local disk.

Phase 2 needs ~50M Qwen3 tokens of chat data. With avg ~400 tokens/row this is
~125K rows; we format 600K to give the activation collector a comfortable buffer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer


def add_text_qwen3(batch, tokenizer):
    texts = []
    for convo in batch["conversation"]:
        msgs = [{"role": m["role"], "content": m["content"]} for m in convo]
        texts.append(
            tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        )
    return {"text_qwen3": texts}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, help="Local save_to_disk path.")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    p.add_argument("--n-train", type=int, default=600_000)
    p.add_argument("--n-val", type=int, default=20_000)
    p.add_argument("--num-proc", type=int, default=16)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    splits = {"train": args.n_train, "validation": args.n_val}
    out = {}
    for split, n in splits.items():
        ds = load_dataset(
            "science-of-finetuning/lmsys-chat-1m-chat-formatted", split=split
        )
        ds = ds.select(range(min(n, len(ds))))
        ds = ds.map(
            lambda batch: add_text_qwen3(batch, tok),
            batched=True,
            batch_size=500,
            num_proc=args.num_proc,
            desc=f"qwen3-tpl-{split}",
        )
        drop = [c for c in ds.column_names if c not in {"conversation_id", "text_qwen3"}]
        ds = ds.remove_columns(drop)
        out[split] = ds
        print(f"{split}: {len(ds)} rows")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    DatasetDict(out).save_to_disk(args.output)
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
