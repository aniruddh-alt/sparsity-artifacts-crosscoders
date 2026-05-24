#!/usr/bin/env python3
"""Format MedQA (+ MedMCQA + PubMedQA) test/val splits with the Qwen3 chat template.

Why: we want to know what features the delta crosscoder picked up from the
MedQA-only SFT. The lmsys-validation activation cache has almost no medical
content, so medical features never surface in the quantile analysis. This
script builds a held-out medical activation source to use in place of
lmsys-qwen3-phase2 for step 2 / step 3.

Output: a `datasets.DatasetDict` saved with save_to_disk, exposing a
`validation` split whose `text_qwen3` column contains a single-turn user
message:

    <|im_start|>user
    {question}
    A) {option A}
    B) {option B}
    ...<|im_end|>

NO assistant turn — we want to see what features fire when the model is
reading a medical question, not what it generates in response.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import AutoTokenizer


def _format_options(options: Any) -> str:
    """Render a dict / list of options into A) ... B) ... lines."""
    if isinstance(options, dict):
        items = sorted(options.items())
        return "\n".join(f"{k}) {v}" for k, v in items)
    if isinstance(options, list):
        letters = ["A", "B", "C", "D", "E", "F", "G"]
        return "\n".join(f"{letters[i]}) {v}" for i, v in enumerate(options))
    return str(options)


def build_medqa_test(tok) -> Dataset:
    """USMLE-style 4-option MCQ. ~1273 held-out test examples.

    Schema: {question, options: dict[A,B,C,D, str], answer, answer_idx, ...}
    """
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")

    def render(ex):
        user = f"{ex['question']}\n{_format_options(ex['options'])}"
        msgs = [{"role": "user", "content": user}]
        return {
            "source": "medqa-test",
            "text_qwen3": tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False
            ),
        }

    return ds.map(render, desc="medqa-test", remove_columns=ds.column_names)


def build_medmcqa_val(tok, limit: int | None = None) -> Dataset:
    """Indian medical entrance MCQ. ~4183 validation examples.

    Schema: {question, opa, opb, opc, opd, cop (int), exp, ...}
    """
    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    def render(ex):
        options = {"A": ex["opa"], "B": ex["opb"], "C": ex["opc"], "D": ex["opd"]}
        user = f"{ex['question']}\n{_format_options(options)}"
        msgs = [{"role": "user", "content": user}]
        return {
            "source": "medmcqa-val",
            "text_qwen3": tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False
            ),
        }

    return ds.map(render, desc="medmcqa-val", remove_columns=ds.column_names)


def build_pubmedqa_test(tok) -> Dataset:
    """PubMedQA labeled: biomedical research Q&A. ~500 examples in pqa_labeled."""
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

    def render(ex):
        ctx = " ".join(ex["context"]["contexts"]) if isinstance(ex["context"], dict) else ""
        user = f"{ex['question']}\nContext: {ctx[:2000]}"
        msgs = [{"role": "user", "content": user}]
        return {
            "source": "pubmedqa-labeled",
            "text_qwen3": tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False
            ),
        }

    return ds.map(render, desc="pubmedqa-labeled", remove_columns=ds.column_names)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, help="save_to_disk path")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    p.add_argument("--medmcqa-limit", type=int, default=4000)
    p.add_argument("--skip-medmcqa", action="store_true")
    p.add_argument("--skip-pubmedqa", action="store_true")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    parts = [build_medqa_test(tok)]
    if not args.skip_medmcqa:
        parts.append(build_medmcqa_val(tok, limit=args.medmcqa_limit))
    if not args.skip_pubmedqa:
        parts.append(build_pubmedqa_test(tok))

    validation = concatenate_datasets(parts).shuffle(seed=42)
    print(f"validation: {len(validation)} rows")
    print(f"  by source: {validation.to_pandas()['source'].value_counts().to_dict()}")

    DatasetDict({"validation": validation}).save_to_disk(args.output)
    print(f"saved → {args.output}")


if __name__ == "__main__":
    main()
