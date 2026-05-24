#!/usr/bin/env python3
"""Causal KL-patching validation of chat-specific crosscoder latents.

Per Minder et al. 2025 (arXiv:2504.02922), the gold-standard test for whether
a crosscoder's chat-specific latents actually capture the fine-tuning effect
is to patch them into the base model and measure KL drop to the chat model:

  hybrid activation at layer L = a_base + sum_{i in S} z_i * (W_dec_chat[i] - W_dec_base[i])

where z = crosscoder.encode(stack(a_base, a_chat)) and S is the chosen
latent set. We then continue the base model's forward pass with this hybrid
activation and compute KL(chat_logits || hybrid_logits). Smaller KL = the
patched latents account for more of the FT effect.

We compare four sets:
  - top_50pct:    top 50 % of latents by |β_chat| - |β_base|
  - bottom_50pct: bottom 50 %
  - random_50pct: random 50 % (seeded)
  - top_K:        top 100 / 1000 specifically
  - none:         no patching — upper bound (base KL vs chat)
  - all:          patch every latent — lower bound (full crosscoder reconstruction)

Run separately for phase3-baseline (delta_loss=0) and phase3-delta
(delta_loss=1). Per Kassem et al. 2026, the delta crosscoder should give
much bigger KL drops on the top-set in this narrow-FT regime.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch as th
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[2]))
from dictionary_learning.dictionary import BatchTopKCrossCoder


PHASE3_BASELINE = (
    "Qwen3-0.6B-L14-k50-lr1e-04-phase3-baseline-local-shuffling-"
    "Crosscoder-delta0-reconmse_layer_sum"
)
PHASE3_DELTA = (
    "Qwen3-0.6B-L14-k50-lr1e-04-phase3-delta-local-shuffling-"
    "Crosscoder-delta1-reconmse_layer_sum"
)
LAYER = 14
BASE_MODEL = "Qwen/Qwen3-0.6B"
CHAT_MODEL = "/data/aniruddhan/models/qwen3-0.6b-custom-ft"
CKPT_ROOT = Path("/data/aniruddhan/sparsity-artifacts-crosscoders/checkpoints")
SCALERS_ROOT = Path("/data/aniruddhan/results/closed_form_scalars")
BETA_BASE_FILE = "betas_base_activation_N5000000_n_offset0.pt"
BETA_CHAT_FILE = "betas_chat_activation_N5000000_n_offset0.pt"


def load_models(device: str):
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=th.bfloat16
    ).to(device).eval()
    chat = AutoModelForCausalLM.from_pretrained(
        CHAT_MODEL, dtype=th.bfloat16
    ).to(device).eval()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return base, chat, tok


def load_crosscoder(ckpt: str, device: str) -> BatchTopKCrossCoder:
    cc = BatchTopKCrossCoder.from_pretrained(
        str(CKPT_ROOT / ckpt / "model_final.pt"),
        dtype=th.float32,
        device=device,
    )
    cc.eval()
    return cc


def load_beta_score(ckpt: str) -> th.Tensor:
    """Return |β_chat| - |β_base| per latent, the FT-specificity score used in summarize_results.py."""
    bb = th.load(SCALERS_ROOT / ckpt / "all_latents" / BETA_BASE_FILE, weights_only=True)
    bc = th.load(SCALERS_ROOT / ckpt / "all_latents" / BETA_CHAT_FILE, weights_only=True)
    return bc.abs() - bb.abs()


def build_latent_sets(scores: th.Tensor) -> dict[str, list[int]]:
    n = scores.numel()
    sorted_idx = scores.argsort(descending=True).tolist()
    half = n // 2
    rng = np.random.default_rng(42)
    rand_idx = rng.choice(n, size=half, replace=False).tolist()
    return {
        "none": [],
        "all": list(range(n)),
        "top_50pct": sorted_idx[:half],
        "bottom_50pct": sorted_idx[half:],
        "random_50pct": rand_idx,
        "top_100": sorted_idx[:100],
        "top_1000": sorted_idx[:1000],
    }


@th.no_grad()
def collect_layer_act_and_logits(model, input_ids, layer: int):
    """Forward model once, return (act_at_layer, logits) — both detached."""
    saved = {}

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        saved["act"] = h.detach().clone()

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        logits = model(input_ids).logits
    finally:
        handle.remove()
    return saved["act"], logits.detach()


@th.no_grad()
def forward_with_patched_layer(model, input_ids, layer: int, patched_act: th.Tensor):
    """Forward model, replacing layer's output with patched_act."""

    def hook(module, inp, out):
        if isinstance(out, tuple):
            return (patched_act,) + tuple(out[1:])
        return patched_act

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        logits = model(input_ids).logits
    finally:
        handle.remove()
    return logits.detach()


def kl_per_token(p_logits: th.Tensor, q_logits: th.Tensor, mask: th.Tensor) -> tuple[float, int]:
    """Return (sum_kl_over_unmasked_tokens, n_unmasked_tokens). KL(P || Q)."""
    pl = th.log_softmax(p_logits.float(), dim=-1)
    ql = th.log_softmax(q_logits.float(), dim=-1)
    kl = (pl.exp() * (pl - ql)).sum(dim=-1)  # (b, s)
    n = mask.sum().item()
    return (kl[mask].sum().item(), n)


def patched_activation(
    base_act: th.Tensor,
    chat_act: th.Tensor,
    cc: BatchTopKCrossCoder,
    latent_idx: th.Tensor,
) -> th.Tensor:
    """Compute base + sum_{i in S} z_i (W_dec_chat[i] - W_dec_base[i])."""
    if latent_idx.numel() == 0:
        return base_act
    b, s, d = base_act.shape
    stacked = th.stack([base_act, chat_act], dim=-2).float().reshape(b * s, 2, d)
    f = cc.encode(stacked)  # (b*s, F)
    f_sub = f[:, latent_idx]  # (b*s, |S|)
    # decoder.weight: (2, F, d) — index 0 = base, 1 = chat
    W_diff = cc.decoder.weight[1, latent_idx, :] - cc.decoder.weight[0, latent_idx, :]
    steering = th.einsum("nk,kd->nd", f_sub, W_diff)  # (b*s, d)
    steering = steering.reshape(b, s, d).to(base_act.dtype)
    return base_act + steering


def evaluate_set(
    base, chat, cc, latent_idx_list, input_ids_batches, pad_token_id, layer
):
    cc_device = next(cc.parameters()).device
    latent_idx = th.tensor(latent_idx_list, device=cc_device, dtype=th.long)
    total_kl, total_tokens = 0.0, 0
    for input_ids in input_ids_batches:
        base_act, _ = collect_layer_act_and_logits(base, input_ids, layer)
        chat_act, chat_logits = collect_layer_act_and_logits(chat, input_ids, layer)
        patched = patched_activation(
            base_act.to(cc_device),
            chat_act.to(cc_device),
            cc,
            latent_idx,
        ).to(base_act.device).to(base_act.dtype)
        patched_logits = forward_with_patched_layer(base, input_ids, layer, patched)
        mask = (input_ids != pad_token_id)
        # KL(chat || patched): how well does the patched model match the chat model?
        kl_sum, n = kl_per_token(chat_logits, patched_logits, mask)
        total_kl += kl_sum
        total_tokens += n
        # free
        del base_act, chat_act, chat_logits, patched_logits, patched
        th.cuda.empty_cache()
    return total_kl / max(total_tokens, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dataset", default="/data/aniruddhan/datasets/medqa-qwen3")
    p.add_argument("--split", default="validation")
    p.add_argument("--n-examples", type=int, default=64)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--output", type=Path, default=Path("/data/aniruddhan/results_medqa/kl_patching.json"))
    args = p.parse_args()

    print(f"loading models on {args.device}…")
    base, chat, tok = load_models(args.device)

    print(f"loading dataset {args.dataset}[{args.split}]…")
    ds = load_from_disk(args.dataset)[args.split].select(range(args.n_examples))
    texts = ds["text_qwen3"]
    enc = tok(
        texts, return_tensors="pt", padding=True, truncation=True,
        max_length=args.max_length,
    )
    input_ids_full = enc["input_ids"].to(args.device)
    print(f"got {input_ids_full.shape[0]} sequences × {input_ids_full.shape[1]} tokens")
    batches = [
        input_ids_full[i : i + args.batch_size]
        for i in range(0, input_ids_full.shape[0], args.batch_size)
    ]

    results = {}
    set_names: list[str] = []
    for cc_name in [PHASE3_BASELINE, PHASE3_DELTA]:
        tag = "delta" if "delta1" in cc_name else "baseline"
        print(f"\n=== crosscoder: {tag} ===")
        cc = load_crosscoder(cc_name, args.device)
        scores = load_beta_score(cc_name)
        sets = build_latent_sets(scores)
        set_names = list(sets.keys())
        results[tag] = {}
        for set_name, idx in sets.items():
            print(f"  evaluating {set_name} ({len(idx)} latents)…", flush=True)
            kl = evaluate_set(
                base, chat, cc, idx, batches, tok.pad_token_id, LAYER
            )
            print(f"    KL = {kl:.4f}")
            results[tag][set_name] = {"kl": kl, "n_latents": len(idx)}
        del cc
        th.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.output}")

    # Pretty print summary
    print("\n=== summary ===")
    print(f"{'set':<14} {'baseline':>10} {'delta':>10}")
    for set_name in set_names:
        b = results['baseline'][set_name]['kl']
        d = results['delta'][set_name]['kl']
        print(f"{set_name:<14} {b:>10.4f} {d:>10.4f}")


if __name__ == "__main__":
    main()
