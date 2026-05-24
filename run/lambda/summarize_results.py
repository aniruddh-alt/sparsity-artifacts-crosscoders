#!/usr/bin/env python3
"""Summarize Phase 2 / Phase 3 results into a markdown report.

For each (phase, lambda) checkpoint we report:
  - Final eval metrics from last_eval_logs.pt
  - Top-N ft-specific latents by closed-form beta ratio
  - For each top latent, the highest-quantile activating examples

We do NOT interpret the features — that's the human's job. We just surface
the evidence in a format suitable for reading.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import torch as th


CKPT_ROOT = Path("/data/aniruddhan/sparsity-artifacts-crosscoders/checkpoints")
RESULTS = Path("/data/aniruddhan/results")
LATENT_DIR = Path("/data/aniruddhan/latent_activations")


PHASE_CHECKPOINTS = {
    "phase2": {
        "baseline": "Qwen3-0.6B-Base-L14-k50-lr1e-04-phase2-baseline-local-shuffling-Crosscoder-delta0-reconmse_layer_sum",
        "delta":    "Qwen3-0.6B-Base-L14-k50-lr1e-04-phase2-delta-local-shuffling-Crosscoder-delta1-reconmse_layer_sum",
        "tokenizer": "Qwen/Qwen3-0.6B",
    },
    "phase3": {
        "baseline": "Qwen3-0.6B-L14-k50-lr1e-04-phase3-baseline-local-shuffling-Crosscoder-delta0-reconmse_layer_sum",
        "delta":    "Qwen3-0.6B-L14-k50-lr1e-04-phase3-delta-local-shuffling-Crosscoder-delta1-reconmse_layer_sum",
        "tokenizer": "Qwen/Qwen3-0.6B",
    },
}


def safe_load(path: Path, **kwargs):
    try:
        return th.load(path, weights_only=False, map_location="cpu", **kwargs)
    except Exception as e:
        return {"_error": f"load failed for {path}: {e!r}"}


def read_eval_metrics(ckpt_dir: Path) -> dict[str, Any]:
    f = ckpt_dir / "last_eval_logs.pt"
    if not f.exists():
        return {"_error": f"{f} missing"}
    logs = safe_load(f)
    if not isinstance(logs, dict):
        return {"_error": f"unexpected type: {type(logs).__name__}"}
    out = {}
    for k, v in logs.items():
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out


def read_betas(ckpt_name: str) -> dict[str, Any]:
    """Find any betas_*.pt under closed_form_scalars/<ckpt>/all_latents/."""
    d = RESULTS / "closed_form_scalars" / ckpt_name / "all_latents"
    if not d.exists():
        return {"_error": f"scalars dir missing: {d}"}
    out: dict[str, Any] = {}
    for f in sorted(d.glob("betas_*.pt")):
        v = safe_load(f)
        if isinstance(v, th.Tensor):
            out[f.stem] = v
    return out


def ft_specificity(betas: dict[str, th.Tensor]) -> tuple[th.Tensor, str] | None:
    """Return per-latent ft-specificity score and what we used to compute it."""
    base_keys = [k for k in betas if k.startswith("betas_base_activation_") and "no_bias" not in k]
    chat_keys = [k for k in betas if k.startswith("betas_chat_activation_") and "no_bias" not in k]
    if not (base_keys and chat_keys):
        return None
    bb = betas[base_keys[0]].float()
    bc = betas[chat_keys[0]].float()
    if bb.shape != bc.shape:
        return None
    # Score: |β_chat| - |β_base|. Big positive = ft-specific.
    # NaN values mean the latent never activated during scaler estimation;
    # treat them as zero so a NaN doesn't outrank a finite score.
    bb = th.nan_to_num(bb, nan=0.0, posinf=0.0, neginf=0.0)
    bc = th.nan_to_num(bc, nan=0.0, posinf=0.0, neginf=0.0)
    score = bc.abs() - bb.abs()
    return score, f"|{chat_keys[0]}| - |{base_keys[0]}|"


_TOKENIZER_CACHE: dict[str, Any] = {}


def _get_tokenizer(model_name: str):
    if model_name in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_name]
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        tok = {"_error": f"tokenizer load failed: {e!r}"}
    _TOKENIZER_CACHE[model_name] = tok
    return tok


def _decode_token_blob(blob: bytes, tok, max_chars: int = 400) -> str:
    if isinstance(tok, dict):
        return tok.get("_error", "<no tokenizer>")
    ids = np.frombuffer(blob, dtype=np.int32).tolist()
    text = tok.decode(ids, skip_special_tokens=False)
    text = text.replace("\n", "\\n")
    if len(text) > max_chars:
        text = text[:max_chars] + "…"
    return text


def read_quantile_examples(
    ckpt_name: str, feature_idx: int, n: int = 5, tokenizer_name: str | None = None
) -> list[dict[str, Any]]:
    """Pull top-n activating examples for a given feature from examples.db.

    Schema (written by scripts/collect_activating_examples.py):
      sequences(sequence_idx, token_ids BLOB int32)
      quantile_examples(feature_idx, quantile_idx, activation, sequence_idx)
      activation_details(feature_idx, sequence_idx, positions BLOB, activation_values BLOB)
    """
    db = RESULTS / "quantile_examples" / ckpt_name / "examples.db"
    if not db.exists():
        return [{"_error": f"db missing: {db}"}]
    tok = _get_tokenizer(tokenizer_name) if tokenizer_name else None
    conn = sqlite3.connect(str(db))
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT qe.feature_idx, qe.quantile_idx, qe.activation,
                   qe.sequence_idx, s.token_ids
            FROM quantile_examples qe
            JOIN sequences s ON qe.sequence_idx = s.sequence_idx
            WHERE qe.feature_idx = ?
            ORDER BY qe.activation DESC
            LIMIT ?
            """,
            (int(feature_idx), n),
        )
        rows = cur.fetchall()
        if not rows:
            return [{"_warn": f"no rows for feature_idx={feature_idx}"}]
        out = []
        for feat, q_idx, act, seq_idx, token_blob in rows:
            entry = {
                "feature_idx": feat,
                "quantile_idx": q_idx,
                "activation": act,
                "sequence_idx": seq_idx,
            }
            if tok is not None:
                entry["text"] = _decode_token_blob(token_blob, tok)
            out.append(entry)
        return out
    finally:
        conn.close()


def summarize_phase(phase: str, out_path: Path) -> str:
    sections = [f"# {phase.upper()}\n"]
    ck = PHASE_CHECKPOINTS[phase]

    # Eval metrics
    sections.append("## Eval metrics (last_eval_logs.pt)\n")
    sections.append("| metric | baseline | delta | Δ |")
    sections.append("|---|---|---|---|")
    base_metrics = read_eval_metrics(CKPT_ROOT / ck["baseline"])
    delta_metrics = read_eval_metrics(CKPT_ROOT / ck["delta"])
    keys = sorted(set(base_metrics) | set(delta_metrics))
    for k in keys:
        b = base_metrics.get(k, float("nan"))
        d = delta_metrics.get(k, float("nan"))
        try:
            diff = f"{d - b:+.4f}"
        except TypeError:
            diff = ""
        sections.append(f"| {k} | {b!r} | {d!r} | {diff} |")

    # Top ft-specific latents — only meaningful for the delta run
    sections.append("\n## Top ft-specific latents (delta crosscoder)\n")
    delta_betas = read_betas(ck["delta"])
    if "_error" in delta_betas:
        sections.append(f"(scalars not available: {delta_betas['_error']})\n")
        out_path.write_text("\n".join(sections))
        return "\n".join(sections)

    spec = ft_specificity(delta_betas)
    if spec is None:
        sections.append("(could not compute ft-specificity score from available betas; "
                       f"available files: {sorted(delta_betas)})\n")
        out_path.write_text("\n".join(sections))
        return "\n".join(sections)

    score, formula = spec
    top_n = 10
    top_idxs = th.topk(score, k=top_n).indices.tolist()
    sections.append(f"Ranking by `{formula}`. Top {top_n}:\n")
    sections.append("| rank | latent | score |")
    sections.append("|---|---|---|")
    for r, idx in enumerate(top_idxs, 1):
        sections.append(f"| {r} | {idx} | {score[idx].item():.4f} |")

    # Quantile examples for those latents
    sections.append("\n## Activating examples for top ft-specific latents\n")
    tokenizer_name = ck.get("tokenizer")
    for idx in top_idxs[:5]:
        rows = read_quantile_examples(ck["delta"], idx, n=3, tokenizer_name=tokenizer_name)
        sections.append(f"### latent {idx}\n")
        if rows and "_error" in rows[0]:
            sections.append(f"_{rows[0]['_error']}_\n")
        elif rows and "_warn" in rows[0]:
            sections.append(f"_{rows[0]['_warn']}_\n")
        else:
            for row in rows:
                act = row.get("activation", 0.0)
                q = row.get("quantile_idx", "?")
                text = row.get("text") or json.dumps(
                    {k: v for k, v in row.items() if k not in {"activation", "quantile_idx", "feature_idx"}},
                    default=str,
                )[:400]
                sections.append(f"- act={act:.3f} q={q} | `{text}`")
            sections.append("")

    text = "\n".join(sections)
    out_path.write_text(text)
    return text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["phase2", "phase3", "both"], default="both")
    p.add_argument("--output", type=Path, default=RESULTS / "summary.md")
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    parts = ["# Delta-crosscoder end-to-end results\n"]
    if args.phase in ("phase2", "both"):
        parts.append(summarize_phase("phase2", args.output.with_name("phase2_summary.md")))
    if args.phase in ("phase3", "both"):
        parts.append(summarize_phase("phase3", args.output.with_name("phase3_summary.md")))

    args.output.write_text("\n\n---\n\n".join(parts))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
