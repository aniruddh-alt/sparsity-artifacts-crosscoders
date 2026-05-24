#!/usr/bin/env python3
"""Token-level firing visualization for crosscoder latents.

For each target latent, pull its top-K activating examples from examples.db,
JOIN with activation_details (per-token positions + values), decode token
ids individually, and render markdown where high-activation tokens are
bolded inline. This disambiguates:

  - "Latent fires on medical content"      → peaks on terms like
                                              `appendectomy`, `pheochromocytoma`.
  - "Latent fires on chat-template tokens" → peaks on `<|im_start|>`,
                                              `<|im_end|>`, `\nuser\n`.
  - "Latent fires on MCQ structure"        → peaks on `A)`, `B)`, `C)`, etc.
  - "Latent fires on a syntactic feature"  → peaks on punctuation, articles.

The activation_details BLOB layout (from collect_activating_examples.py):
  th.stack([token_indices.int(), values.float().view(th.int32)], dim=1)
so positions are int32 bytes and values are float32 bytes (typed as int32
in torch but the underlying bytes are float32).
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


PHASE3_DELTA = (
    "Qwen3-0.6B-L14-k50-lr1e-04-phase3-delta-local-shuffling-"
    "Crosscoder-delta1-reconmse_layer_sum"
)


def get_top_examples(cur, feature_idx: int, n: int):
    cur.execute(
        """SELECT qe.activation, qe.sequence_idx
           FROM quantile_examples qe
           WHERE qe.feature_idx = ?
           ORDER BY qe.activation DESC LIMIT ?""",
        (int(feature_idx), n),
    )
    return cur.fetchall()


def get_sequence_tokens(cur, sequence_idx: int) -> list[int]:
    cur.execute(
        "SELECT token_ids FROM sequences WHERE sequence_idx = ?", (int(sequence_idx),)
    )
    row = cur.fetchone()
    if row is None:
        return []
    return np.frombuffer(row[0], dtype=np.int32).tolist()


def get_activation_details(cur, feature_idx: int, sequence_idx: int):
    cur.execute(
        "SELECT positions, activation_values FROM activation_details "
        "WHERE feature_idx = ? AND sequence_idx = ?",
        (int(feature_idx), int(sequence_idx)),
    )
    row = cur.fetchone()
    if row is None:
        return None, None
    pos_blob, val_blob = row
    positions = np.frombuffer(pos_blob, dtype=np.int32)
    values = np.frombuffer(val_blob, dtype=np.float32)
    return positions, values


def render_with_highlights(
    tokens: list[int],
    positions,
    values,
    tok,
    threshold_frac: float = 0.25,
    max_chars: int = 1200,
) -> str:
    """Render tokens; bold the ones whose activation is ≥ threshold_frac × max."""
    if positions is None or len(positions) == 0:
        return tok.decode(tokens, skip_special_tokens=False)[:max_chars]

    pos_to_val = dict(zip(positions.tolist(), values.tolist()))
    max_val = max(pos_to_val.values()) if pos_to_val else 0.0
    threshold = max(max_val * threshold_frac, 1e-6)

    parts = []
    for i, tok_id in enumerate(tokens):
        s = tok.decode([tok_id], skip_special_tokens=False)
        # markdown-escape special-token brackets so they render literally
        s_display = s.replace("\\", "\\\\").replace("`", "\\`")
        # collapse newline + carriage return into visible markers
        s_display = s_display.replace("\n", "\\n").replace("\r", "\\r")
        val = pos_to_val.get(i, 0.0)
        if val >= threshold:
            parts.append(f"`{s_display}`**[{val:.2f}]**")
        else:
            parts.append(s_display)
    rendered = " ".join(p for p in parts if p)
    if len(rendered) > max_chars:
        rendered = rendered[:max_chars] + "…"
    return rendered


def top_tokens(tokens: list[int], positions, values, tok, k: int = 8):
    if positions is None or len(positions) == 0:
        return []
    paired = sorted(
        zip(positions.tolist(), values.tolist()), key=lambda x: x[1], reverse=True
    )[:k]
    out = []
    for p, v in paired:
        if 0 <= p < len(tokens):
            s = tok.decode([tokens[p]], skip_special_tokens=False)
            s = s.replace("\n", "\\n").replace("`", "\\`")
            out.append(f"  - pos {p}: `{s}` → {v:.3f}")
    return out


def render_latent(cur, feat: int, tok, n_examples: int) -> str:
    examples = get_top_examples(cur, feat, n_examples)
    if not examples:
        return f"\n## Latent {feat} — no examples in db\n"
    lines = [f"\n## Latent {feat}\n"]
    for act, seq_idx in examples:
        tokens = get_sequence_tokens(cur, seq_idx)
        positions, values = get_activation_details(cur, feat, seq_idx)
        lines.append(f"\n### seq={seq_idx}  max_act={act:.3f}\n")
        lines.append("**Top-firing tokens** (position : token → activation):")
        lines.extend(top_tokens(tokens, positions, values, tok))
        rendered = render_with_highlights(tokens, positions, values, tok)
        lines.append("\n**Full text** (bolded = activation ≥ 25 % of max in this example):")
        lines.append(f"\n> {rendered}\n")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--db",
        type=Path,
        default=Path(
            "/data/aniruddhan/results_medqa/quantile_examples/"
            f"{PHASE3_DELTA}/examples.db"
        ),
    )
    p.add_argument(
        "--latents",
        type=int,
        nargs="+",
        default=[
            # Medical-only candidates (strong selectivity)
            23929, 27235, 18576, 2582, 10689,
            # High med/gen ratio
            6669, 26172, 16454,
            # Suspicious "always-on" tier (per med_max ranking)
            22296,
            # Comparison: the lmsys-top "jokes" latent
            14297,
        ],
    )
    p.add_argument("--n-examples", type=int, default=3)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("/data/aniruddhan/results_medqa/token_firing.md"),
    )
    p.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    con = sqlite3.connect(str(args.db))
    cur = con.cursor()

    parts = [
        f"# Token-level firing — {args.db.parent.name}\n",
        "For each latent, top activating examples shown with per-token "
        "activation values. Bolded tokens have activation ≥ 25 % of the max "
        "in that example. The **Top-firing tokens** list gives the exact peak "
        "tokens — if a latent peaks on `<|im_start|>` or `A)`, it's a "
        "template/structure feature regardless of the surrounding text.\n\n---\n",
    ]
    for feat in args.latents:
        parts.append(render_latent(cur, feat, tok, args.n_examples))

    con.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(parts))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
