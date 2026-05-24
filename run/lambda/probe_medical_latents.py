#!/usr/bin/env python3
"""Discover which delta-crosscoder latents fire strongest on MEDICAL content.

Why: the β-score ranking surfaces latents the delta crosscoder uses more in
the chat (custom-FT) model than the base. That can be "reasoning structure"
features the FT amplified, not necessarily medical content per se. To see
medical features specifically, we look at the medqa examples.db and ask:
which latents have their TOP activating examples coming from medical text?

We classify sequences as "medical" by source — the medqa-qwen3 dataset is a
mix of medqa-test + medmcqa-val + pubmedqa-labeled; the other half of the
cache is fineweb pretraining text. The cache stores sequences in
[fineweb..., lmsys-slot...] order, so the boundary index between them
identifies which half is medical.

Outputs a markdown report comparing:
  - Top-N latents by max-activation-on-medical
  - Top-N latents by (medical_max / general_max) ratio — "selectively medical"
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import torch as th
from collections import defaultdict
from transformers import AutoTokenizer


def load_med_seq_boundary() -> int:
    """Sequences with idx >= boundary are from the medqa half of the cache."""
    fineweb_seq_ranges = th.load(
        "/data/aniruddhan/activations/Qwen3-0.6B/fineweb-1m-sample/validation/sequence_ranges.pt",
        weights_only=True,
    )
    return len(fineweb_seq_ranges) - 1


def probe(db_path: Path, top_n: int, tok, label: str) -> str:
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()

    boundary = load_med_seq_boundary()
    cur.execute("SELECT COUNT(*) FROM sequences")
    total_seqs = cur.fetchone()[0]
    print(f"[{label}] total sequences: {total_seqs}, medical boundary (≥): {boundary}")

    cur.execute(
        "SELECT feature_idx, MAX(activation) FROM quantile_examples WHERE sequence_idx >= ? GROUP BY feature_idx",
        (boundary,),
    )
    med_max = dict(cur.fetchall())

    cur.execute(
        "SELECT feature_idx, MAX(activation) FROM quantile_examples WHERE sequence_idx < ? GROUP BY feature_idx",
        (boundary,),
    )
    gen_max = dict(cur.fetchall())

    # Count how many top-quantile activating examples per latent fall on
    # medical sequences, as a "fraction medical" signal.
    cur.execute(
        "SELECT feature_idx, sequence_idx, activation FROM quantile_examples"
    )
    counts: dict[int, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for f, sid, _ in cur.fetchall():
        n_med, n_gen = counts[f]
        if sid >= boundary:
            counts[f] = (n_med + 1, n_gen)
        else:
            counts[f] = (n_med, n_gen + 1)

    # Ranking 1: pure max-on-medical
    by_med_max = sorted(med_max.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

    # Ranking 2: medical/general ratio (only for latents that fire on both;
    # latents with no general activations get a separate "exclusively medical"
    # category to avoid divide-by-zero exploits).
    ratio: list[tuple[int, float, float, float]] = []
    exclusive: list[tuple[int, float]] = []
    for f, m in med_max.items():
        g = gen_max.get(f, 0.0)
        if g <= 1e-6:
            exclusive.append((f, m))
        else:
            ratio.append((f, m / g, m, g))
    by_ratio = sorted(ratio, key=lambda x: x[1], reverse=True)[:top_n]
    by_exclusive = sorted(exclusive, key=lambda x: x[1], reverse=True)[:top_n]

    def render_examples(feat: int, k: int = 3, only_medical: bool = True) -> list[str]:
        clause = "AND qe.sequence_idx >= ?" if only_medical else ""
        params = [feat, boundary] if only_medical else [feat]
        cur.execute(
            f"""SELECT qe.activation, qe.sequence_idx, s.token_ids
                FROM quantile_examples qe JOIN sequences s ON qe.sequence_idx=s.sequence_idx
                WHERE qe.feature_idx = ? {clause}
                ORDER BY qe.activation DESC LIMIT ?""",
            params + [k],
        )
        out = []
        for act, sid, blob in cur.fetchall():
            text = tok.decode(np.frombuffer(blob, dtype=np.int32).tolist(), skip_special_tokens=False)
            text = text.replace("\n", "\\n")
            if len(text) > 280:
                text = text[:280] + "…"
            tag = "MED" if sid >= boundary else "GEN"
            out.append(f"  - {tag} act={act:.3f} seq={sid} | `{text}`")
        return out

    lines = [f"# Medical-feature probe — {label}\n"]
    lines.append(
        f"Total sequences in db: **{total_seqs}** (medical = sequence_idx ≥ {boundary}; "
        f"medical count = {total_seqs - boundary}, general = {boundary})\n"
    )

    lines.append(f"\n## Top-{top_n} latents by **max activation on medical sequences**\n")
    lines.append("| rank | latent | med_max | gen_max | n_top_med | n_top_gen |")
    lines.append("|---|---|---|---|---|---|")
    for r, (f, m) in enumerate(by_med_max, 1):
        g = gen_max.get(f, 0.0)
        nm, ng = counts.get(f, (0, 0))
        lines.append(f"| {r} | {f} | {m:.3f} | {g:.3f} | {nm} | {ng} |")
    for r, (f, _) in enumerate(by_med_max[:10], 1):
        lines.append(f"\n### latent {f} — top medical activations")
        lines += render_examples(f, k=3, only_medical=True)

    lines.append(f"\n## Top-{top_n} latents by **med_max / gen_max** (selective for medical content)\n")
    lines.append("| rank | latent | ratio | med_max | gen_max |")
    lines.append("|---|---|---|---|---|")
    for r, (f, rat, m, g) in enumerate(by_ratio, 1):
        lines.append(f"| {r} | {f} | {rat:.2f} | {m:.3f} | {g:.3f} |")
    for r, (f, _, _, _) in enumerate(by_ratio[:10], 1):
        lines.append(f"\n### latent {f} — top medical activations (ratio rank {r})")
        lines += render_examples(f, k=3, only_medical=True)

    lines.append(f"\n## Top-{top_n} latents that fire **only on medical** (no general activations)\n")
    lines.append("| rank | latent | med_max |")
    lines.append("|---|---|---|")
    for r, (f, m) in enumerate(by_exclusive, 1):
        lines.append(f"| {r} | {f} | {m:.3f} |")
    for r, (f, _) in enumerate(by_exclusive[:10], 1):
        lines.append(f"\n### latent {f} — medical-only activations (rank {r})")
        lines += render_examples(f, k=3, only_medical=True)

    con.close()
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("/data/aniruddhan/results_medqa"),
    )
    p.add_argument(
        "--latent-dir",
        type=Path,
        default=Path("/data/aniruddhan/latent_activations_medqa"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("/data/aniruddhan/results_medqa/medical_features.md"),
    )
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-0.6B")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    parts = ["# Delta-crosscoder medical-features probe (phase 3)\n"]
    for ckpt_short, ckpt in [
        ("phase3-baseline", "Qwen3-0.6B-L14-k50-lr1e-04-phase3-baseline-local-shuffling-Crosscoder-delta0-reconmse_layer_sum"),
        ("phase3-delta", "Qwen3-0.6B-L14-k50-lr1e-04-phase3-delta-local-shuffling-Crosscoder-delta1-reconmse_layer_sum"),
    ]:
        db = args.results_dir / "quantile_examples" / ckpt / "examples.db"
        if not db.exists():
            parts.append(f"\n---\n\n# {ckpt_short}\n\n_db missing: {db}_\n")
            continue
        parts.append("\n---\n\n")
        parts.append(probe(db, args.top_n, tok, label=ckpt_short))

    args.output.write_text("\n".join(parts))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
