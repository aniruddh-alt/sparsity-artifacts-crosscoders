# Phase 2 + 3 end-to-end run: changelog and timing

## What we changed

### Library (fork: `aniruddh-alt/crosscoder_learning@delta-crosscoder-recon-loss-type`)

| File | Change |
|---|---|
| `dictionary_learning/trainers/crosscoder.py` | Add `lambda_delta` weight + `recon_loss_type` knob (`l2 / mse / mse_layer_sum`) decoupled from `lambda_delta`. `compute_delta_mse_loss` + `compute_per_layer_mse_loss` helpers with shape assertion. Persists into trainer config. |

### Application (fork: `aniruddh-alt/sparsity-artifacts-crosscoders@delta-crosscoder-lambda-experiments`)

| File | Change |
|---|---|
| `scripts/train_crosscoder.py` | New CLI: `--lambda-delta`, `--recon-loss-type`, `--lmsys-name`, `--fineweb-name`, `--warmup-steps`. Run name always emits `-delta{λ:g}` + optional `-recon{type}`. Wandb `legacy-service` call wrapped in try/except. |
| `scripts/collect_activations.py` | New `--dataset-from-disk` for locally formatted datasets (Qwen3-templated LMSYS). |
| `scripts/compute_scalers.py` | Plumb `--lmsys-name` / `--fineweb-name` through `load_activation_dataset`. |
| `scripts/collect_dictionary_activations.py` | (1) Plumb `--lmsys-name` / `--fineweb-name` / `--lmsys-col`. (2) Extract checkpoint-dir name from `.pt` path for output dir. (3) `split_into_sequences` prefers `sequence_ranges.pt` over BOS scan, with fallback to pad/eos. Handles 1-D cumulative `sequence_ranges` format. (4) Cast activations to crosscoder dtype before encoding (bf16/f32 mismatch fix). |
| `scripts/collect_activating_examples.py` | New `--no-upload` flag to skip HF Hub upload. |
| `tools/tokenization_utils.py` | Make `google/gemma-2-2b-it` tokenizer import non-fatal — Qwen3 / Llama pipelines no longer need access to that gated repo. |
| `tools/utils.py` | `PairedActivationCache` now constructed with explicit `submodule_name=` so it uses the new layout (`tokens.pt` in parent dir, activations in submodule dir). |
| `tools/configs.py` | Register `Qwen/Qwen3-0.6B-Base`, `Qwen/Qwen3-0.6B`, and the local custom-FT path. |
| `tools/training_utils.py` | New `effective_warmup_steps(max_steps, warmup_steps=None)` — clamps warmup below `max_steps` so short smoke runs don't crash on `warmup=1000` default. |
| `tools/model_pairs.py` + `configs/model_pairs.yaml` | Pair presets including `qwen3-0.6b-benign` and `qwen3-0.6b-custom`. |
| `configs/experiment_phases.yaml` | Phase 1/2/3 budgets (tokens, samples, k, λ). |
| `requirements.txt` | Bump `transformers>=4.51,<4.58` for Qwen3. |
| `run/generic/compute-activations.sh` | Per-dataset text-column switch (chat → `CC_TEXT_COLUMN`, fineweb → `text`). Opt-in `CC_CHAT_FROM_DISK=1` / `CC_FINEWEB_FROM_DISK=1`. |
| `run/generic/train-delta-crosscoder.sh` | Force `--recon-loss-type mse_layer_sum` by default so λ=0 and λ>0 share the same recon term. Wire `--lmsys-name` / `--fineweb-name` / `--warmup-steps`. Default `CC_RUN_NAME` so `set -u` doesn't kill the script. |
| `run/generic/env_from_pair.sh` + `run/generic/pipeline.sh` | Pair-name → env-var loader + end-to-end shell wrapper. |
| `run/lambda/format_lmsys_qwen3.py` | Apply Qwen3 chat template to LMSYS, save to local PVC disk. |
| `run/lambda/parallel_collect_phase2.sh` | 8-way parallel activation collection (4 splits × 2 models on 8 GPUs). |
| `run/lambda/parallel_collect_phase3.sh` | 4-way parallel custom-FT collection (reuses Phase 2's chat-model activations as Phase 3 base). |
| `run/lambda/parallel_train_phase2.sh` + `run/lambda/parallel_train_phase3.sh` | Baseline + delta concurrent training on 2 GPUs. `--workers 0` to dodge K8s `/dev/shm` bus error. |
| `run/lambda/analyze_checkpoint.sh` | One-call analyzer: scalers → latent-activation cache → quantile examples. |
| `run/lambda/end_to_end_driver.sh` | Polls for Phase 2 checkpoints + Phase 3 activations, fires Phase 2 analysis + Phase 3 training in parallel, then Phase 3 analysis + summary. |
| `run/lambda/summarize_results.py` | Reads eval logs + β-scalars + quantile-example DBs, writes `summary.md`. |
| `run/lambda/smoke_test_train.sh` | Train-only smoke runner (reuses cached activations, `--workers 0`). |
| `run/lambda/load_phase_env.py` + `run/lambda/train_phase.sh` + `run/lambda/compare_runs.py` | Phase YAML → env, phase launcher, baseline-vs-delta comparison. |
| `scripts/analyze_benign_delta.py` | Per-feature ft/base decoder-norm ratio analysis. |
| `scripts/list_model_pairs.py` + `scripts/setup_env.sh` + `scripts/format_qwen3_chat_column.py` | Pair listing, env install, Qwen3 dataset formatter (single-split). |
| `tests/test_delta_loss.py` | Math + trainer-level smoke tests for `compute_delta_mse_loss`, `select_recon_loss`, total-loss arithmetic, layer-shape assertion. |
| `tests/test_qwen3_config.py` | Verifies Qwen3 pair entries are registered. |
| `tests/test_warmup_steps.py` | Covers warmup clamp / user override paths. |
| `k8s/aniruddhan-delta-crosscoder.yaml` | Pod manifest. Scaled 1 GPU → 8 GPUs with `strategy: Recreate` (rolling update can't fit 1+8 on an 8-GPU node). |
| `k8s/bootstrap-pod.sh` + `k8s/upload-custom-model.sh` | Pod apply + repo clone, local custom-FT upload + verify. |
| `docs/superpowers/plans/2026-05-22-delta-crosscoder-lambda-experiments.md` | Original three-phase plan. |

## Ideal wall time if everything were pre-correct

Assuming the bugs above had all been fixed in advance and we just executed the plan from a clean pod:

| Stage | Wall time | Notes |
|---|---|---|
| Pod bootstrap + env install | 5 min | One-time |
| Format LMSYS with Qwen3 chat template (600K + 20K rows) | 3 min | CPU-bound, 16 procs |
| Upload custom-FT `model.zip` (1.2 GB) to PVC | 4 min | `kubectl cp` |
| Phase 2 activation collection (8 GPUs parallel) | 25 min | Chat-train is the long pole (~25 min for 50M tokens); val splits finish in <2 min |
| Phase 3 activation collection (4 GPUs, custom FT only) | 25 min | Can run in parallel with Phase 2 training → 0 added time |
| Phase 2 training (2 GPUs parallel) | ~45 min | 20K steps × 12 it/s ≈ 28 min training + ~3 min × 9 evals if we use `num_validation_samples=1M`. **The 3-hour wall time we hit was almost entirely from a 10M-sample validation budget — should have been 1M.** |
| Phase 3 training (2 GPUs parallel) | ~45 min | Same recipe, smaller val by default |
| Phase 2 analysis (2 GPUs parallel, run **during** Phase 3 training) | ~30 min | Overlaps with Phase 3 training, so 0 added time |
| Phase 3 analysis (2 GPUs parallel) | ~30 min | After Phase 3 training |
| Summary | <1 min | |
| **Total wall time** | **≈ 2.5 – 3.5 hours** | Bottlenecked by Phase 2 training → Phase 3 training → Phase 3 analysis chain |

## What actually slowed us down

| Bug | Where | Cost |
|---|---|---|
| `wandb.require("legacy-service")` crashed on newer wandb | `train_crosscoder.py:21` | One crash + relaunch (~5 min) |
| `tools/tokenization_utils.py` loads gated Gemma tokenizer at import | top-level | One crash + relaunch (~5 min) |
| `PairedActivationCache` constructed in legacy mode → `tokens.pt` looked-up in submodule dir | `tools/utils.py:90` | One crash + relaunch (~5 min) |
| Trainer hardcoded `lmsys_name="lmsys-chat-1m-chat-formatted"` | `tools/utils.py:35`, `train_crosscoder.py:145` | One crash + 2 commits |
| `warmup_steps=1000` ≥ `max_steps=1000` for smoke run | trainer | One crash + new helper |
| `CC_RUN_NAME` empty under `set -u` | launcher shell | One crash |
| `num_validation_samples=10_000_000` made Phase 2 training take **3 h** instead of ~45 min | `parallel_train_phase2.sh` | ~2 extra hours |
| `kubectl cp` truncated the custom-FT `model.safetensors` (362 MB instead of 1.2 GB) | upload | One re-upload (~4 min) |
| DataLoader workers blew out K8s `/dev/shm` (64 MB default) | training | One crash + workers=0 |
| `collect_dictionary_activations.py` used the full `.pt` path as the output dir name | step 2 | One crash |
| `collect_dictionary_activations.py` required a BOS token (Qwen3 has none) | step 2 | Two crashes (tried pad-fallback, then sequence_ranges) |
| `sequence_ranges.pt` is 1-D cumulative, not 2-D `(start, end)` rows | my own fix | One crash |
| Activations stored in bf16 but crosscoder weights in f32 → einsum dtype mismatch | step 2 | One crash + cast |

Each crash cost a relaunch (~minutes) plus the time to diagnose + push a fix (~10–20 min per round). I'd estimate the total bug-handling overhead was **~5–6 hours** on top of the ideal ~3 hours.

## What you'd need for a clean repeat

If you wanted to reproduce just the Phase 2 + 3 part on a fresh pod with no surprises:

```bash
# 1. Pod with 8 GPUs (Recreate strategy)
kubectl apply -f k8s/aniruddhan-delta-crosscoder.yaml

# 2. Inside the pod:
DICTIONARY_LEARNING_PATH=/data/aniruddhan/dictionary-learning bash scripts/setup_env.sh

# 3. Upload custom-FT (Mac side; verifies safetensors load)
bash k8s/upload-custom-model.sh

# 4. Format the dataset (one-time, ~3 min)
python run/lambda/format_lmsys_qwen3.py \
  --output /data/aniruddhan/datasets/lmsys-qwen3-phase2 \
  --n-train 600000 --n-val 20000

# 5. Drive everything (Phase 2 + Phase 3 chained)
nohup bash run/lambda/end_to_end_driver.sh > /data/aniruddhan/logs/e2e.log 2>&1 &

# 6. Final report at /data/aniruddhan/results/summary.md
```

Wall time for that fresh run on an 8-H100 pod, **with all of today's fixes baked in**: **about 3 hours.**
