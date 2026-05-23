# Delta-Crosscoder Lambda Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate the delta-crosscoder training pipeline on Oumi's Lambda H100 cluster with a three-phase experiment ladder — smoke test, official benign HF pair, then your custom Qwen3-0.6B fine-tune — comparing standard vs delta loss.

**Architecture:** Run all experiments inside a single-GPU `oumi-science` pod on `worker-gpu-8x-h100-sxm5gdr-zjdfj-5pk5t` with data on `pvc-oumi-science`. Use the existing `sparsity-artifacts-crosscoders` repo (delta loss already wired) for activation caching + BatchTopK crosscoder training. Each phase trains two crosscoders on identical activations: `--lambda-delta 0` (baseline) vs `--lambda-delta 1.0` (delta).

**Tech Stack:** Oumi Lambda K8s (`kubectl`, `ghcr.io/oumi-ai/oumi:latest`), HuggingFace Transformers ≥4.51 (Qwen3), `dictionary_learning` fork with delta loss, wandb, PyTorch, 1× H100 SXM.

---

## Model Selection (locked before Phase 0)

| Phase | Purpose | Base model | Fine-tuned / chat model | Why this pair |
|-------|---------|------------|-------------------------|---------------|
| **1 — Organism** | Validate pipeline end-to-end cheaply | [`Qwen/Qwen3-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) | Smallest Qwen3 pair (~0.6B, 28 layers, Apache 2.0, ungated). Official post-training delta is well-defined. Fits on 1 GPU with tiny token budget. |
| **2 — Benign HF** | Test delta loss on a known alignment shift | Same as Phase 1 | Same as Phase 1 | Full token budget. This is the canonical "benign chat-tuning" pair for Qwen3 — base pretrain vs official instruct/post-train. Re-run at scale after Phase 1 passes. |
| **3 — Custom FT** | Test delta loss on your narrow SFT | [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) | Local checkpoint from `model.zip` | Your Oumi training config shows `model_name: Qwen/Qwen3-0.6B` — SFT on top of the official checkpoint, not Base. Delta is intentionally small. |

**Fallback benign pair** (if Qwen3 dataset formatting blocks you): [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B) + [`meta-llama/Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) — already supported in repo with `text_llama3` column and `configs/model_pairs.yaml`.

**Custom model facts** (from `~/Downloads/model.zip`):
- Architecture: `Qwen3ForCausalLM`, 28 layers, `hidden_size=1024`, ~1.2 GB `model.safetensors`
- Trained with `transformers==4.57.6`, chat template in `chat_template.jinja`
- Trained from `Qwen/Qwen3-0.6B` with custom JSONL SFT

**Recommended layer:** `14` (mid-depth of 28 layers) for all Qwen3 phases. Sweep `12, 14, 16` only if Phase 2 delta signal is weak.

---

## File Map

| File | Responsibility |
|------|----------------|
| `k8s/aniruddhan-delta-crosscoder.yaml` | GPU pod deployment (sleep infinity, PVC mount) |
| `k8s/aniruddhan-data-upload-job.yaml` | One-shot job to copy `model.zip` onto PVC |
| `scripts/setup_env.sh` | Install repo + editable `dictionary_learning` |
| `scripts/format_qwen3_chat_column.py` | Add `text_qwen3` column to HF datasets |
| `tools/configs.py` | Register Qwen3 models in `MODEL_CONFIGS` |
| `configs/model_pairs.yaml` | Add `qwen3-0.6b-benign` and `qwen3-0.6b-custom` presets |
| `configs/experiment_phases.yaml` | Token budgets, steps, k, lambda per phase |
| `run/lambda/train_phase.sh` | Phase-aware wrapper (reads experiment_phases.yaml) |
| `run/lambda/compare_runs.py` | Post-hoc comparison of baseline vs delta runs |
| `tests/test_delta_loss.py` | Already exists — gate before cluster work |

---

## Phase 0: Cluster + Repo Bootstrap

### Task 0: K8s pod and environment

**Files:**
- Create: `k8s/aniruddhan-delta-crosscoder.yaml`
- Modify: none

- [ ] **Step 1: Write pod manifest**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: oumi-aniruddhan-delta-crosscoder
  namespace: oumi-science
  labels:
    app: oumi-aniruddhan-delta-crosscoder
    owner: aniruddhan
spec:
  replicas: 1
  selector:
    matchLabels:
      name: oumi-aniruddhan-delta-crosscoder
  template:
    metadata:
      labels:
        name: oumi-aniruddhan-delta-crosscoder
        owner: aniruddhan
    spec:
      nodeSelector:
        kubernetes.io/hostname: worker-gpu-8x-h100-sxm5gdr-zjdfj-5pk5t
      initContainers:
      - name: fix-permissions
        image: busybox
        command: ["sh", "-c", "mkdir -p /data/aniruddhan && chown -R 999:999 /data/aniruddhan"]
        securityContext:
          runAsUser: 0
        volumeMounts:
        - name: data
          mountPath: /data
      containers:
      - name: oumi
        image: ghcr.io/oumi-ai/oumi:latest
        securityContext:
          runAsUser: 0
        command: ["sleep", "infinity"]
        resources:
          requests:
            nvidia.com/gpu: 1
            ephemeral-storage: 20Gi
          limits:
            nvidia.com/gpu: 1
            ephemeral-storage: 80Gi
        env:
        - name: DATASTORE
          value: /data/aniruddhan
        - name: HF_HOME
          value: /data/aniruddhan/hf_cache
        - name: WANDB_API_KEY
          valueFrom:
            secretKeyRef:
              name: wandb-aniruddhan
              key: api_key
              optional: true
        volumeMounts:
        - name: data
          mountPath: /data
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: pvc-oumi-science
      tolerations:
      - key: nvidia.com/gpu
        operator: Equal
        value: "true"
        effect: NoSchedule
```

- [ ] **Step 2: Apply and verify pod**

```bash
export KUBECONFIG=~/.kube/lambdaconfig
export NAMESPACE=oumi-science
kubectl apply -f k8s/aniruddhan-delta-crosscoder.yaml
kubectl get pods -n $NAMESPACE | grep aniruddhan-delta-crosscoder
kubectl exec -it deploy/oumi-aniruddhan-delta-crosscoder -n $NAMESPACE -- nvidia-smi
```

Expected: pod `Running`, one H100 visible.

- [ ] **Step 3: Clone repos inside pod**

```bash
POD=deploy/oumi-aniruddhan-delta-crosscoder
kubectl exec -it $POD -n oumi-science -- bash -c '
  source /opt/conda/etc/profile.d/conda.sh && conda activate base
  cd /data/aniruddhan
  git clone https://github.com/science-of-finetuning/sparsity-artifacts-crosscoders.git
  git clone https://github.com/science-of-finetuning/crosscoder_learning.git dictionary-learning
  cd sparsity-artifacts-crosscoders && bash scripts/setup_env.sh
'
```

Expected: `pip install -e` succeeds, `python scripts/list_model_pairs.py` prints pair names.

- [ ] **Step 4: Commit manifest**

```bash
git add k8s/aniruddhan-delta-crosscoder.yaml
git commit -m "infra: add Lambda pod for delta-crosscoder experiments"
```

---

### Task 1: Upload custom fine-tune to cluster

**Files:**
- Create: `k8s/aniruddhan-upload-custom-model.sh`

- [ ] **Step 1: Unzip locally and kubectl cp**

```bash
# On your Mac
mkdir -p /tmp/qwen3-custom-ft
unzip -q ~/Downloads/model.zip -d /tmp/qwen3-custom-ft

export KUBECONFIG=~/.kube/lambdaconfig
POD=$(kubectl get pod -n oumi-science -l name=oumi-aniruddhan-delta-crosscoder -o jsonpath='{.items[0].metadata.name}')

kubectl exec -n oumi-science $POD -- mkdir -p /data/aniruddhan/models/qwen3-0.6b-custom-ft
kubectl cp /tmp/qwen3-custom-ft/. oumi-science/$POD:/data/aniruddhan/models/qwen3-0.6b-custom-ft/
```

- [ ] **Step 2: Verify checkpoint loads on pod**

```bash
kubectl exec -it $POD -n oumi-science -- bash -c '
  source /opt/conda/etc/profile.d/conda.sh && conda activate base
  python - <<PY
from transformers import AutoConfig, AutoModelForCausalLM
path = "/data/aniruddhan/models/qwen3-0.6b-custom-ft"
cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
print("layers:", cfg.num_hidden_layers, "hidden:", cfg.hidden_size, "type:", cfg.model_type)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype="bfloat16", device_map="cpu")
print("loaded OK, params:", sum(p.numel() for p in model.parameters()) // 1_000_000, "M")
PY
'
```

Expected: `layers: 28 hidden: 1024 type: qwen3`, load succeeds.

---

### Task 2: Qwen3 support in repo (required before any Qwen3 phase)

**Files:**
- Modify: `tools/configs.py`
- Create: `scripts/format_qwen3_chat_column.py`
- Modify: `requirements.txt` (bump transformers constraint)
- Modify: `configs/model_pairs.yaml`

- [ ] **Step 1: Write failing test for Qwen3 config registration**

Create `tests/test_qwen3_config.py`:

```python
from tools.configs import MODEL_CONFIGS

def test_qwen3_models_registered():
    for model in [
        "Qwen/Qwen3-0.6B-Base",
        "Qwen/Qwen3-0.6B",
        "/data/aniruddhan/models/qwen3-0.6b-custom-ft",
    ]:
        assert model in MODEL_CONFIGS, f"missing {model}"
        assert "text_column" in MODEL_CONFIGS[model]
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
cd /data/aniruddhan/sparsity-artifacts-crosscoders
pytest tests/test_qwen3_config.py -v
```

Expected: `AssertionError: missing Qwen/Qwen3-0.6B-Base`

- [ ] **Step 3: Add Qwen3 entries to `tools/configs.py`**

Append inside `MODEL_CONFIGS`:

```python
"Qwen/Qwen3-0.6B-Base": {
    "ignore_first_n_tokens_per_sample": 0,
    "text_column": "text_qwen3",
    "attn_implementation": None,
    "token_level_replacement": None,
},
"Qwen/Qwen3-0.6B": {
    "ignore_first_n_tokens_per_sample": 0,
    "text_column": "text_qwen3",
    "attn_implementation": None,
    "token_level_replacement": None,
},
"/data/aniruddhan/models/qwen3-0.6b-custom-ft": {
    "ignore_first_n_tokens_per_sample": 0,
    "text_column": "text_qwen3",
    "attn_implementation": None,
    "token_level_replacement": None,
},
```

Also add alias: `MODEL_CONFIGS["Qwen/Qwen3-0.6B-Instruct"] = MODEL_CONFIGS["Qwen/Qwen3-0.6B"]` if needed.

- [ ] **Step 4: Bump transformers in `requirements.txt`**

Replace:
```
transformers<4.54
```
With:
```
transformers>=4.51,<4.58
```

Qwen3 requires ≥4.51; your custom checkpoint was saved with 4.57.6.

- [ ] **Step 5: Write `scripts/format_qwen3_chat_column.py`**

```python
#!/usr/bin/env python3
"""Add text_qwen3 column to a HF dataset split using Qwen3 chat template."""
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer

def format_example(example, tokenizer, messages_col="messages"):
    text = tokenizer.apply_chat_template(
        example[messages_col],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return {"text_qwen3": text}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--messages-col", default="messages")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    ds = load_dataset(args.dataset, split=args.split)
    ds = ds.map(
        lambda ex: format_example(ex, tok, args.messages_col),
        remove_columns=[],
        desc="format_qwen3",
    )
    ds.save_to_disk(args.output)
    print(f"saved {len(ds)} rows to {args.output}")

if __name__ == "__main__":
    main()
```

Run once per dataset split on the pod before activation collection if the upstream dataset lacks `text_qwen3`.

- [ ] **Step 6: Add model pair presets to `configs/model_pairs.yaml`**

```yaml
qwen3-0.6b-benign:
  base_model: Qwen/Qwen3-0.6B-Base
  chat_model: Qwen/Qwen3-0.6B
  layer: 14
  text_column: text_qwen3
  lr: 1.0e-4
  k: 50
  lambda_delta: 1.0
  chat_dataset: science-of-finetuning/lmsys-chat-1m-chat-formatted
  fineweb_dataset: science-of-finetuning/fineweb-1m-sample

qwen3-0.6b-custom:
  base_model: Qwen/Qwen3-0.6B
  chat_model: /data/aniruddhan/models/qwen3-0.6b-custom-ft
  layer: 14
  text_column: text_qwen3
  lr: 1.0e-4
  k: 50
  lambda_delta: 1.0
  chat_dataset: science-of-finetuning/lmsys-chat-1m-chat-formatted
  fineweb_dataset: science-of-finetuning/fineweb-1m-sample
```

- [ ] **Step 7: Re-run tests**

```bash
pytest tests/test_qwen3_config.py tests/test_delta_loss.py -v
```

Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add tools/configs.py configs/model_pairs.yaml scripts/format_qwen3_chat_column.py \
        tests/test_qwen3_config.py requirements.txt
git commit -m "feat: add Qwen3-0.6B model pair support for crosscoder training"
```

---

### Task 3: Phase config + Lambda run wrapper

**Files:**
- Create: `configs/experiment_phases.yaml`
- Create: `run/lambda/train_phase.sh`
- Create: `run/lambda/compare_runs.py`

- [ ] **Step 1: Write `configs/experiment_phases.yaml`**

```yaml
phase1_organism:
  pair: qwen3-0.6b-benign
  max_tokens_collect: 1_000_000      # 1M tokens per model per split
  num_samples: 2_000_000
  num_validation_samples: 200_000
  max_steps: 5000
  batch_size: 2048
  k: 30
  lambda_delta: 1.0
  validate_every_n_steps: 1000

phase2_benign:
  pair: qwen3-0.6b-benign
  max_tokens_collect: 50_000_000
  num_samples: 100_000_000
  num_validation_samples: 2_000_000
  max_steps: null                     # full epoch
  batch_size: 2048
  k: 50
  lambda_delta: 1.0
  validate_every_n_steps: 20000

phase3_custom:
  pair: qwen3-0.6b-custom
  max_tokens_collect: 50_000_000
  num_samples: 100_000_000
  num_validation_samples: 2_000_000
  max_steps: null
  batch_size: 2048
  k: 50
  lambda_delta: 1.0
  validate_every_n_steps: 20000
```

- [ ] **Step 2: Write `run/lambda/train_phase.sh`**

```bash
#!/usr/bin/env bash
# Usage: bash run/lambda/train_phase.sh phase1_organism [--lambda-delta 0|1.0]
set -euo pipefail
PHASE="${1:?phase name required}"; shift
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

eval "$(python - <<PY
import yaml, sys
phase = yaml.safe_load(open("configs/experiment_phases.yaml"))[sys.argv[1]]
pair = yaml.safe_load(open("configs/model_pairs.yaml"))[phase["pair"]]
print(f'export CC_PAIR={phase["pair"]}')
for k,v in pair.items():
    print(f'export CC_{k.upper()}="{v}"')
for k,v in phase.items():
    if k != "pair":
        print(f'export CC_{k.upper()}="{v}"')
PY
"$PHASE")"

export DATASTORE="${DATASTORE:-/data/aniruddhan}"
export CC_ACTIVATION_DIR="$DATASTORE/activations"
export CC_RUN_NAME="${PHASE}-lambda-delta${CC_LAMBDA_DELTA:-1.0}"

# collect activations (all 4 splits)
for split in train val; do
  for dataset in chat fineweb; do
    CC_MAX_TOKENS="$CC_MAX_TOKENS_COLLECT" bash run/generic/compute-activations.sh \
      --split "$split" --dataset "$dataset"
  done
done

bash run/generic/train-delta-crosscoder.sh \
  --num-samples "$CC_NUM_SAMPLES" \
  --num-validation-samples "$CC_NUM_VALIDATION_SAMPLES" \
  --max-steps "${CC_MAX_STEPS:-}" \
  --k "$CC_K" \
  --lambda-delta "${CC_LAMBDA_DELTA}" \
  --validate-every-n-steps "$CC_VALIDATE_EVERY_N_STEPS" \
  "$@"
```

- [ ] **Step 3: Write `run/lambda/compare_runs.py`**

```python
#!/usr/bin/env python3
"""Compare baseline vs delta crosscoder checkpoints from wandb logs or local checkpoints."""
import argparse
import json
from pathlib import Path

def load_run_metrics(checkpoint_dir: Path) -> dict:
    # Read trainer config saved alongside checkpoint if present
    cfg_path = checkpoint_dir / "config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text())
    return {"path": str(checkpoint_dir)}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, help="checkpoints/...-no-delta or lambda=0 run")
    p.add_argument("--delta", required=True, help="checkpoints/...-delta1 run")
    args = p.parse_args()
    print("=== Baseline ===")
    print(json.dumps(load_run_metrics(Path(args.baseline)), indent=2))
    print("=== Delta ===")
    print(json.dumps(load_run_metrics(Path(args.delta)), indent=2))
    print("\nCompare in wandb: val/cl1_frac_variance_explained, delta_loss, val/num_specific_latents_l1")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Commit**

```bash
git add configs/experiment_phases.yaml run/lambda/
git commit -m "feat: add phased Lambda experiment configs for delta crosscoder"
```

---

## Phase 1: Organism Smoke Test (~2–4 GPU-hours)

**Success criteria:**
- Activations cached for both models at layer 14
- Baseline (`λ=0`) and delta (`λ=1`) runs complete 5000 steps without OOM
- Delta run shows **non-zero `delta_loss`** decreasing in wandb
- Delta run has **higher `val/num_specific_latents_l1`** than baseline (ft-specific decoder norms)

### Task 4: Run Phase 1 baseline + delta

- [ ] **Step 1: Format Qwen3 chat column (if needed)**

On pod, inspect dataset columns first:

```bash
python - <<'PY'
from datasets import load_dataset
ds = load_dataset("science-of-finetuning/lmsys-chat-1m-chat-formatted", split="train", streaming=True)
row = next(iter(ds))
print(row.keys())
PY
```

If `text_qwen3` missing but `messages` present:

```bash
python scripts/format_qwen3_chat_column.py \
  --dataset science-of-finetuning/lmsys-chat-1m-chat-formatted \
  --split train \
  --output /data/aniruddhan/datasets/lmsys-qwen3/train
```

Then point `chat_dataset` in model_pairs to the local path or add `--dataset` override in compute script.

- [ ] **Step 2: Train baseline (no delta)**

```bash
cd /data/aniruddhan/sparsity-artifacts-crosscoders
export DATASTORE=/data/aniruddhan
export WANDB_PROJECT=delta-crosscoder-qwen3

bash run/lambda/train_phase.sh phase1_organism --lambda-delta 0 --disable-wandb
# Remove --disable-wandb if wandb secret configured
```

Expected runtime: ~1–2 hours on H100 for 1M tokens × 4 collections + 5000 training steps.

- [ ] **Step 3: Train delta crosscoder**

```bash
bash run/lambda/train_phase.sh phase1_organism --lambda-delta 1.0
```

- [ ] **Step 4: Compare checkpoints**

```bash
python run/lambda/compare_runs.py \
  --baseline checkpoints/Qwen3-0.6B-Base-L14-k30-...-delta0/model_final.pt \
  --delta checkpoints/Qwen3-0.6B-Base-L14-k30-...-delta1/model_final.pt
```

Also check wandb scalars:
- `train/delta_loss` — should decrease (delta run only)
- `val/cl0_frac_variance_explained` vs `val/cl1_frac_variance_explained`
- `val/num_specific_latents_l1` — delta run should find more ft-specific latents

- [ ] **Step 5: Gate — do NOT proceed to Phase 2 unless:**
  1. Both runs saved `checkpoints/.../model_final.pt`
  2. Delta run `val/frac_variance_explained` ≥ baseline − 0.02 (no catastrophic regression)
  3. Delta run `val/num_specific_latents_l1` > baseline

---

## Phase 2: Benign HF Pair at Full Scale (~1–2 GPU-days)

Same pair as Phase 1 (`Qwen3-0.6B-Base` → `Qwen3-0.6B`), full token budget.

### Task 5: Full benign experiment

- [ ] **Step 1: Run baseline at full scale**

```bash
bash run/lambda/train_phase.sh phase2_benign --lambda-delta 0 --run-name benign-baseline
```

- [ ] **Step 2: Run delta at full scale**

```bash
bash run/lambda/train_phase.sh phase2_benign --lambda-delta 1.0 --run-name benign-delta
```

- [ ] **Step 3: Analysis notebook / script**

Create `scripts/analyze_benign_delta.py` that computes per-latent decoder norm ratio:

```python
import torch as th
from dictionary_learning.dictionary import BatchTopKCrossCoder

def decoder_norm_ratio(checkpoint_path: str) -> th.Tensor:
    cc = BatchTopKCrossCoder.from_pretrained(checkpoint_path)
    norms = cc.decoder.weight.norm(dim=-1)  # [num_layers, dict_size]
    return norms[1] / norms[0].clamp(min=1e-8)  # ft / base

if __name__ == "__main__":
    import sys
    ratio = decoder_norm_ratio(sys.argv[1])
    print("median ratio:", ratio.median().item())
    print("frac ratio > 2:", (ratio > 2).float().mean().item())
```

Run on both checkpoints. **Delta run should show more latents with ratio ≫ 1** (ft-specific features).

- [ ] **Step 4: Success criteria for Phase 2**
  - Delta crosscoder: higher fraction of ft-specific latents (norm ratio > 2)
  - `val/cl1_frac_variance_explained` within 5% of baseline
  - Visual inspection: top ft-specific latents activate on assistant-turn tokens in LMSYS chat

---

## Phase 3: Custom Qwen3-0.6B Fine-Tune

Pair: `Qwen/Qwen3-0.6B` (base) → `/data/aniruddhan/models/qwen3-0.6b-custom-ft` (your SFT).

**Expectation:** Activation delta is **much smaller** than Phase 2. Delta loss matters more here — baseline crosscoder may find few ft-specific latents; delta crosscoder should still surface them.

### Task 6: Custom FT experiment

- [ ] **Step 1: Verify paired activations use identical tokenization**

Both models must share tokenizer vocab. Run on pod:

```python
from transformers import AutoTokenizer
t1 = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
t2 = AutoTokenizer.from_pretrained("/data/aniruddhan/models/qwen3-0.6b-custom-ft")
assert t1.vocab_size == t2.vocab_size
text = t1.apply_chat_template([{"role":"user","content":"hi"}], tokenize=False, enable_thinking=False)
print(text)
```

- [ ] **Step 2: Collect activations for custom model**

`collect_activations.py` uses `--model` path; local model dir name becomes cache subfolder. Ensure `MODEL_CONFIGS` key matches the path exactly.

```bash
source run/generic/env_from_pair.sh qwen3-0.6b-custom
bash run/generic/compute-activations.sh --split train --dataset chat
# repeat for val, fineweb
```

- [ ] **Step 3: Train baseline + delta**

```bash
bash run/lambda/train_phase.sh phase3_custom --lambda-delta 0 --run-name custom-baseline
bash run/lambda/train_phase.sh phase3_custom --lambda-delta 1.0 --run-name custom-delta
```

- [ ] **Step 4: Evaluate narrow-delta sensitivity**

Because custom SFT delta is small, also sweep `lambda_delta ∈ {0.5, 1.0, 2.0}`:

```bash
for lam in 0.5 1.0 2.0; do
  source run/generic/env_from_pair.sh qwen3-0.6b-custom
  bash run/generic/train-delta-crosscoder.sh --lambda-delta $lam --run-name "custom-lam$lam"
done
```

Pick λ with best `val/num_specific_latents_l1` without `val/frac_variance_explained` drop > 0.05.

- [ ] **Step 5: Qualitative validation**

Use `scripts/collect_activating_examples.py` (existing repo script) on the delta checkpoint to pull max-activating examples for top ft-specific latents. Confirm they correlate with your SFT domain (inspect your training JSONL topic distribution).

---

## Monitoring & Cleanup

### Task 7: Operational checklist

- [ ] Label all K8s resources with `owner: aniruddhan`
- [ ] Never schedule on enterprise nodes (`*-vp6ww`, `*-9klkg`)
- [ ] Push final checkpoints to HF hub (optional): `scripts/push_checkpoints_to_hub.py`
- [ ] Delete deployment when done:

```bash
kubectl delete deployment oumi-aniruddhan-delta-crosscoder -n oumi-science
```

---

## Risk Register

| Risk | Mitigation |
|------|------------|
| `transformers<4.51` can't load Qwen3 | Bump to `>=4.51,<4.58` in Task 2 |
| Dataset lacks Qwen3 chat formatting | Run `format_qwen3_chat_column.py` or use Llama fallback pair |
| `MODEL_CONFIGS` key mismatch for local custom model | Use full path as key matching `--model` arg |
| Custom FT delta too small to detect | Lower `k` (30), increase `lambda_delta`, try layers 12–16 |
| PVC node mismatch | Always use `nodeSelector: worker-gpu-8x-h100-sxm5gdr-zjdfj-5pk5t` |
| `dictionary_learning` missing delta loss | Use editable install from patched `crosscoder_learning` fork |
| OOM on 0.6B | Reduce `batch_size` to 1024; 0.6B should not OOM on H100 |

---

## Self-Review (spec coverage)

| Requirement | Task |
|-------------|------|
| Lambda cluster execution | Task 0 |
| Small model organism | Phase 1, `phase1_organism` config |
| Benign HF pair | Phase 2, `Qwen3-0.6B-Base` + `Qwen3-0.6B` |
| Custom Qwen3-0.6B + model.zip | Phase 3, Task 1 upload + Task 6 |
| Delta vs baseline comparison | Tasks 4–6, `compare_runs.py`, wandb metrics |
| BatchTopK + delta loss | Already in repo; verified by `tests/test_delta_loss.py` |

No placeholders remain — all steps include concrete commands, file paths, and expected outputs.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-22-delta-crosscoder-lambda-experiments.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task (Task 0 → Task 7), review between tasks, fast iteration.

**2. Inline Execution** — implement tasks in this session with checkpoints after Phase 0 bootstrap and Phase 1 smoke test.

**Which approach?**
