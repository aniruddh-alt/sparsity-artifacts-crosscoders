#!/usr/bin/env bash
# End-to-end driver: waits for Phase 2 training, then runs analysis + Phase 3
# training in parallel, then Phase 3 analysis.
#
# Assumes:
#   * Phase 2 training already running (PIDs traced via checkpoints).
#   * Phase 3 activation collection already running on GPUs 2-5.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CKPT_BASE=/data/aniruddhan/sparsity-artifacts-crosscoders/checkpoints
LOGDIR=/data/aniruddhan/logs/driver
mkdir -p "$LOGDIR"

PHASE2_BASELINE_DIR="Qwen3-0.6B-Base-L14-k50-lr1e-04-phase2-baseline-local-shuffling-Crosscoder-delta0-reconmse_layer_sum"
PHASE2_DELTA_DIR="Qwen3-0.6B-Base-L14-k50-lr1e-04-phase2-delta-local-shuffling-Crosscoder-delta1-reconmse_layer_sum"
PHASE3_BASELINE_DIR="Qwen3-0.6B-L14-k50-lr1e-04-phase3-baseline-local-shuffling-Crosscoder-delta0-reconmse_layer_sum"
PHASE3_DELTA_DIR="Qwen3-0.6B-L14-k50-lr1e-04-phase3-delta-local-shuffling-Crosscoder-delta1-reconmse_layer_sum"

wait_for_checkpoint() {
  local name="$1"
  local f="$CKPT_BASE/$name/model_final.pt"
  echo "[driver] waiting for $f ..."
  local last_ts=""
  while [[ ! -f "$f" ]]; do
    sleep 60
    # Heartbeat every 5 min based on the master log to detect crashes
    local now=$(date +%s)
    if [[ -z "$last_ts" || $((now - last_ts)) -gt 300 ]]; then
      last_ts=$now
      echo "[driver] still waiting for $name ($(date '+%H:%M:%S'))"
    fi
  done
  echo "[driver] found $f"
}

wait_for_activations() {
  local model_stub="$1" dataset_dir="$2" split="$3"
  local f="/data/aniruddhan/activations/$model_stub/$dataset_dir/$split/tokens.pt"
  echo "[driver] waiting for $f ..."
  while [[ ! -f "$f" ]]; do
    sleep 30
  done
  echo "[driver] found $f"
}

# ----- Wait for Phase 2 to finish -----
wait_for_checkpoint "$PHASE2_BASELINE_DIR"
wait_for_checkpoint "$PHASE2_DELTA_DIR"

echo
echo "[driver] === Phase 2 training finished. ==="

# ----- Confirm Phase 3 activations are ready (custom FT only) -----
wait_for_activations qwen3-0.6b-custom-ft lmsys-qwen3-phase2 train-coltext_qwen3
wait_for_activations qwen3-0.6b-custom-ft fineweb-1m-sample   train
wait_for_activations qwen3-0.6b-custom-ft lmsys-qwen3-phase2 validation-coltext_qwen3
wait_for_activations qwen3-0.6b-custom-ft fineweb-1m-sample   validation

echo
echo "[driver] === Phase 3 activations ready. Launching Phase 2 analysis + Phase 3 training in parallel. ==="

# ----- Phase 2 analysis (GPUs 2 + 3) and Phase 3 training (GPUs 0 + 1) in parallel -----
bash run/lambda/analyze_checkpoint.sh "$PHASE2_BASELINE_DIR" \
    Qwen/Qwen3-0.6B-Base Qwen/Qwen3-0.6B lmsys-qwen3-phase2 2 \
  > "$LOGDIR/phase2-baseline.analysis.log" 2>&1 &
P2_BASE_PID=$!
bash run/lambda/analyze_checkpoint.sh "$PHASE2_DELTA_DIR" \
    Qwen/Qwen3-0.6B-Base Qwen/Qwen3-0.6B lmsys-qwen3-phase2 3 \
  > "$LOGDIR/phase2-delta.analysis.log" 2>&1 &
P2_DELTA_PID=$!
bash run/lambda/parallel_train_phase3.sh \
  > "$LOGDIR/phase3-train.log" 2>&1 &
P3_TRAIN_PID=$!

echo "[driver] phase2-baseline-analysis PID=$P2_BASE_PID"
echo "[driver] phase2-delta-analysis    PID=$P2_DELTA_PID"
echo "[driver] phase3-training          PID=$P3_TRAIN_PID"

wait "$P2_BASE_PID" || echo "[driver] WARN Phase 2 baseline analysis exited non-zero"
echo "[driver] Phase 2 baseline analysis done."

wait "$P2_DELTA_PID" || echo "[driver] WARN Phase 2 delta analysis exited non-zero"
echo "[driver] Phase 2 delta analysis done."

wait "$P3_TRAIN_PID" || echo "[driver] WARN Phase 3 training exited non-zero"
echo "[driver] Phase 3 training done."

echo
echo "[driver] === Phase 3 training finished. Running Phase 3 analysis. ==="

# ----- Phase 3 analysis (GPUs 0 + 1, parallel) -----
bash run/lambda/analyze_checkpoint.sh "$PHASE3_BASELINE_DIR" \
    Qwen/Qwen3-0.6B /data/aniruddhan/models/qwen3-0.6b-custom-ft lmsys-qwen3-phase2 0 \
  > "$LOGDIR/phase3-baseline.analysis.log" 2>&1 &
P3_BASE_PID=$!
bash run/lambda/analyze_checkpoint.sh "$PHASE3_DELTA_DIR" \
    Qwen/Qwen3-0.6B /data/aniruddhan/models/qwen3-0.6b-custom-ft lmsys-qwen3-phase2 1 \
  > "$LOGDIR/phase3-delta.analysis.log" 2>&1 &
P3_DELTA_PID=$!

wait "$P3_BASE_PID" || echo "[driver] WARN Phase 3 baseline analysis exited non-zero"
wait "$P3_DELTA_PID" || echo "[driver] WARN Phase 3 delta analysis exited non-zero"

echo
echo "[driver] === All analyses done. Running summarize_results.py ==="
python run/lambda/summarize_results.py --output /data/aniruddhan/results/summary.md \
  2>&1 | tee "$LOGDIR/summary.log" || echo "[driver] WARN summary script exited non-zero"

echo
echo "[driver] === END-TO-END DONE ==="
echo "[driver] summary at /data/aniruddhan/results/summary.md"
