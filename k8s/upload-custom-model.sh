#!/usr/bin/env bash
# Upload local Qwen3 custom fine-tune (model.zip) to the Lambda pod PVC.
#
# Usage:
#   bash k8s/upload-custom-model.sh
#   bash k8s/upload-custom-model.sh ~/Downloads/model.zip

set -euo pipefail

MODEL_ZIP="${1:-$HOME/Downloads/model.zip}"
NAMESPACE="${NAMESPACE:-oumi-science}"
DEPLOYMENT="${DEPLOYMENT:-oumi-aniruddhan-delta-crosscoder}"
REMOTE_DIR="/data/aniruddhan/models/qwen3-0.6b-custom-ft"
LOCAL_EXTRACT="${LOCAL_EXTRACT:-/tmp/qwen3-custom-ft}"

export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/lambdaconfig}"

if [[ ! -f "$MODEL_ZIP" ]]; then
  echo "Error: model zip not found at $MODEL_ZIP" >&2
  exit 1
fi

POD=$(kubectl get pod -n "$NAMESPACE" -l "name=$DEPLOYMENT" -o jsonpath='{.items[0].metadata.name}')
if [[ -z "$POD" ]]; then
  echo "Error: no pod found for deployment $DEPLOYMENT in $NAMESPACE" >&2
  echo "Apply k8s/aniruddhan-delta-crosscoder.yaml first." >&2
  exit 1
fi

echo "Using pod: $POD"
rm -rf "$LOCAL_EXTRACT"
mkdir -p "$LOCAL_EXTRACT"
unzip -q "$MODEL_ZIP" -d "$LOCAL_EXTRACT"

kubectl exec -n "$NAMESPACE" "$POD" -- mkdir -p "$REMOTE_DIR"
kubectl cp "$LOCAL_EXTRACT/." "$NAMESPACE/$POD:$REMOTE_DIR/"

echo "Verifying checkpoint on pod..."
kubectl exec -n "$NAMESPACE" "$POD" -- bash -c '
  source /opt/conda/etc/profile.d/conda.sh && conda activate base
  python - <<PY
from transformers import AutoConfig
path = "/data/aniruddhan/models/qwen3-0.6b-custom-ft"
cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
print("OK:", cfg.model_type, "layers=", cfg.num_hidden_layers, "hidden=", cfg.hidden_size)
PY
'

echo "Upload complete: $REMOTE_DIR"
