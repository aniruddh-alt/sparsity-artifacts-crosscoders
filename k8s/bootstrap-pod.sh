#!/usr/bin/env bash
# Apply pod manifest and bootstrap repos on the Lambda cluster.
#
# Usage:
#   bash k8s/bootstrap-pod.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/lambdaconfig}"
export NAMESPACE="${NAMESPACE:-oumi-science}"
DEPLOYMENT="oumi-aniruddhan-delta-crosscoder"

kubectl apply -f "$REPO_ROOT/k8s/aniruddhan-delta-crosscoder.yaml"
echo "Waiting for pod..."
kubectl rollout status "deployment/$DEPLOYMENT" -n "$NAMESPACE" --timeout=600s

POD=$(kubectl get pod -n "$NAMESPACE" -l "name=$DEPLOYMENT" -o jsonpath='{.items[0].metadata.name}')
echo "Pod ready: $POD"
kubectl exec -n "$NAMESPACE" "$POD" -- nvidia-smi -L

kubectl exec -n "$NAMESPACE" "$POD" -- bash -c '
  set -euo pipefail
  source /opt/conda/etc/profile.d/conda.sh && conda activate base
  cd /data/aniruddhan
  if [[ ! -d sparsity-artifacts-crosscoders ]]; then
    git clone https://github.com/science-of-finetuning/sparsity-artifacts-crosscoders.git
  fi
  if [[ ! -d dictionary-learning ]]; then
    git clone https://github.com/science-of-finetuning/crosscoder_learning.git dictionary-learning
  fi
  cd sparsity-artifacts-crosscoders
  DICTIONARY_LEARNING_PATH=/data/aniruddhan/dictionary-learning bash scripts/setup_env.sh
  python scripts/list_model_pairs.py
'

echo "Bootstrap complete. Exec: kubectl exec -it $POD -n $NAMESPACE -- bash"
