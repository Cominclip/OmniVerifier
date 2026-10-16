#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
export REWARD_GPUS="${REWARD_GPUS:-8,9,10,11,12,13,14,15}"
export REWARD_SERVER_URL="${REWARD_SERVER_URL:-http://127.0.0.1:18085}"
# Set DATASET_DIR and POINTWISE_VERIFIER_MODEL_PATH.
bash scripts/train.sh pointwise "$@"
