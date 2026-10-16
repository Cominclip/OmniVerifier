#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:?Choose pointwise or pairwise}"
shift
case "$mode" in
    pointwise) reward_port=18085 ;;
    pairwise) reward_port=18086 ;;
    *) echo "Unknown verification mode: $mode" >&2; exit 1 ;;
esac

TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
REWARD_GPUS="${REWARD_GPUS:-8,9,10,11,12,13,14,15}"
TRAIN_PYTHON="${TRAIN_PYTHON:-python3}"
REWARD_PYTHON="${REWARD_PYTHON:-python3}"
START_REWARD_SERVER="${START_REWARD_SERVER:-1}"
export REWARD_SERVER_URL="${REWARD_SERVER_URL:-http://127.0.0.1:$reward_port}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

if [[ -z "${DATASET_DIR:-}" ]]; then
    echo "Set DATASET_DIR to the directory containing train_metadata.jsonl." >&2
    exit 1
fi
for split in train; do
    if [[ ! -s "$DATASET_DIR/${split}_metadata.jsonl" ]]; then
        echo "Missing dataset: $DATASET_DIR/${split}_metadata.jsonl" >&2
        exit 1
    fi
done
IFS=',' read -r -a train_devices <<< "$TRAIN_GPUS"
IFS=',' read -r -a reward_devices <<< "$REWARD_GPUS"
if [[ "${#train_devices[@]}" -ne 8 || "${#reward_devices[@]}" -ne 8 ]]; then
    echo "TRAIN_GPUS and REWARD_GPUS must each contain eight GPU IDs." >&2
    exit 1
fi
for devices in "$TRAIN_GPUS" "$REWARD_GPUS"; do
    IFS=',' read -r -a ids <<< "$devices"
    for i in "${!ids[@]}"; do
        if [[ -z "${ids[$i]}" ]]; then
            echo "GPU IDs must not be empty." >&2; exit 1
        fi
        for j in "${!ids[@]}"; do
            if [[ "$i" -ne "$j" && "${ids[$i]}" == "${ids[$j]}" ]]; then
                echo "Each GPU list must contain distinct IDs." >&2; exit 1
            fi
        done
    done
done
if [[ "$START_REWARD_SERVER" == 1 ]]; then
    for train_device in "${train_devices[@]}"; do
        for reward_device in "${reward_devices[@]}"; do
            if [[ "$train_device" == "$reward_device" ]]; then
                echo "Local training and reward GPU lists must not overlap." >&2
                exit 1
            fi
        done
    done
elif [[ "$START_REWARD_SERVER" != 0 ]]; then
    echo "START_REWARD_SERVER must be 0 or 1." >&2; exit 1
fi

reward_pid=""
cleanup() {
    if [[ -n "$reward_pid" ]]; then
        kill "$reward_pid" 2>/dev/null || true
        wait "$reward_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "$START_REWARD_SERVER" == 1 ]]; then
    reward_log_dir="${REWARD_LOG_DIR:-logs/reward}"
    mkdir -p "$reward_log_dir"
    CUDA_VISIBLE_DEVICES="$REWARD_GPUS" "$REWARD_PYTHON" "reward-server/app_${mode}.py" \
        --port "$reward_port" --max-input-tokens 4096 --max-output-tokens 16384 \
        > "$reward_log_dir/${mode}.log" 2>&1 &
    reward_pid=$!
fi

"$TRAIN_PYTHON" - "$REWARD_SERVER_URL" "$mode" "$reward_pid" "${REWARD_STARTUP_TIMEOUT:-1800}" <<'PY'
import json
import os
import sys
import time
from urllib.request import urlopen

url, mode, pid, timeout = sys.argv[1:]
deadline = time.monotonic() + float(timeout)
last_error = "not ready"
while time.monotonic() < deadline:
    if pid:
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            raise SystemExit(f"Reward server exited; check logs/reward/{mode}.log")
    try:
        with urlopen(url.rstrip('/') + '/health', timeout=5) as response:
            state = json.load(response)
    except (OSError, ValueError) as error:
        last_error = str(error)
        time.sleep(2)
        continue
    if state.get('mode') != mode or state.get('workers') != 8:
        raise SystemExit(f"Expected eight-worker {mode} reward server, received: {state}")
    if pid and state.get('process_id') != int(pid):
        raise SystemExit("Another reward service is already listening; use START_REWARD_SERVER=0 to connect to it")
    if state.get('max_input_tokens') != 4096 or state.get('max_output_tokens') != 16384:
        raise SystemExit(f"Reward server must use 4096 input / 16384 output tokens: {state}")
    if state.get('ready'):
        print(f"{mode} reward service is ready at {url}")
        break
else:
    raise SystemExit(f"Reward server startup timed out: {last_error}")
PY

CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "$TRAIN_PYTHON" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train_nft_qwen_image_edit.py \
    --config="config/qwen_image_edit_nft.py:${mode}_verification" "$@"
