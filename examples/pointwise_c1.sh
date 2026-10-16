#!/bin/bash

set -euxo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
VERIFIER_PYTHON="${VERIFIER_PYTHON:-python3}"
TRAIN_FILES="${TRAIN_FILES:-}"
: "${TRAIN_FILES:?Set TRAIN_FILES to the verifier training dataset}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

"$VERIFIER_PYTHON" scripts/train_c1.py \
    config=examples/config_c1.yaml \
    "data.train_files='${TRAIN_FILES}'" \
    data.initial_image_key=initial_image \
    worker.rollout.limit_images=2 \
    data.verification_mode=pointwise \
    data.format_prompt=./examples/format_prompt/omniverifier_c1.jinja \
    data.max_prompt_length=4096 \
    data.max_response_length=16384 \
    data.rollout_batch_size=512 \
    worker.actor.ulysses_size=8 \
    "worker.actor.model.model_path=${MODEL_PATH}" \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.global_batch_size=512 \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    worker.actor.clip_ratio_dual=10.0 \
    worker.actor.loss_avg_mode=token \
    worker.rollout.n=8 \
    worker.rollout.max_model_len=20480 \
    worker.rollout.max_num_batched_tokens=20480 \
    worker.rollout.tensor_parallel_size=4 \
    worker.reward.reward_function=./examples/reward_function/omniverifier_c1.py:compute_score \
    algorithm.adv_estimator=grpo \
    algorithm.disable_kl=True \
    algorithm.use_kl_loss=False \
    algorithm.kl_coef=0.0 \
    algorithm.online_filtering=True \
    algorithm.filter_key=accuracy \
    algorithm.filter_low=0.01 \
    algorithm.filter_high=0.99 \
    trainer.experiment_name=qwen3_vl_8b_omniverifier_c1_pointwise_dapo \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 "$@"
