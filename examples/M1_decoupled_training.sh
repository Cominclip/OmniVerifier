set -x
export PYTHONUNBUFFERED=1

project_name='OmniVerifier-M1'
experiment_name="decoupled_training"

MODEL_PATH='Qwen/Qwen3-VL-8B-Instruct'
TRAIN_DATA_PATH=''
VAL_DATA_PATH=''

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=${TRAIN_DATA_PATH} \
    data.val_files=${VAL_DATA_PATH} \
    data.rollout_batch_size=384 \
    worker.actor.global_batch_size=192\
    data.max_prompt_length=6143 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    worker.reward.reward_function=./examples/reward_function/M1_decoupled_training.py:compute_score \
    data.format_prompt=./examples/format_prompt/M1.jinja \
    worker.rollout.limit_images=8 \
    worker.rollout.val_override_config.temperature=0.01 \
    worker.rollout.val_override_config.top_p=0.001 \
    algorithm.disable_kl=True \
    algorithm.online_filtering=False \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=8 \
    trainer.total_epochs=50 \
    trainer.max_try_make_batch=-1 \
    trainer.save_checkpoint_path=${experiment_name} \
    trainer.find_last_checkpoint=False