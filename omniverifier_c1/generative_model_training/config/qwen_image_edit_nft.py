"""Eight-GPU Qwen-Image-Edit-2509 DiffusionNFT experiments."""

import os

from config.base import get_config as get_base_config


def get_config(name):
    experiments = {
        "pointwise_verification": pointwise_verification,
        "pairwise_verification": pairwise_verification,
    }
    if name not in experiments:
        raise ValueError(f"Unknown experiment: {name}; choose one of {tuple(experiments)}")
    return experiments[name]()


def _get_config(mode):
    config = get_base_config()
    config.algorithm = "DiffusionNFT"
    config.base_model = "qwen_image_edit"
    config.n_gpus = 8
    config.dataset = os.getenv("DATASET_DIR", "")
    config.pretrained.model = os.getenv("QWEN_IMAGE_EDIT_MODEL", "Qwen/Qwen-Image-Edit-2509")
    config.sample.num_steps = 10
    config.sample.max_sequence_length = 1024
    config.sample.guidance_scale = 1.0
    config.sample.noise_level = 0.7
    config.sample.num_image_per_prompt = 8
    config.sample.train_batch_size = 16
    config.sample.num_batches_per_update = 2
    config.train.batch_size = 16
    config.train.gradient_accumulation_steps = (
        config.sample.num_batches_per_update * config.sample.train_batch_size // config.train.batch_size
    )
    config.sample.ban_std_thres = 0.05
    config.sample.ban_mean_thres = 0.9
    config.sample.ban_prompt = False
    config.sample.deterministic = True
    config.sample.solver = "dpm2"
    config.resolution = 512
    config.train.beta = 0.0001
    config.decay_type = 1
    config.beta = 1.0
    config.train.adv_mode = "all"
    config.run_name = f"qwen_image_edit_2509_{mode}_diffusion_nft"
    config.save_dir = f"logs/diffusion_nft/{mode}"
    config.reward.mode = mode
    default_port = 18085 if mode == "pointwise" else 18086
    config.reward.server_url = os.getenv("REWARD_SERVER_URL", f"http://127.0.0.1:{default_port}")
    config.reward.timeout = 1800
    return config


def pointwise_verification():
    return _get_config("pointwise")


def pairwise_verification():
    return _get_config("pairwise")
