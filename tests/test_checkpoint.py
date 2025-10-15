


import json
import os
import shutil
import uuid

import pytest

from verl.utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt


@pytest.fixture
def save_checkpoint_path():
    ckpt_dir = os.path.join("checkpoints", str(uuid.uuid4()))
    os.makedirs(ckpt_dir, exist_ok=True)
    yield ckpt_dir
    shutil.rmtree(ckpt_dir, ignore_errors=True)


def test_find_latest_ckpt(save_checkpoint_path):
    with open(os.path.join(save_checkpoint_path, CHECKPOINT_TRACKER), "w") as f:
        json.dump({"last_global_step": 10}, f, ensure_ascii=False, indent=2)

    assert find_latest_ckpt(save_checkpoint_path) is None
    os.makedirs(os.path.join(save_checkpoint_path, "global_step_10"), exist_ok=True)
    assert find_latest_ckpt(save_checkpoint_path) == os.path.join(save_checkpoint_path, "global_step_10")


def test_remove_obsolete_ckpt(save_checkpoint_path):
    for step in range(5, 30, 5):
        os.makedirs(os.path.join(save_checkpoint_path, f"global_step_{step}"), exist_ok=True)

    remove_obsolete_ckpt(save_checkpoint_path, global_step=30, best_global_step=10, save_limit=3)
    for step in range(5, 30, 5):
        is_exist = step in [10, 25]
        assert os.path.exists(os.path.join(save_checkpoint_path, f"global_step_{step}")) == is_exist
