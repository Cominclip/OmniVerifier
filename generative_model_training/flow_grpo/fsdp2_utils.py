"""FSDP2 sharding used by the frozen Qwen Image Edit text encoder."""

from typing import Callable, List, Optional

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from torch.distributed.device_mesh import DeviceMesh


def prepare_fsdp_model(
    model: nn.Module,
    shard_conditions: List[Callable[[str, nn.Module], bool]],
    reshard_after_forward: bool = True,
    dp_mesh: Optional[DeviceMesh] = None,
    cpu_offload: bool = False,
    weight_dtype: torch.dtype = torch.bfloat16,
):
    fsdp_kwargs = {
        "reshard_after_forward": reshard_after_forward,
        "mesh": dp_mesh,
    }  # dp_mesh is None means distributed to all nodes.

    if cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

    fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
        param_dtype=weight_dtype,
        reduce_dtype=weight_dtype,
        output_dtype=weight_dtype,
    )

    num_layers_sharded = 0
    for n, m in reversed(list(model.named_modules())):
        if any(shard_condition(n, m) for shard_condition in shard_conditions):
            fully_shard(m, **fsdp_kwargs)
            num_layers_sharded += 1

    if num_layers_sharded == 0:
        raise ValueError("No layer modules were sharded. Please check if shard conditions are working as expected.")

    fully_shard(model, **fsdp_kwargs)
