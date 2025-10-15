
"""
Critic config
"""

from dataclasses import dataclass, field

from ..actor.config import FSDPConfig, ModelConfig, OffloadConfig, OptimConfig


@dataclass
class CriticConfig:
    strategy: str = "fsdp"
    global_batch_size: int = 256
    """number of samples per minibatch for updating critic"""
    micro_batch_size_per_device_for_update: int = 4
    """number of samples per forward pass for updating critic"""
    micro_batch_size_per_device_for_experience: int = 16
    """number of samples per forward pass for computing values"""
    max_grad_norm: float = 1.0
    """number to clip grad norm"""
    cliprange_value: float = 0.5
    """clip range for value loss"""
    loss_avg_mode: str = "token"
    """loss average mode: `token`, `seq`"""
    ppo_epochs: int = 1
    """number of ppo epochs for each rollout batch"""
    padding_free: bool = False
    """use padding-free training"""
    dynamic_batching: bool = True
    """enable dynamic batching"""
    ulysses_size: int = 1
    """ulysses sequence parallel size"""
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    # below are auto keys
    global_batch_size_per_device: int = field(default=-1, init=False)
