from typing import Union, Callable, Optional, Tuple
from flax import struct

"""
Optimizer hyperparameters and meta-information.
Like `nn.py`, this is made to capture multiple deep learning frameworks (jax, pytorch).
"""

@struct.dataclass
class OptimizerConfig:
    optimizer_name: str
    self_tuning: bool
    reset_opt_state: bool

# -------------------------------------------------------------------------------------------------
# ----------------------------------- Standard Optax Optimizers -----------------------------------
# -------------------------------------------------------------------------------------------------

@struct.dataclass
class SGDConfig():
    learning_rate: Union[float, Callable[[int], float]]  # learning rate or schedule
    momentum: float
    nesterov: bool
    weight_decay: float
    grad_clip: float

    optimizer_name: str = 'SGD'
    self_tuning: bool = False
    reset_opt_state: bool = True  # whether to also reset the optimizer state during the episodic resets

@struct.dataclass
class AdamWConfig():
    learning_rate: Union[float, Callable[[int], float]]  # learning rate or schedule
    b1: float
    b2: float
    eps: float
    weight_decay: float
    grad_clip: float

    optimizer_name: str = 'AdamW'
    self_tuning: bool = False
    reset_opt_state: bool = True  # whether to also reset the optimizer state during the episodic resets

# -------------------------------------------------------------------------------------------------
# ----------------------------------- Our Meta-Optimizer ------------------------------------------
# -------------------------------------------------------------------------------------------------

@struct.dataclass
class MetaOptConfig():
    # params of the base optimizer
    initial_learning_rate: float
    weight_decay: float
    grad_clip: float
    scale_by_adam_betas: Optional[Tuple[float, float]]  # set to `None` to not rescale disturbances with Adam rescaling

    # params of the meta-optimizer
    H: int  # number of past disturbances to use
    HH: int  # rollout length
    m_method: str  # how to compute controls from past disturbances, must be one of ['scalar', 'diagonal', 'full']
    meta_optimizer_cfg: OptimizerConfig  # presumably one of `SGDConfig` or `AdamWConfig`
    meta_grad_clip: float
    fake_the_dynamics: bool  # whether to use the gradient buffer to time-evolve the system rather than taking bona fide train_steps during counterfactual rollout
    freeze_meta_params: bool  # whether to skip the controller update step. set this to False to learn optimizer, and True to deploy it
    freeze_batch_during_rollouts: bool  # whether to use one fixed batch during counterfactual rollouts

    # jax implementation details
    jax_pmap_in_rollouts: bool = False  #  whether to use parallelized functions within metaopt, (regardless, we pmap on the outer train_step call)
    jax_compute_loss_with_scan: bool = True  # whether to use jax.lax.scan to calculate the counterfactual loss (i.e. whether to scan or loop over train_step calls)

    optimizer_name: str = 'MetaOpt'
    self_tuning: bool = True
    reset_opt_state: bool = True  # Whether to also reset the optimizer state during the episodic resets. Dont worry, this resets everything except the M parameters (including things like disturbance transformation state, for example)
