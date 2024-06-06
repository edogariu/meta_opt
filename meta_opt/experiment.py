from typing import Union
from flax import struct

@struct.dataclass
class ExperimentConfig:
    """Experiment hyperparameters used to minimize obnoxious kwarg plumbing."""

    # name of the experiment
    experiment_name: str

    # workload details
    workload_name: str  # one of ['MNIST', 'CIFAR', 'WMT', 'GNN', ...]
    framework: str  # one of ['pytorch', 'jax']
    num_iters: int
    batch_size: int
    full_batch: bool
    reset_opt_state: bool  # whether to reset optimizer state between episodes

    # how often to do things
    eval_every: Union[int, None]
    checkpoint_every: Union[int, None]
    print_every: Union[int, None]
    log_every: Union[int, None]
    reset_every: Union[int, None]

    # other details
    profile: bool  # whether to profile or not
    use_wandb: bool
    resume_last_run: bool
    overwrite: bool
