from typing import Optional
import json

from flax import struct
from dataclasses import asdict

@struct.dataclass
class ExperimentConfig:
    """
    Experiment hyperparameters used to minimize obnoxious kwarg plumbing.
    Parameters defaulting to `None` are optional and workload-specific,
        and are handled in `workload_defaults`
    """

    # name of the experiment
    experiment_name: str

    # workload details
    seed: int
    experimental_setup: str  # one of ['algoperf', 'init2winit']
    workload_name: str  # one of ['mnist', 'cifar', 'wmt', 'ogbg', 'lm1b', ...]
    full_batch: bool
    num_episodes: int
    num_iters: Optional[int] = None
    batch_size: Optional[int] = None

    # backend details. the number of devices should multiply to `jax.local_device_count()`
    framework: str = 'jax'  # one of ['pytorch', 'jax']
    num_batch_devices: Optional[int] = None
    num_opt_devices: Optional[int] = None

    # how often to do things. set to `-1` to never do them
    eval_every: int = -1
    checkpoint_every: int = -1

    # other details
    print_with_colors: bool = True  # whether to use colors when printing
    
    # algoperf-specific args
    log_every: int = -1
    use_wandb: bool = False
