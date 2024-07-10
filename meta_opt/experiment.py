from flax import struct

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
    workload_name: str  # one of ['mnist', 'cifar', 'wmt', 'ogbg', ...]
    full_batch: bool
    num_episodes: int
    num_iters: int = None
    batch_size: int = None  # does nothing for init2winit experiments

    # backend details. the number of devices should multiply to `jax.local_device_count()`
    framework: str = 'jax'  # one of ['pytorch', 'jax']
    num_batch_devices: int = 4
    num_opt_devices: int = 2

    # how often to do things. set to `-1` to never do them
    eval_every: int = -1
    checkpoint_every: int = -1
    log_every: int = -1  # does nothing for init2winit experiments

    # other details
    print_with_colors: bool = True  # whether to use colors when printing
    use_wandb: bool = False
