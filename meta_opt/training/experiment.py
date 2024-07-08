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
    framework: str = 'jax'  # one of ['pytorch', 'jax']
    num_iters: int = None
    batch_size: int = None

    # how often to do things. set to `-1` to never do them
    eval_every: int = None
    checkpoint_every: int = None
    print_every: int = None
    log_every: int = None

    # other details
    profile: bool = True  # whether to profile or not
    print_with_colors: bool = True  # whether to use colors when printing
    use_wandb: bool = False
    resume_last_run: bool = False
    overwrite: bool = True
