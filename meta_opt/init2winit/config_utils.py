import datetime
from ml_collections.config_dict import config_dict
from dataclasses import asdict

def make_default(workload: str) -> config_dict.ConfigDict:
    config = get_base_config()

    if workload == 'mnist':
        batch_size = 64
        train_size = 50000

        config.train_hardware = 'cpu'

        config.dataset = 'mnist'
        config.model = 'fully_connected'
        config.hparam_overrides = {
            'batch_size': batch_size,
        }
        config.num_train_steps = int(300 * train_size / batch_size)

    elif workload == 'cifar':
        batch_size = 128
        train_size = 50000

        config.train_hardware = 'tpu'
        config.tpu_type = 'jf'
        config.tpu_topology = '2x2'

        config.dataset = 'cifar10'
        config.model = 'wide_resnet'
        config.hparam_overrides = {
            'batch_size': batch_size,
            'blocks_per_group': 4,
            'channel_multiplier': 10,
            'train_size': train_size,
            'valid_size': 5000,
        }
        config.num_train_steps = int(300 * train_size / batch_size)

    elif workload == 'lm1b':
        batch_size = 256
        train_size = 425000

        config.train_hardware = 'tpu'
        config.tpu_type = 'jf'
        config.tpu_topology = '2x2'

        config.dataset = 'lm1b_v2'
        config.model = 'transformer'
        config.hparam_overrides = {
            'batch_size': batch_size,
            'emb_dim': 512,
            'num_heads': 8,
            'num_layers': 6,
            'qkv_dim': 512,
            'mlp_dim': 2048,
            'dropout_rate': 0.1,
            'attention_dropout_rate': 0.1,
            'model_dtype': 'bfloat16',
            'vocab_path': '/cns/iw-d/home/init2winit/lm1b/sentencepiece_model2'
        }
        config.num_train_steps = int(300 * train_size / batch_size)
        config.eval_train_num_batches = 256
        config.eval_num_batches = None

    else:
        raise NotImplementedError
    
    return config


def convert_configs(experiment_cfg, optimizer_cfg):
    assert experiment_cfg.experimental_setup == 'init2winit', 'this function only works in init2winit'
    assert experiment_cfg.framework == 'jax', 'init2winit only works in jax'
    assert experiment_cfg.num_opt_devices == 1, 'havent set up optimizer state sharding in init2winit yet'

    config = make_default(experiment_cfg.workload_name)
    hparam_overrides = config.hparam_overrides

    time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    config.experiment_name = f'{experiment_cfg.experiment_name}_{experiment_cfg.workload_name}_{time}'
    config.cell = 'pw'
    hparam_overrides['experiment_cfg'] = asdict(experiment_cfg)
    hparam_overrides['optimizer_cfg'] = asdict(optimizer_cfg)

    # parse experiment config
    if experiment_cfg.batch_size is not None:
        ratio = hparam_overrides['batch_size'] / experiment_cfg.batch_size
        config.num_train_steps = int(config.num_train_steps * ratio)
        hparam_overrides['batch_size'] = experiment_cfg.batch_size
    if experiment_cfg.num_iters is not None:
        config.num_train_steps = experiment_cfg.num_iters
    hparam_overrides['rng_seed'] = experiment_cfg.seed
    config.eval_frequency = experiment_cfg.eval_every if experiment_cfg.eval_every > 0 else int(1e9)
    config.eval_steps = compute_steps(config.num_train_steps * experiment_cfg.num_episodes, experiment_cfg.eval_every)
    config.checkpoint_steps = compute_steps(config.num_train_steps * experiment_cfg.num_episodes, experiment_cfg.checkpoint_every)

    # parse optimizer config
    if optimizer_cfg.optimizer_name == 'AdamW':
        hparam_overrides['optimizer'] = 'adam'
        lr_hparams = {
            'base_lr': optimizer_cfg.learning_rate,
            'schedule': 'constant',
        }
        opt_hparams = {
            'beta1': optimizer_cfg.b1,
            'beta2': optimizer_cfg.b2,
            'epsilon': optimizer_cfg.eps,
            'weight_decay': optimizer_cfg.weight_decay
        }
        hparam_overrides['lr_hparams'] = lr_hparams
        hparam_overrides['opt_hparams'] = opt_hparams
    elif optimizer_cfg.optimizer_name == 'MetaOpt':
        hparam_overrides['optimizer'] = 'metaopt'
    else:
        raise NotImplementedError(optimizer_cfg.optimizer_name)

    config.hparam_overrides = hparam_overrides
    return config

def compute_steps(total_iters, freq):  # compute at which steps to do something
    if freq > 0:
        l = list(range(0, total_iters, freq))
        l.append(total_iters - 1)
        return l
    else: 
        return []

"""Base XManager experiment configuration. Forked from `init2winit/experiments/base_config.py`"""
from ml_collections.config_dict import config_dict
def get_base_config() -> config_dict.ConfigDict:
    """Returns the base configuration for XManager jobs."""
    config = config_dict.ConfigDict()
    config.trainer = 'standard'
    # Borg parameters.
    config.experiment_name = config_dict.placeholder(str)
    config.project_name = config_dict.placeholder(str)
    # `cell` supports Brain global quota - go/marketplace-global-quota.
    # To allow XM to choose which cell to run in, set this field to 'global'.
    config.cell = config_dict.placeholder(str)
    # Note: If cns_cell is set to None, the launcher will default to 'auto',
    # meaning it will try to use the same cell as the compute cell unless it has
    # been mapped to a fallback cell, in which case use the fallback.
    config.cns_cell = 'auto'
    config.cpu_priority = 200
    config.train_hardware = 'cpu'
    config.gpu_type = 'v100'
    config.gpu_count = 1
    config.tpu_type = 'jf'
    config.tpu_topology = '2x2'
    config.tags = []
    # Additional directories to create in the experiment_dir
    config.hlo_dir = 'hlo/ttl=45d'
    # Idle polling configuration.
    config.poll_frequency_secs = 100
    config.email_frequency_secs = 3600
    config.idle_threshold_secs = 3600
    config.grace_period_secs = 7200
    # Performance parameters.
    # Number of network to host prefetches per step (e.g., passed to tf.data's
    # prefetch). Set to -1 for tf.data.AUTOTUNE.
    config.num_tf_data_prefetches = -1
    # Number of host to device prefetches per step (used in train loop).
    config.num_device_prefetches = 0
    # Number of parallel calls to make from tf.data.map. Set to -1 for
    # tf.data.AUTOTUNE.
    config.num_tf_data_map_parallel_calls = -1
    # Set this to True to have tpu matmul use float32, will be 6x slower but may
    # be more numerically stable.
    config.xla_jf_conv_full_precision = False
    # Dataset parameters.
    config.dataset = config_dict.placeholder(str)
    config.model = config_dict.placeholder(str)
    config.initializer = 'noop'
    config.data_selector = 'noop'
    # Loss/Metrics parameters.
    config.loss = 'cross_entropy'
    config.metrics = 'classification_metrics'
    # Model parameters.
    config.hparam_overrides = config_dict.placeholder(dict)
    config.training_metrics_config = config_dict.placeholder(dict)
    config.callback_configs = None
    config.checkpoint_steps = []
    config.eval_steps = []
    config.sweep = config_dict.placeholder(list)
    config.num_train_steps = config_dict.placeholder(int)
    config.eval_batch_size = config_dict.placeholder(int)
    config.eval_use_ema = False
    config.eval_num_batches = config_dict.placeholder(int)
    config.test_num_batches = config_dict.placeholder(int)
    config.eval_train_num_batches = config_dict.placeholder(int)
    config.eval_frequency = config_dict.placeholder(int)
    # Training parameters.
    config.root_dir = config_dict.placeholder(str)
    config.external_checkpoint_path = config_dict.placeholder(str, required=False)
    config.allowed_unrecognized_hparams = []
    # Vizier parameters.
    config.use_halton_generator = False
    config.vizier_study_config = None
    # config.vizier_study_config = config_dict.placeholder(vizier_pb2.StudyConfig)
    config.vizier_num_clients = config_dict.placeholder(int)
    config.vizier_max_feasible_trials = config_dict.placeholder(int)
    config.pythia_method = config_dict.placeholder(str)
    # If the Halton generator code path supported max_feasible_trials we would not
    # need a separate configuration parameter.
    config.halton_num_trials = config_dict.placeholder(int)
    config.halton_seed = config_dict.placeholder(int)
    # Optional early stopping.
    config.early_stopping_target_name = config_dict.placeholder(str)
    config.early_stopping_target_value = config_dict.placeholder(float)
    config.early_stopping_mode = config_dict.placeholder(str)
    config.early_stopping_min_steps = 0
    config.experiment_cfg = None
    config.optimizer_cfg = None
    # Prevent adding new fields. Existing fields can be overridden.
    config.lock()
    return config
    