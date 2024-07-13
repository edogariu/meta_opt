from ml_collections.config_dict import config_dict
from dataclasses import asdict

from ..utils import bcolors


def make_default(workload: str, config: config_dict.ConfigDict) -> config_dict.ConfigDict:

    # parse workload
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


def convert_configs(experiment_cfg, optimizer_cfg, base_config: config_dict.ConfigDict):
    assert experiment_cfg.experimental_setup == 'init2winit', 'this function only works in init2winit'
    assert experiment_cfg.framework == 'jax', 'init2winit only works in jax'
    assert experiment_cfg.num_opt_devices == 1, 'havent set up optimizer state sharding in init2winit yet'
    assert isinstance(base_config, config_dict.ConfigDict), 'base_config must be a ConfigDict'

    config = make_default(experiment_cfg.workload_name, base_config)
    hparam_overrides = config.hparam_overrides

    config.experiment_name = experiment_cfg.experiment_name
    config.cell = 'pw'

    # parse experiment config
    if experiment_cfg.batch_size is not None:
        ratio = hparam_overrides['batch_size'] / experiment_cfg.batch_size
        config.num_train_steps = int(config.num_train_steps * ratio)
        hparam_overrides['batch_size'] = experiment_cfg.batch_size
    if experiment_cfg.num_iters is not None:
        config.num_train_steps = experiment_cfg.num_iters
    config.num_train_steps *= experiment_cfg.num_episodes 
    hparam_overrides['rng_seed'] = experiment_cfg.seed
    config.eval_frequency = experiment_cfg.eval_every if experiment_cfg.eval_every > 0 else int(1e9)
    config.eval_steps = compute_steps(config.num_train_steps * experiment_cfg.num_episodes, experiment_cfg.eval_every)
    config.checkpoint_steps = compute_steps(config.num_train_steps * experiment_cfg.num_episodes, experiment_cfg.checkpoint_every)

    # handle printing with colors
    if experiment_cfg.print_with_colors:
        bcolors.enable()
    else:
        bcolors.disable()

    # parse optimizer config
    lr_hparams, opt_hparams = {}, {}
    hparam_overrides['l2_decay_factor'] = None  # make it so weight decay is handled by optimizer and not cost function
    if optimizer_cfg.optimizer_name == 'SGD':
        if optimizer_cfg.momentum is None:
            hparam_overrides['optimizer'] = 'sgd'
            lr_hparams = {
            'base_lr': optimizer_cfg.learning_rate,
            'schedule': 'constant',
            }
            opt_hparams = {
                'weight_decay': optimizer_cfg.weight_decay
            }
        else:
            hparam_overrides['optimizer'] = 'momentum' if not optimizer_cfg.nesterov else 'nesterov'
            lr_hparams = {
            'base_lr': optimizer_cfg.learning_rate,
            'schedule': 'constant',
            }
            opt_hparams = {
                'momentum': optimizer_cfg.momentum,
                'weight_decay': optimizer_cfg.weight_decay
            }
    elif optimizer_cfg.optimizer_name == 'AdamW':
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
    elif optimizer_cfg.optimizer_name == 'MetaOpt':
        hparam_overrides['optimizer'] = 'metaopt'
    else:
        raise NotImplementedError(optimizer_cfg.optimizer_name)
    opt_hparams.update({'experiment_cfg': asdict(experiment_cfg), 'optimizer_cfg': asdict(optimizer_cfg)})
    hparam_overrides['lr_hparams'] = lr_hparams
    hparam_overrides['opt_hparams'] = opt_hparams
    config.hparam_overrides = hparam_overrides
    return config

def compute_steps(total_iters, freq):  # compute at which steps to do something
    if freq > 0:
        l = list(range(0, total_iters, freq))
        l.append(total_iters - 1)
        return l
    else: 
        return []
