try:
    base_config = None
    from meta_opt import experiment
    from meta_opt.optimizers import sgd, adamw, metaopt
    from meta_opt.init2winit import config_utils
    IS_INTERNAL = False
except:  # internal google imports
    import jax
    from google3.learning.deepmind.python.adhoc_import import binary_import
    with binary_import.AutoGoogle3():
        from init2winit.experiments import base_config
        from init2winit.experiments.meta_opt.meta_opt.init2winit import config_utils
        from init2winit.experiments.meta_opt.meta_opt.optimizers import sgd, adamw, metaopt
        from init2winit.experiments.meta_opt.meta_opt import experiment
    IS_INTERNAL = True

import dataclasses
import copy

def get_config():
    
    experiment_cfg = experiment.ExperimentConfig(
        
        # name of the experiment
        experiment_name='cifar_sgd_sweep',
        
        # workload details
        seed=42,
        experimental_setup='init2winit',
        workload_name='cifar', 
        full_batch=False,  # whether to do full gradient descent on one batch (fixed during the whole training) or regular minibatch SGD
        num_episodes=1,
        num_iters=None,  # if None, uses default for the workload

        # backend details
        num_batch_devices=8,
        num_opt_devices=1,

        # how often to do things
        eval_every=1000,
        checkpoint_every=-1,

        # other details
        print_with_colors=False,

        # algoperf-only args
        log_every=50,
        use_wandb=False)

    optimizer_cfg = sgd.SGDConfig(learning_rate=None,
                                  momentum=None,
                                  nesterov=None,
                                  weight_decay=None,
                                  grad_clip=None)

    sweep = []
    for lr in [1e-4, 1e-3, 1e-2, 1e-1]:
        for momentum in [0., 0.8, 0.9, 0.99]:
            for nesterov in [True, False]:
                for wd in [1e-4, 1e-3]:
                    o = dataclasses.replace(optimizer_cfg, learning_rate=lr, momentum=momentum, nesterov=nesterov, weight_decay=wd)
                    sweep.append((experiment_cfg, o))
    
    if experiment_cfg.experimental_setup == 'algoperf':
        return sweep
    elif experiment_cfg.experimental_setup == 'init2winit':
        assert IS_INTERNAL, 'havent set up init2winit on external yet'
        cfgs = [config_utils.convert_configs(experiment_cfg, optimizer_cfg, base_config.get_base_config()) for (experiment_cfg, optimizer_cfg) in sweep]
        ret = cfgs[0]
        ret.sweep = [copy.deepcopy(c.hparam_overrides) for c in cfgs]
        ret.unlock()
        ret.hparam_overrides = {}
        ret.lock()
        return ret
    else:
        raise NotImplementedError(experiment_cfg.experimental_setup)
