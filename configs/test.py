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

def get_config():
    
    experiment_cfg = experiment.ExperimentConfig(
        
        # name of the experiment
        experiment_name='test',
        
        # workload details
        seed=42,
        experimental_setup='init2winit',
        workload_name='mnist', 
        full_batch=False,  # whether to do full gradient descent on one batch (fixed during the whole training) or regular minibatch SGD
        num_episodes=20,
        num_iters=500,  # if None, uses default for the workload

        # backend details
        framework='jax',
        num_batch_devices=1,
        num_opt_devices=1,

        # how often to do things
        eval_every=50,
        checkpoint_every=-1,

        # other details
        print_with_colors=False,

        # algoperf-only args
        log_every=50,
        use_wandb=False)

    # optimizer_cfg = sgd.SGDConfig(learning_rate=0.01, momentum=0.9, nesterov=False, weight_decay=None, grad_clip=None)
    # optimizer_cfg = adamw.AdamWConfig(learning_rate=0.001, b1=0.9, b2=0.999, eps=1e-8, weight_decay=None, grad_clip=None)

    # meta_optimizer_cfg = SGDConfig(learning_rate=1e-5, momentum=0, nesterov=False, weight_decay=None, grad_clip=None)
    meta_optimizer_cfg = adamw.AdamWConfig(learning_rate=4e-4, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0, grad_clip=None)
    optimizer_cfg = metaopt.MetaOptConfig(base_learning_rate=0.001, weight_decay=1e-4, grad_clip=None,
                                H=16, HH=2, m_method='scalar', scale_by_adam_betas=None, 
                                fake_the_dynamics=False, freeze_gpc_params=False, freeze_cost_fn_during_rollouts=False,
                                meta_optimizer_cfg=meta_optimizer_cfg, use_bfloat16=False)

    # meta_optimizer_cfg = sgd.SGDConfig(learning_rate=1e-5, momentum=0, nesterov=False, weight_decay=None, grad_clip=None)
    # optimizer_cfg = metaopt.MetaOptConfig(base_learning_rate=0.001, weight_decay=1e-4, grad_clip=None,
    #                               H=16, HH=2, m_method='diagonal', scale_by_adam_betas=(0.9, 0.999),
    #                               meta_optimizer_cfg=meta_optimizer_cfg, meta_grad_clip=10.0)
    
    if experiment_cfg.experimental_setup == 'algoperf':
        assert not IS_INTERNAL, 'havent set up algoperf on internal google yet'
        return experiment_cfg, optimizer_cfg
    elif experiment_cfg.experimental_setup == 'init2winit':
        assert IS_INTERNAL, 'havent set up init2winit on external google yet'
        return config_utils.convert_configs(experiment_cfg, optimizer_cfg, base_config.get_base_config())
    else:
        raise NotImplementedError(experiment_cfg.experimental_setup)
