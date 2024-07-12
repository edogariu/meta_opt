try:
    from meta_opt.experiment import ExperimentConfig
    from meta_opt.optimizers import SGDConfig, AdamWConfig, MetaOptConfig
except:  # internal google imports
    import jax
    from google3.learning.deepmind.python.adhoc_import import binary_import
    with binary_import.AutoGoogle3():
        from init2winit.experiments.meta_opt.meta_opt.init2winit import config_utils
        from init2winit.experiments.meta_opt.meta_opt.experiment import ExperimentConfig
        from init2winit.experiments.meta_opt.meta_opt.optimizers import SGDConfig, AdamWConfig, MetaOptConfig

def get_config():
    
    experiment_cfg = ExperimentConfig(
        
        # name of the experiment
        experiment_name='test',
        
        # workload details
        seed=42,
        experimental_setup='algoperf',
        workload_name='mnist', 
        full_batch=False,  # whether to do full gradient descent on one batch (fixed during the whole training) or regular minibatch SGD
        num_episodes=20,
        num_iters=500,  # if None, uses default for the workload

        framework='jax',
        num_batch_devices=4,
        num_opt_devices=2,

        # how often to do things
        eval_every=50,
        checkpoint_every=-1,

        # algoperf-specific args
        log_every=50,
        use_wandb=False,
        print_with_colors=True)

    # optimizer_cfg = SGDConfig(learning_rate=0.01, momentum=0.9, nesterov=False, weight_decay=None, grad_clip=None)
    # optimizer_cfg = AdamWConfig(learning_rate=0.001, b1=0.9, b2=0.999, eps=1e-8, weight_decay=None, grad_clip=None)

    # meta_optimizer_cfg = SGDConfig(learning_rate=1e-5, momentum=0, nesterov=False, weight_decay=None, grad_clip=None)
    meta_optimizer_cfg = AdamWConfig(learning_rate=4e-4, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0, grad_clip=None)
    optimizer_cfg = MetaOptConfig(base_learning_rate=0.001, weight_decay=1e-4, grad_clip=None,
                                H=16, HH=2, m_method='scalar', scale_by_adam_betas=None, 
                                fake_the_dynamics=False, freeze_gpc_params=False, freeze_cost_fn_during_rollouts=False,
                                meta_optimizer_cfg=meta_optimizer_cfg, use_bfloat16=False)

    # meta_optimizer_cfg = SGDConfig(learning_rate=1e-5, momentum=0, nesterov=False, weight_decay=None, grad_clip=None)
    # optimizer_cfg = MetaOptConfig(base_learning_rate=0.001, weight_decay=1e-4, grad_clip=None,
    #                               H=16, HH=2, m_method='diagonal', scale_by_adam_betas=(0.9, 0.999),
    #                               meta_optimizer_cfg=meta_optimizer_cfg, meta_grad_clip=10.0)
    
    if experiment_cfg.experimental_setup == 'algoperf':
        return experiment_cfg, optimizer_cfg
    elif experiment_cfg.experimental_setup == 'init2winit':
        return config_utils.convert_configs(experiment_cfg, optimizer_cfg)
    else:
        raise NotImplementedError(experiment_cfg.experimental_setup)
