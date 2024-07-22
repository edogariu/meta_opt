try:
    base_config = None
    from meta_opt import experiment
    from meta_opt.optimizers import schedules, sgd, adamw, metaopt, polyak, cocob, dog, dowg, mechanic
    from meta_opt.init2winit import config_utils
    IS_INTERNAL = False
except:  # internal google imports
    import jax
    from google3.learning.deepmind.python.adhoc_import import binary_import
    with binary_import.AutoGoogle3():
        from init2winit.experiments import base_config
        from init2winit.experiments.meta_opt.meta_opt.init2winit import config_utils
        from init2winit.experiments.meta_opt.meta_opt.optimizers import sgd, adamw, metaopt, schedules
        from init2winit.experiments.meta_opt.meta_opt import experiment
    IS_INTERNAL = True

def get_config():
    
    experiment_cfg = experiment.ExperimentConfig(
        
        # name of the experiment
        experiment_name='test',
        
        # workload details
        seed=42,
        experimental_setup='algoperf',
        workload_name='cifar', 
        full_batch=True,  # whether to do full gradient descent on one batch (fixed during the whole training) or regular minibatch SGD
        num_episodes=3,
        num_iters=400,  # if None, uses default for the workload

        # backend details
        num_batch_devices=8,
        num_opt_devices=1,

        # how often to do things
        eval_every=-1,
        checkpoint_every=-1,

        # other details
        print_with_colors=True,

        # algoperf-only args
        log_every=50,
        use_wandb=False)

    # optimizer_cfg = sgd.SGDConfig(learning_rate_schedule_cfg=schedules.CosineScheduleConfig(0.1, decay_steps=30), momentum=None, nesterov=False, weight_decay=None, grad_clip=None)
    # optimizer_cfg = adamw.AdamWConfig(learning_rate_schedule_cfg=schedules.CosineScheduleConfig(0.1, decay_steps=30), b1=0.9, b2=0.999)
    # optimizer_cfg = polyak.PolyakConfig(f_min=0)
    # optimizer_cfg = dadaptation.DAdaptationConfig(b1=0.9, b2=0.9)
    optimizer_cfg = cocob.COCOBConfig()
    # optimizer_cfg = dowg.DoWGConfig(init_estim_sq_dist=10)
    # optimizer_cfg = mechanic.MechanicConfig(base_optimizer_cfg=optimizer_cfg)
    
    sweep = [(experiment_cfg, optimizer_cfg)]
    if experiment_cfg.experimental_setup == 'algoperf':
        return sweep
    elif experiment_cfg.experimental_setup == 'init2winit':
        assert IS_INTERNAL, 'havent set up init2winit on external yet'
        return [config_utils.convert_configs(experiment_cfg, optimizer_cfg, base_config.get_base_config()) for (experiment_cfg, optimizer_cfg) in sweep]
    else:
        raise NotImplementedError(experiment_cfg.experimental_setup)
