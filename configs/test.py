from absl import logging

from configs.experiment import ExperimentConfig
from configs.optimizers import SGDConfig, AdamWConfig, MetaOptConfig
from meta_opt.utils import bcolors

def get_configs():
    
    experiment_cfg = ExperimentConfig(
        
        # name of the experiment
        experiment_name='test',
        
        # workload details
        workload_name='mnist', 
        full_batch=True,  # whether to do full gradient descent on one batch (fixed during the whole training) or regular minibatch SGD
        num_episodes=1,

        framework='jax',
        num_iters=None,  # if None, uses default for the workload
        batch_size=None,  # if None, uses default for the workload

        # how often to do things
        eval_every=-1,
        checkpoint_every=-1,
        print_every=-1,
        log_every=5,

        # other details
        profile=True,
        use_wandb=False,
        resume_last_run=False,
        overwrite=True)

    # optimizer_cfg = SGDConfig(learning_rate=0.01, momentum=0.9, nesterov=False, weight_decay=None, grad_clip=None)
    # optimizer_cfg = AdamWConfig(learning_rate=0.001, b1=0.9, b2=0.999, eps=1e-8, weight_decay=None, grad_clip=None)

    # meta_optimizer_cfg = SGDConfig(learning_rate=1e-5, momentum=0, nesterov=False, weight_decay=None, grad_clip=None)
    meta_optimizer_cfg = AdamWConfig(learning_rate=1e-3, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0, grad_clip=None)
    optimizer_cfg = MetaOptConfig(initial_learning_rate=0.1, weight_decay=1e-4, grad_clip=None,
                                H=16, HH=2, m_method='scalar', scale_by_adam_betas=None, fake_the_dynamics=False, freeze_meta_params=False,
                                meta_optimizer_cfg=meta_optimizer_cfg, meta_grad_clip=10.0)

    # meta_optimizer_cfg = SGDConfig(learning_rate=1e-5, momentum=0, nesterov=False, weight_decay=None, grad_clip=None)
    # optimizer_cfg = MetaOptConfig(initial_learning_rate=0.001, weight_decay=1e-4, grad_clip=None,
    #                               H=16, HH=2, m_method='diagonal', scale_by_adam_betas=(0.9, 0.999),
    #                               meta_optimizer_cfg=meta_optimizer_cfg, meta_grad_clip=10.0)
        
    return experiment_cfg, optimizer_cfg
