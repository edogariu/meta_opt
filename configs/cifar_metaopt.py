from meta_opt.experiment import ExperimentConfig
from meta_opt.optimizers.sgd import SGDConfig
from meta_opt.optimizers.adamw import AdamWConfig
from meta_opt.optimizers.metaopt import MetaOptConfig

# requires running
#       tfds.load('wmt17_translate/de-en', download=True, data_dir='./datasets')
#       tfds.load('wmt14_translate/de-en', download=True, data_dir='./datasets')
# beforehand so that the dataset is downloaded. 
# also needs the sentencepiece tokenizer, see https://github.com/mlcommons/algorithmic-efficiency/blob/main/datasets/dataset_setup.py#L684 

def get_configs():
    
    experiment_cfg = ExperimentConfig(
        
        # name of the experiment
        experiment_name='cifar_metaopt',
        
        # workload details
        seed=0,
        workload_name='cifar', 
        full_batch=False,  # whether to do full gradient descent on one batch (fixed during the whole training) or regular minibatch SGD
        num_episodes=1,

        framework='jax',
        num_iters=None,  # if None, uses default for the workload

        # how often to do things
        eval_every=1000,
        checkpoint_every=-1,
        log_every=50,

        # other details
        use_wandb=True,
        print_with_colors=True)

    meta_optimizer_cfg = SGDConfig(learning_rate=1e-5, momentum=0, nesterov=False, weight_decay=None, grad_clip=None)
    # meta_optimizer_cfg = AdamWConfig(learning_rate=1e-3, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0, grad_clip=None)
    optimizer_cfg = MetaOptConfig(initial_learning_rate=0.1, weight_decay=1e-4, grad_clip=None,
                                H=16, HH=2, m_method='scalar', scale_by_adam_betas=None, fake_the_dynamics=False, freeze_gpc_params=False,
                                freeze_cost_fn_during_rollouts=False,
                                meta_optimizer_cfg=meta_optimizer_cfg)

    return experiment_cfg, optimizer_cfg
