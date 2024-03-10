"""
FOR SLURM SCRIPTS
"""

# handle the system stuff, colab stuff, etc
import os
DIR = os.path.abspath(".")

# make sure we have the necessary folders
for subdir in ['data', 'figs', 'datasets']: 
    temp = os.path.join(DIR, subdir)
    if not os.path.isdir(temp): os.mkdir(temp)

from meta_opt.train_loops import train_standard_opt, train_hgd, train_meta_opt
from meta_opt.utils.experiment_utils import make, save_checkpoint, process_results, bcolors, plot, get_final_cparams
import meta_opt.configs as configs

import re
import matplotlib.pyplot as plt
import numpy as np
import dill as pkl
import optax

# configuration and seeds for each trial
SEEDS = range(50)
NAME = 'cifar_fullbatch'
CFG = {
    # training options
    'workload': 'CIFAR',
    'num_iters': 10000,
    'eval_every': -1,
    'num_eval_iters': -1,
    'batch_size': 256,
    'full_batch': True,
    'reset_every': 1000,

    # experiment options
    'experiment_name': NAME,
    'load_checkpoint': False,
    'overwrite': True,  # whether to allow us to overwrite existing checkpoints or throw errors
    'directory': DIR,
}


if __name__ == '__main__':
    idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    CFG['experiment_name'] = f'{NAME}_{idx}'
    results = make(CFG)
    
    # uncomment the ones to run, with correctly chosen hyperparameters
    s = SEEDS[idx]
    CFG['seed'] = s
    
    # ours
    opt = optax.inject_hyperparams(optax.adam)(learning_rate=4e-4, b1=0.9, b2=0.999)
    results['cf'].append(train_meta_opt(CFG, counterfactual=True, H=32, HH=2, meta_optimizer=opt, initial_lr=0.1))
    results['cf_3'].append(train_meta_opt(CFG, counterfactual=True, H=32, HH=3, meta_optimizer=opt, initial_lr=0.1))
    results['ncf'].append(train_meta_opt(CFG, counterfactual=False, H=32, HH=2, meta_optimizer=opt, initial_lr=0.1))

    # standard benchmarks
    benchmarks = {
        'sgd': optax.inject_hyperparams(optax.sgd)(learning_rate=0.2),
        'momentum': optax.chain(optax.add_decayed_weights(1e-4), optax.inject_hyperparams(optax.sgd)(learning_rate=0.1, momentum=0.9)),
        'adamw': optax.inject_hyperparams(optax.adamw)(learning_rate=1e-3, b1=0.9, b2=0.999, weight_decay=1e-4),
        # 'rmsprop': optax.inject_hyperparams(optax.rmsprop)(learning_rate=1e-3),
    }
    for k, opt in benchmarks.items(): results[k].append(train_standard_opt(CFG, opt))

    # other
    results['hgd'].append(train_hgd(CFG, initial_lr=0.1, hypergrad_lr=1e-3))

    save_checkpoint(CFG, results, checkpoint_name=f'seed {s}')
    processed_results = process_results(CFG, results)
