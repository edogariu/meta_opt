from meta_opt.train_loops import train_standard_opt, train_hgd, train_meta_opt
from meta_opt.utils.experiment_utils import make, save_checkpoint, process_results, bcolors, plot, get_final_cparams
from meta_opt import DIR

import re
import matplotlib.pyplot as plt
import numpy as np
import dill as pkl
import optax

# ==================================================
# configuration and seeds for each trial
SEEDS = [0, 1, 2, 3, 4, 5]  # the length of this list is the number of trials we will run :)
CFG = {
    # training options
    'workload': 'MNIST',
    'num_iters': 20000,
    'eval_every': 100,
    'num_eval_iters': -1,
    'batch_size': 512,
    'full_batch': False,
    'reset_every': 4000,

    # experiment options
    'experiment_name': 'mnist_baselines',
    'load_checkpoint': False,
    'overwrite': True,  # whether to allow us to overwrite existing checkpoints or throw errors
    'directory': DIR,
}

def run(seeds, cfg):
    results = make(cfg)
    
    # uncomment the ones to run, with correctly chosen hyperparameters
    for s in seeds:
        CFG['seed'] = s
        print(f'running with seed {s}')
        
        # standard benchmarks
        benchmarks = {
            'sgd': optax.inject_hyperparams(optax.sgd)(learning_rate=0.4),
            'momentum': optax.chain(optax.add_decayed_weights(1e-4), optax.inject_hyperparams(optax.sgd)(learning_rate=0.1, momentum=0.9)),
            'adamw': optax.inject_hyperparams(optax.adamw)(learning_rate=1e-3, b1=0.9, b2=0.999, weight_decay=1e-4),
            'dadamw': optax.inject_hyperparams(optax.contrib.dadapt_adamw)(),
            'mechadamw': optax.contrib.mechanize(optax.inject_hyperparams(optax.adamw)(learning_rate=1e-3, b1=0.9, b2=0.999, weight_decay=1e-4)),
            # 'rmsprop': optax.inject_hyperparams(optax.rmsprop)(learning_rate=1e-3),
        }
        for k, opt in benchmarks.items(): results[k].append(train_standard_opt(CFG, opt))

        # other
        results['hgd'].append(train_hgd(CFG, initial_lr=0.1, hypergrad_lr=1e-2))

        save_checkpoint(CFG, results, checkpoint_name=f'seed {s}')
    processed_results = process_results(CFG, results)
# ==================================================


if __name__ == '__main__':
    run(SEEDS, CFG)
