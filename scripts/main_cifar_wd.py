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

# ==================================================
# configuration and seeds for each trial
SEEDS = [0, 1, 2]

NAME = 'cifar_wd'
CFG = {
    # training options
    'workload': 'CIFAR',
    'num_iters': 20000,
    'eval_every': 200,
    'num_eval_iters': -1,
    'batch_size': 512,
    'full_batch': False,
    'reset_every': 10000,

    # experiment options
    'experiment_name': NAME,
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
        
        # ours
        wds = [1e-4, 1e-3, 1e-2]
        for w in wds:
            opt = optax.inject_hyperparams(optax.sgd)(learning_rate=2e-4)
            opt = optax.chain(optax.add_decayed_weights(w), opt)
            results[f'cf_{w}'].append(train_meta_opt(CFG, counterfactual=True, H=32, HH=2, meta_optimizer=opt, initial_lr=0.1))
            results[f'ncf_{w}'].append(train_meta_opt(CFG, counterfactual=False, H=32, HH=2, meta_optimizer=opt, initial_lr=0.1))

        save_checkpoint(CFG, results, checkpoint_name=f'seed {s}')
    processed_results = process_results(CFG, results)
# ==================================================


if __name__ == '__main__':
    run(SEEDS, CFG)
