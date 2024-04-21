from meta_opt.train_loops import train_standard_opt, train_hgd, train_meta_opt
from meta_opt.utils.experiment_utils import make, save_checkpoint, process_results, bcolors, plot, get_final_cparams
from meta_opt import DIR

import os
import re
import matplotlib.pyplot as plt
import numpy as np
import dill as pkl
import optax

# ==================================================
# configuration and seeds for each trial
SEEDS = [0, 1, 2]

NAME = 'scale_by_adam_test'
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
        b2s = [0.8, 0.9, 0.99, 0.999]
        for b2 in b2s:
            print(f'RUNNING WITH {b2}')
            opt = optax.inject_hyperparams(optax.adamw)(learning_rate=2e-4)
            results[f'cf_{b2}'].append(train_meta_opt(CFG, counterfactual=True, H=32, HH=2, meta_optimizer=opt, initial_lr=0.1, disturbance_transformation=optax.inject_hyperparams(optax.scale_by_adam)(b1=0, b2=b2)))

        save_checkpoint(CFG, results, checkpoint_name=f'seed {s}')
    processed_results = process_results(CFG, results)
# ==================================================


if __name__ == '__main__':
    try: 
        idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
        name = CFG['experiment_name']
        CFG['experiment_name'] = f'{name}_{idx}'
        SEEDS = [idx,]  # set seed to the index
    except:
        pass  
    
    run(SEEDS, CFG)
