from meta_opt.train_loops import train_standard_opt, train_hgd, train_meta_opt
from meta_opt.utils.experiment_utils import make, save_checkpoint, process_results, bcolors, plot, get_final_cparams
from meta_opt.workloads.wmt import rsqrt
from meta_opt import DIR

import re
import matplotlib.pyplot as plt
import numpy as np
import dill as pkl
import optax

# ==================================================
# configuration and seeds for each trial
SEEDS = [0,]  # the length of this list is the number of trials we will run :)
CFG = {
    # training options
    'workload': 'WMT',
    'num_iters': 96000,
    'eval_every': int(1e9),
    'num_eval_iters': -1,
    'batch_size': 16,
    'full_batch': True,
    'reset_every': 8000,
    
    # wmt options
    'bleu_every': int(1e9),
    'transformer_size': 'base',
    
    # experiment options
    'experiment_name': 'wmt_fullbatch',
    'load_checkpoint': True,
    'overwrite': True,  # whether to allow us to overwrite existing checkpoints or throw errors
    'directory': DIR,
}

def run(seeds, cfg):
    results = make(cfg)
    
    # uncomment the ones to run, with correctly chosen hyperparameters
    for s in seeds:
        CFG['seed'] = s
        print(f'running with seed {s}')
        
        # # ours
        # opt = optax.inject_hyperparams(optax.adam)(learning_rate=2e-4)
        # results['cf_adam_2e-4_clip=1.0'].append(train_meta_opt(CFG, counterfactual=True, H=16, HH=2, meta_optimizer=opt, initial_lr=1.0, grad_clip=1))
        # save_checkpoint(CFG, results, checkpoint_name=f'seed {s}')
        
        # standard benchmarks
        benchmarks = {
            # 'sgd': optax.inject_hyperparams(optax.sgd)(learning_rate=2.0),
            'momentum': optax.chain(optax.add_decayed_weights(0.1), optax.inject_hyperparams(optax.sgd)(learning_rate=0.1, momentum=0.9)),
            'adamw': optax.inject_hyperparams(optax.adamw)(learning_rate=1e-3, b1=0.9, b2=0.999, weight_decay=1e-4),
            'dadamw': optax.inject_hyperparams(optax.contrib.dadapt_adamw)(),
            # 'mechadamw': optax.contrib.mechanize(optax.inject_hyperparams(optax.adamw)(learning_rate=1e-3, b1=0.9, b2=0.999, weight_decay=1e-4)),
            # 'rmsprop': optax.inject_hyperparams(optax.rmsprop)(learning_rate=1e-3),
            # 'rsqrt': rsqrt(lr=0.008, warmup_steps=1000),
        }
        for k, opt in benchmarks.items(): 
            results[k].append(train_standard_opt(CFG, opt))
            save_checkpoint(CFG, results, checkpoint_name=f'seed {s}')

        # other
        # results['hgd'].append(train_hgd(CFG, initial_lr=0.1, hypergrad_lr=1e-2))

        # save_checkpoint(CFG, results, checkpoint_name=f'seed {s}')
    processed_results = process_results(CFG, results)
# ==================================================

import os
if __name__ == '__main__':
    try: 
        idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
        name = CFG['experiment_name']
        CFG['experiment_name'] = f'{name}_{idx}'
        SEEDS = [idx,]  # set seed to the index
    except:
        pass  
    
    run(SEEDS, CFG)