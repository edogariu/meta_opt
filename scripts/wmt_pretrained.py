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
SEEDS = [10,]  # the length of this list is the number of trials we will run :)
CFG = {
    # training options
    'workload': 'WMT',
    'num_iters': 100000,
    'eval_every': 1000,
    'num_eval_iters': 20,
    'batch_size': 16,
    'full_batch': False,
    'reset_every': int(1e9),
    
    # wmt options
    'bleu_every': 5000,
    'transformer_size': 'base',
    
    # experiment options
    'experiment_name': 'wmt_pretrained',
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
        
        INITIAL_PARAMS_EXPERIMENT_NAME = 'wmt_fullbatch_clip'
        INITIAL_PARAMS_RUN_NAME = 'cf_adam_1e-3_clip=1.0'
        INITIAL_PARAMS_TIMESTEP = -1
        
        processed_results = pkl.load(open('{}/data/{}_processed.pkl'.format(cfg['directory'], INITIAL_PARAMS_EXPERIMENT_NAME), 'rb'))
        initial_cparams = get_final_cparams(processed_results, INITIAL_PARAMS_RUN_NAME, idx=INITIAL_PARAMS_TIMESTEP)
        name = '{}/{}[{}]'.format(INITIAL_PARAMS_EXPERIMENT_NAME, INITIAL_PARAMS_RUN_NAME, INITIAL_PARAMS_TIMESTEP)
        
        results = make(cfg)
        
        opt = optax.inject_hyperparams(optax.sgd)(learning_rate=0)
        results[f'frozen_{name}'].append(train_meta_opt(CFG, counterfactual=False, H=16, HH=1, meta_optimizer=opt, initial_lr=1.0, cparams_initial=initial_cparams, grad_clip=0.1))
        save_checkpoint(CFG, results, checkpoint_name=f'seed {s}')
        

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