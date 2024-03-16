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
    'num_iters': 100000,
    'eval_every': 1000,
    'num_eval_iters': 20,
    'batch_size': 16,
    'full_batch': False,
    'reset_every': int(1e9),
    
    # wmt options
    'bleu_every': 5000,
    'transformer_size': 'miniscule',
    
    # experiment options
    'experiment_name': 'wmt_pretrained_miniscule',
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
        
        # ours
        processed_results = pkl.load(open('{}/data/wmt_fullbatch_miniscule_long_processed.pkl'.format(cfg['directory']), 'rb'))
        initial_cparams = get_final_cparams(processed_results, 'ncf_adam')
        results = make(cfg)
        opt = optax.inject_hyperparams(optax.sgd)(learning_rate=0)
        results['ncf_adam_frozen'].append(train_meta_opt(CFG, counterfactual=False, H=32, HH=1, meta_optimizer=opt, initial_lr=0.1, cparams_initial=initial_cparams))
        opt = optax.inject_hyperparams(optax.sgd)(learning_rate=1e-4)
        results['ncf_adam_ncf'].append(train_meta_opt(CFG, counterfactual=False, H=32, HH=3, meta_optimizer=opt, initial_lr=0.1, cparams_initial=initial_cparams))
        
        # standard benchmarks
        benchmarks = {
            'sgd': optax.inject_hyperparams(optax.sgd)(learning_rate=1.4),
            # 'momentum': optax.chain(optax.add_decayed_weights(1e-4), optax.inject_hyperparams(optax.sgd)(learning_rate=0.1, momentum=0.9)),
            'adamw': optax.inject_hyperparams(optax.adamw)(learning_rate=4e-3, b1=0.9, b2=0.999, weight_decay=1e-4),
            # 'dadamw': optax.inject_hyperparams(optax.contrib.dadapt_adamw)(),
            # 'mechadamw': optax.contrib.mechanize(optax.inject_hyperparams(optax.adamw)(learning_rate=1e-3, b1=0.9, b2=0.999, weight_decay=1e-4)),
            # 'rmsprop': optax.inject_hyperparams(optax.rmsprop)(learning_rate=1e-3),
            'rsqrt': rsqrt(lr=0.004, warmup_steps=4000),
        }
        for k, opt in benchmarks.items(): results[k].append(train_standard_opt(CFG, opt))

        # other
        # results['hgd'].append(train_hgd(CFG, initial_lr=0.1, hypergrad_lr=1e-2))

        save_checkpoint(CFG, results, checkpoint_name=f'seed {s}')
    processed_results = process_results(CFG, results)
# ==================================================


if __name__ == '__main__':
    run(SEEDS, CFG)
