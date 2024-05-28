from meta_opt.train_loops import train_standard_opt, train_hgd, train_meta_opt
from meta_opt.utils.experiment_utils import make, save_checkpoint, process_results, get_final_cparams
from meta_opt import DIR

import dill as pkl
import optax

# ==================================================
# configuration and seeds for each trial
SEEDS = [0, 1, 2, 3, 4, 5, 6]

NAME = 'wmt_stochastic'
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
    'experiment_name': NAME,
    'load_checkpoint': False,
    'overwrite': True,  # whether to allow us to overwrite existing checkpoints or throw errors
    'directory': DIR,
}

def run(seeds, cfg):
    # processed_results = pkl.load(open('{}/data/mnist_fullbatch_processed.pkl'.format(cfg['directory']), 'rb'))
    # initial_cparams = get_final_cparams(processed_results, 'ncf')
    results = make(cfg)
    
    # uncomment the ones to run, with correctly chosen hyperparameters
    for s in seeds:
        CFG['seed'] = s
        print(f'running with seed {s}')
        
        # ours
        results['ours'].append(train_meta_opt(CFG, counterfactual=False, H=16, HH=1, meta_optimizer=optax.inject_hyperparams(optax.sgd)(learning_rate=0), cparams_initial=initial_cparams, initial_lr=1.0))

        # standard benchmarks
        benchmarks = {
            'sgd': optax.inject_hyperparams(optax.sgd)(learning_rate=2.0),
            'momentum': optax.chain(optax.add_decayed_weights(1e-4), optax.inject_hyperparams(optax.sgd)(learning_rate=0.1, momentum=0.9)),
            'adamw': optax.inject_hyperparams(optax.adamw)(learning_rate=1e-3, b1=0.9, b2=0.999, weight_decay=1e-4),
            'dadamw': optax.inject_hyperparams(optax.contrib.dadapt_adamw)(),
            'mechadamw': optax.contrib.mechanize(optax.inject_hyperparams(optax.adamw)(learning_rate=1e-3, b1=0.9, b2=0.999, weight_decay=1e-4)),
        }
        for k, opt in benchmarks.items(): results[k].append(train_standard_opt(CFG, opt))

        # other
        results['hgd'].append(train_hgd(CFG, initial_lr=2.0, hypergrad_lr=1e-4))

        save_checkpoint(CFG, results, checkpoint_name=f'seed {s}')
    processed_results = process_results(CFG, results)
    return processed_results
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