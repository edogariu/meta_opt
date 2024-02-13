from time import perf_counter
from typing import Tuple
from collections import defaultdict
from copy import deepcopy
import tqdm
import dill as pkl
import os

import numpy as np
import tensorflow as tf; tf.config.experimental.set_visible_devices([], "GPU")
import jax
import jax.numpy as jnp
import optax

from meta_opt.nn.trainer import create_train_state, reset_model, train_step, eval
from meta_opt.problems import mnist, cifar10#, wmt  # WMT is a bit broken atm
from meta_opt.meta_opt import MetaOpt
# from meta_opt.gaps import MetaOptGAPS  # we arent running GAPS experiments atm

"""
All the utilities and such that are necessary to run our experiments, but aren't really an integral part of the `meta_opt` package itself.

- To add a workload, add an extra entry in the `_get_problem()` function
    - current list: ['MNIST', 'CIFAR',]
- To add an optimization algorithm, create a new `train_*()` function like the ones below
    - current list: any standard optax optimizer, meta opt, hypergradient decent
"""

def _get_problem(cfg, optimizer):
    rng, cfg['seed'] = _set_seed(cfg['seed'])
    init_rng, rng = jax.random.split(rng)
    directory = cfg['directory']

    # get dataset and model
    if cfg['workload'] == 'MNIST':
        train_ds, test_ds, example_input, loss_fn, metric_fns = mnist.load_mnist(cfg, dataset_dir=os.path.join(directory, 'datasets'))
        model = mnist.MLP([28 * 28, 100, 100, 10])
    elif cfg['workload'] == 'CIFAR':
        train_ds, test_ds, example_input, loss_fn, metric_fns = cifar10.load_cifar10(cfg, dataset_dir=os.path.join(directory, 'datasets'))
        model = cifar10.VGG16()
    # elif cfg['dataset'] == 'WMT':  # WMT is a bit broken atm
    #     train_ds, test_ds, example_input, loss_fn, metric_fns, tokenizer = wmt.load_wmt(cfg, dataset_dir=os.path.join(directory, 'datasets'))
    #     train_ds.cache()
    #     model = wmt.make_transformer(num_heads=8, num_layers=6, emb_dim=512, qkv_dim=512, mlp_dim=2048)
    #     # model = wmt.make_transformer(num_heads=4, num_layers=3, emb_dim=64, qkv_dim=64, mlp_dim=256)
    #     raise NotImplementedError('gotta figure out how to keep the tokenizer around')
    else:
        raise NotImplementedError(cfg['workload'])

    tstate = create_train_state(init_rng, model, example_input, optimizer, loss_fn, metric_fns=metric_fns)
    del init_rng

    args = deepcopy(cfg)
    args.update({'model': str(model), 
                 'num_params': sum(x.size for x in jax.tree_util.tree_leaves(tstate.params))})
    print(args['num_params'], 'params in the model!')

    return tstate, train_ds, test_ds, rng, args

# -------------------------------------------------------------------------------------------------
# ------------------------------ Standard Optax Optimizers ----------------------------------------
# -------------------------------------------------------------------------------------------------

def train_standard_opt(cfg, optimizer):
    tstate, train_ds, test_ds, rng, args = _get_problem(cfg, optimizer)

    stats = defaultdict(dict)
    args['optimizer_args'] = _get_opt_hyperparams(tstate.opt_state)
    args['optimizer_name'] = 'standard'
    stats['args'] = args

    t0 = perf_counter()
    last_eval_step = None
    for t, batch in enumerate(pbar := tqdm.tqdm(train_ds.as_numpy_iterator(), total=args['num_iters'])):

        if t % args['reset_every'] == 0:
            reset_rng, rng = jax.random.split(rng)
            tstate = reset_model(reset_rng, tstate)
            del reset_rng

        tstate, (loss, grads) = train_step(tstate, batch)

        # update all the stats
        s = {}
        s['timestamp'] = perf_counter() - t0
        s['loss'] = loss
        if t % args['eval_every'] == 0:
            for k, v in eval(tstate, test_ds.as_numpy_iterator()).items(): s[f'eval_{k}'] = v
            s['param_sq_norm'] = sum(jax.tree_util.tree_flatten(jax.tree_map(lambda p: (p * p).sum(), tstate.params))[0])
            s['grad_sq_norm'] = sum(jax.tree_util.tree_flatten(jax.tree_map(lambda g: (g * g).sum(), grads))[0])
            last_eval_step = t
            
        stats[t] = s
        pbar.set_postfix({'loss': round(s['loss'].item(), 3), 
                          'eval_loss': round(stats[last_eval_step]['eval_loss'].item(), 3) if last_eval_step is not None else 'N/A',
                          })

    # stats['tstate'] = deepcopy(tstate)
    return dict(stats)


# -------------------------------------------------------------------------------------------------
# ----------------------------------- Our Meta Optimizer ------------------------------------------
# -------------------------------------------------------------------------------------------------

def train_meta_opt(cfg, 
                   counterfactual: bool, 
                   meta_optimizer, 
                   H: int, HH: int, 
                   m_method: str = 'scalar', 
                   initial_lr: float = 1e-4, ema_keys = [], grad_clip = 10): 
    
    """
    note that if we aren't counterfactual, we have to rescale the number of iterations by HH to account for taking HH training steps every noncounterfactual meta step
    """
    cfg = deepcopy(cfg)
    
    def check(t, k):  # to check conditions that happen every `n` steps, since `t` will increment by 1 if counterfactual and by `HH` otherwise
        n = cfg[k]
        if counterfactual: return t % n == 0
        else: return t % n < HH
    
    optimizer = optax.chain(optax.add_decayed_weights(1e-5), optax.sgd(learning_rate=initial_lr))
    tstate, train_ds, test_ds, rng, args = _get_problem(dict(cfg, **({'num_iters': cfg['num_iters'] // HH} if not counterfactual else {})), optimizer)
    meta_opt = MetaOpt(tstate, H=H, HH=HH, m_method=m_method, meta_optimizer=meta_optimizer, ema_keys=ema_keys, grad_clip=grad_clip)

    stats = defaultdict(dict)
    if not counterfactual: args['num_iters'] *= HH
    args['optimizer_name'] = 'meta'
    args['optimizer_args'] = {'initial_lr': initial_lr,
                              'm_method': m_method,
                              'meta_optimizer_args': _get_opt_hyperparams(meta_opt.cstate.opt_state),
                              'H': H,
                              'HH': HH,
                              'ema_keys': ema_keys,
                              'grad_clip': grad_clip,
                              }
    stats['args'] = args

    t0 = perf_counter()
    last_eval_step = None
    for t, batch in enumerate(pbar := tqdm.tqdm(train_ds.as_numpy_iterator(), total=cfg['num_iters'])):
        if not counterfactual: t *= HH

        if check(t, 'reset_every'):
            reset_rng, rng = jax.random.split(rng)
            tstate = reset_model(reset_rng, tstate)
            meta_opt = meta_opt.episode_reset()
            del reset_rng

        if counterfactual:
            tstate, (loss, grads) = train_step(tstate, batch)
            tstate = meta_opt.counterfactual_step(tstate, grads, batch)
        else:
            tstate, (loss, grads) = meta_opt.noncounterfactual_step(tstate, batch)

        # update all the stats
        s = {}
        s['timestamp'] = perf_counter() - t0
        s['loss'] = loss
        if check(t, 'eval_every'):
            for k, v in eval(tstate, test_ds.as_numpy_iterator()).items(): s[f'eval_{k}'] = v
            s['param_sq_norm'] = sum(jax.tree_util.tree_flatten(jax.tree_map(lambda p: (p * p).sum(), tstate.params))[0])
            s['grad_sq_norm'] = sum(jax.tree_util.tree_flatten(jax.tree_map(lambda g: (g * g).sum(), grads))[0])
            last_eval_step = t

        # log the value of the Ms
        if m_method == 'scalar':
            s['M'] = meta_opt.cstate.cparams['M'].reshape(-1)
            for k, v in meta_opt.cstate.cparams['M_ema'].items(): s[f'M_ema_{k}'] = v
        else:
            s['M'] = jnp.stack([m.reshape((m.shape[0], -1)).mean(axis=-1) for m in jax.tree_util.tree_leaves(meta_opt.cstate.cparams['M'])], axis=0).mean(axis=0)
            for k, v in meta_opt.cstate.cparams['M_ema'].items(): s[f'M_ema_{k}'] = jnp.stack([m.mean() for m in jax.tree_util.tree_leaves(v)], axis=0).mean(axis=0)
        stats[t] = s
        pbar.set_postfix({'loss': round(s['loss'].item(), 3), 
                          'eval_loss': round(stats[last_eval_step]['eval_loss'].item(), 3) if last_eval_step is not None else 'N/A',
                          'M': s['M'].sum()})
        if not counterfactual: pbar.update(HH)

    # stats['tstate'] = deepcopy(tstate)
    # stats['cstate'] = deepcopy(meta_opt.cstate)
    return dict(stats)

# -------------------------------------------------------------------------------------------------
# ----------------------------------- Hypergradient Descent ---------------------------------------
# -------------------------------------------------------------------------------------------------

def train_hgd(cfg, initial_lr: float, hypergrad_lr: float):

    optimizer = optax.inject_hyperparams(optax.sgd)(learning_rate=initial_lr)
    tstate, train_ds, test_ds, rng, args = _get_problem(cfg, optimizer)

    stats = defaultdict(dict)
    args['optimizer_name'] = 'hgd'
    args['optimizer_args'] = {'initial_lr': initial_lr,
                              'hypergrad_lr': hypergrad_lr,
                              }
    stats['args'] = args

    prev_grads = None
    t0 = perf_counter()
    last_eval_step = None
    for t, batch in enumerate(pbar := tqdm.tqdm(train_ds.as_numpy_iterator(), total=args['num_iters'])):

        if t % args['reset_every'] == 0:
            reset_rng, rng = jax.random.split(rng)
            tstate = reset_model(reset_rng, tstate)
            del reset_rng

        tstate, (loss, grads) = train_step(tstate, batch)
        if prev_grads is not None:
            hypergrad = -sum([(g1 * g2).sum() for g1, g2 in zip(jax.tree_util.tree_leaves(grads), jax.tree_util.tree_leaves(prev_grads))])
            tstate.opt_state.hyperparams['learning_rate'] -= hypergrad_lr * hypergrad
        else: hypergrad = 0.
        prev_grads = grads

        # update all the stats
        s = {}
        s['timestamp'] = perf_counter() - t0
        s['loss'] = loss
        if t % args['eval_every'] == 0:
            for k, v in eval(tstate, test_ds.as_numpy_iterator()).items(): s[f'eval_{k}'] = v
            s['param_sq_norm'] = sum(jax.tree_util.tree_flatten(jax.tree_map(lambda p: (p * p).sum(), tstate.params))[0])
            s['grad_sq_norm'] = sum(jax.tree_util.tree_flatten(jax.tree_map(lambda g: (g * g).sum(), grads))[0])
            last_eval_step = t
        s['hypergrad'] = hypergrad
        s['lr'] = float(tstate.opt_state.hyperparams['learning_rate'])
        stats[t] = s
        pbar.set_postfix({'loss': round(s['loss'].item(), 3), 
                          'eval_loss': round(stats[last_eval_step]['eval_loss'].item(), 3) if last_eval_step is not None else 'N/A',
                          'lr': s['lr']})

    # stats['tstate'] = deepcopy(tstate)
    return dict(stats)



# some utilities
def _set_seed(seed):
    if seed is None:
        seed = np.random.randint()
        print('seed set to {}'.format(seed))
    np.random.seed(seed)
    tf.random.set_seed(seed)
    rng = jax.random.PRNGKey(seed)
    return rng, seed

def _get_opt_hyperparams(opt_state):
    """
    helper fn to serialize optax optimizer hyperparameters from the opt_state
    """
    if isinstance(opt_state, Tuple): h = [deepcopy(o.hyperparams) for o in opt_state if hasattr(o, 'hyperparams')]
    else: h = deepcopy(opt_state.hyperparams)
    return h


class bcolors:  # for printing pretty colors :)
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_stuff_and_load_checkpoint(cfg):
    name, directory, load_experiment_checkpoint = cfg['experiment_name'], cfg['directory'], cfg['load_checkpoint']
    print(f'using {bcolors.WARNING}{bcolors.BOLD}{jax.lib.xla_bridge.get_backend().platform}{bcolors.ENDC} for jax')
    print(f'results will be stored at: {bcolors.OKCYAN}{bcolors.BOLD}{directory}/data/{name}_*.pkl{bcolors.ENDC}')
    print('we will {}try to load experiment checkpoint first'.format('' if load_experiment_checkpoint else f'{bcolors.FAIL}{bcolors.BOLD}NOT{bcolors.ENDC} '))
    
    # load checkpoints
    filename = '{}/data/{}_raw.pkl'.format(directory, cfg['experiment_name'])
    exists = os.path.isfile(filename)
    if load_experiment_checkpoint:
        if not exists: 
            print(f'\t{bcolors.FAIL}{bcolors.BOLD}checkpoint could not be found!{bcolors.ENDC}')
            load_experiment_checkpoint = False
        else:
            results = pkl.load(open(filename, 'rb'))
            ret = pkl.load(open('{}/data/{}_processed.pkl'.format(directory, cfg['experiment_name']), 'rb'))
            print(f'\t{bcolors.OKGREEN}{bcolors.BOLD}loaded checkpoint from {filename}, containing {list(results.keys())}{bcolors.ENDC}')
    if not load_experiment_checkpoint:
        results = defaultdict(list)
        if exists:
            print(f'{bcolors.FAIL}{bcolors.BOLD}WARNING: there already exists a checkpoint with this name! make sure you want to overwrite{bcolors.ENDC}')
            assert cfg['overwrite'], 'cannot start from scratch with an existing checkpoint and `overwrite=False`'
        print('starting the experiment from scratch :)')
    return results

def process_results(cfg, results):
    # clean the stats
    to_del = []
    for k, v in results.items(): 
        if len(v) == 0: to_del.append(k)
    for k in to_del: del results[k]

    # gather stats
    aggregated = {}  # experiment name -> 'args' or timestamp -> stat key -> stat value
    for k, v in results.items():  # for each experiment
        aggregated[k] = {'args': []}
        for n in range(len(v)):  # for each trial
            aggregated[k]['args'].append(v[n]['args'])
            for t in range(v[0]['args']['num_iters']):  # for each timestamp
                if t not in v[n]: continue
                for stat_key, value in v[n][t].items():  # for each stat recorded at that timestamp
                    if stat_key not in aggregated[k]: aggregated[k][stat_key] = {}
                    if t not in aggregated[k][stat_key]: aggregated[k][stat_key][t] = []
                    aggregated[k][stat_key][t].append(value)

    # aggregate stats
    ret = defaultdict(dict)  # stat key -> experiment name -> 't' or 'avg' or 'std' ->
    args = {}
    for k, v in aggregated.items():  # for experiment
        for stat_key in v.keys():  # for stat
            if stat_key in ['args', 'bleu']:
                args[k] = v[stat_key]
                continue
            if k not in ret[stat_key]: ret[stat_key][k] = {}
            ret[stat_key][k]['t'] = list(v[stat_key].keys())
            arr = np.array(list(v[stat_key].values()))
            ret[stat_key][k]['avg'] = np.mean(arr, axis=1)
            ret[stat_key][k]['std'] = np.std(arr, axis=1)

    filename = '{}/data/{}_processed.pkl'.format(cfg['directory'], cfg['experiment_name'])
    assert cfg['overwrite'] or not os.path.isfile(filename), 'cannot save processed results with existing processed results and `overwrite=False`'
    with open(filename, 'wb') as f: pkl.dump(ret, f)
    print(f'{bcolors.OKGREEN}{bcolors.BOLD}Saved processed results to {filename}{bcolors.ENDC}')
    return ret
