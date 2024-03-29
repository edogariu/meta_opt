import os
from time import perf_counter
from copy import deepcopy
from collections import defaultdict
import tqdm
import re
import functools
import matplotlib.pyplot as plt
import numpy as np
import dill as pkl
import jax
import jax.numpy as jnp
import optax

from meta_opt.train_loops import train_standard_opt, train_hgd, train_meta_opt
from meta_opt.utils.experiment_utils import make, save_checkpoint, process_results, bcolors, plot, get_final_cparams
from meta_opt.nn import reset_model, train_step, eval
from meta_opt import DIR
from meta_opt.workloads import get_workload
from meta_opt.workloads.wmt import rsqrt
from meta_opt.utils.pytree_utils import pytree_sq_norm, pytree_proj, append
from meta_opt.utils.experiment_utils import get_opt_hyperparams, save_checkpoint

CFG = {
    # training options
    'workload': 'MNIST',
    'num_iters': 5000,
    'eval_every': int(1e9),
    'num_eval_iters': -1,
    'batch_size': 32,
    'full_batch': True,
    'reset_every': int(1e9),
    'model_size': None,

    # experiment options
    'seed': None,
    'experiment_name': None,
    'load_checkpoint': False,
    'overwrite': True,  # whether to allow us to overwrite existing checkpoints or throw errors
    'directory': DIR,
} 


# @jax.jit
def forward_and_backward_with_hessian(tstate, batch):
    if tstate.rng is not None:
        next_key, dropout_key = jax.random.split(tstate.rng)
        tstate = tstate.replace(rng=next_key)
    else: dropout_key = None
    def loss_fn(params):
        yhat = tstate.apply_fn({'params': params}, batch['x'], train=True, rngs={'dropout': dropout_key})
        loss = tstate.loss_fn(yhat, batch['y'])
        return loss
    loss, grads = jax.value_and_grad(loss_fn)(tstate.params)

    p, td = jax.tree_util.tree_flatten(tstate.params)
    def loss_fn_from_flat(params_flat):  # for hessian computation
        q = []
        n = 0
        for v in p:
            d = np.prod(v.shape)
            q.append(params_flat[n: n + d].reshape(v.shape))
            n += d
        params = jax.tree_util.tree_unflatten(td, q)
        return loss_fn(params)
        
    hessians = jax.hessian(loss_fn_from_flat)(jnp.concatenate([_p.reshape(-1) for _p in p], axis=0))
    return tstate, (loss, grads, hessians)

# @jax.jit
def sequential_stability(tstate, batch, carry, delta):
    # the vanilla stuff, but also computing hessian
    stats = {}
    tstate, (loss, grads, hessians) = forward_and_backward_with_hessian(tstate, batch)
    tstate = tstate.apply_gradients(grads=grads)

    # use hessian to compute transition matrix and append to the buffer. note that this is using batch averages
    def f(H, eta, d, carry):
        I = jnp.eye(H.shape[0])
        A = jnp.block([[(1 - d) * I, 0 * I, -eta * I], [I, 0 * I, 0 * I], [H, -H, 0 * I]])  # transition matrix for this step
        carry = A @ append(carry, jnp.eye(A.shape[0]))  # append an entry of 1 to the right, then left multiply each entry by A. this dynamically handles the cumprod
        spectral_norms = jnp.linalg.norm(carry, axis=(1, 2), ord=2)
        return carry, spectral_norms
    
    H = hessians # + 2 * beta * jnp.eye(hessians.shape[0])  # TODO CHECK THIS!!!
    carry, spectral_norms = f(H, tstate.opt_state.hyperparams['learning_rate'], delta, carry)
    # print(carry.shape, spectral_norms.shape, spectral_norms)
    stats['sequential_stability'] = spectral_norms

    return tstate, (loss, grads, stats, carry)

def run_experiment(seed, name, opt, exp_fn, max_len, model_size, train_idxs, duration):
    cfg = deepcopy(CFG)
    cfg['model_size'] = model_size
    cfg['seed'] = seed
    cfg['experiment_name'] = name
    tstate, train_ds, test_ds, rng, args = get_workload(cfg, opt)

    stats = defaultdict(dict)
    args['optimizer_args'] = get_opt_hyperparams(tstate.opt_state)
    args['optimizer_name'] = 'standard'
    stats['args'] = args

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(tstate.params))    
    carry = jnp.zeros((max_len, param_count * 3, param_count * 3))
    t0 = perf_counter()
    last_eval_step = None
    pbar = tqdm.tqdm(train_ds.as_numpy_iterator(), total=args['num_iters'])
    b = -1
    for t, batch in enumerate(pbar):

        if t in train_idxs: 
            b = t
            print('set b to', b)

        if t <= b + duration:
            tstate, (loss, grads, s, carry) = exp_fn(tstate, batch, carry)
        else:
            tstate, (loss, grads) = train_step(tstate, batch)
            s = {}
        
        # update all the stats
        s['timestamp'] = perf_counter() - t0
        s['loss'] = loss
        if t % args['eval_every'] == 0 and t != 0:
            for k, v in eval(tstate, test_ds.as_numpy_iterator()).items(): s[f'eval_{k}'] = v
            s['param_sq_norm'] = pytree_sq_norm(tstate.params)
            s['grad_sq_norm'] = pytree_sq_norm(grads)
            if hasattr(tstate.model, 'radius'):
                proj_grads = pytree_proj(grads, tstate.params)
                s['proj_grad_sq_norm'] = pytree_sq_norm(proj_grads)
            last_eval_step = t
        if 'bleu_every' in args and t % args['bleu_every'] == 0 and t != 0:
            s['bleu'], s['bleu_exemplars'] = tstate.model.bleu(tstate, test_ds.as_numpy_iterator())
            print(s['bleu'], s['bleu_exemplars'])
        if hasattr(tstate.opt_state, 'hyperparams'): s['lr'] = float(tstate.opt_state.hyperparams['learning_rate'])
        else: s['lr'] = 0.
        
        stats[t] = s
        pbar.set_postfix({'loss': round(s['loss'].item(), 3), 
                          'eval_loss': round(stats[last_eval_step]['eval_loss'].item(), 3) if last_eval_step is not None else 'N/A',
                          'lr': round(s['lr'], 5)
                          })
        if t % args['reset_every'] == 0:
            reset_rng, rng = jax.random.split(rng)
            tstate = reset_model(reset_rng, tstate)
            del reset_rng
    return dict(stats)

if __name__ == '__main__':
    
    MAX_LEN = 30  # for computational reasons, we will only compute with lengths up to this value
    DELTA = 0.05  # (1-delta) decay factor for state
    NAME = 'seq_stab'
    MODEL_SIZE = [28 * 28, 4, 10]
    TRAIN_IDXS = [0, 1000, 2500]
    DURATION = 90
    SEED = 3

    results = run_experiment(SEED, 
                            NAME, 
                            optax.inject_hyperparams(optax.sgd)(0.3), 
                            functools.partial(sequential_stability, delta=DELTA), 
                            MAX_LEN, MODEL_SIZE, TRAIN_IDXS, DURATION)
    cfg = deepcopy(CFG)
    cfg['model_size'] = MODEL_SIZE
    cfg['seed'] = SEED
    cfg['experiment_name'] = NAME
    save_checkpoint(cfg, results, 'sequential stability')
    