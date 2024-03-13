from time import perf_counter
from collections import defaultdict
import tqdm

import jax
import optax

from meta_opt.nn import reset_model, train_step, eval
from meta_opt.workloads import get_workload
from meta_opt.utils.pytree_utils import pytree_sq_norm

# -------------------------------------------------------------------------------------------------
# ----------------------------------- Hypergradient Descent ---------------------------------------
# -------------------------------------------------------------------------------------------------

def train_hgd(cfg, initial_lr: float, hypergrad_lr: float):

    optimizer = optax.inject_hyperparams(optax.sgd)(learning_rate=initial_lr)
    tstate, train_ds, test_ds, rng, args = get_workload(cfg, optimizer)

    stats = defaultdict(dict)
    args['optimizer_name'] = 'hgd'
    args['optimizer_args'] = {'initial_lr': initial_lr,
                              'hypergrad_lr': hypergrad_lr,
                              }
    stats['args'] = args

    prev_grads = None
    t0 = perf_counter()
    last_eval_step = None
    pbar = tqdm.tqdm(train_ds.as_numpy_iterator(), total=args['num_iters'])
    for t, batch in enumerate(pbar):

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
        if t % args['eval_every'] == 0 and t != 0:
            for k, v in eval(tstate, test_ds.as_numpy_iterator()).items(): s[f'eval_{k}'] = v
            s['param_sq_norm'] = pytree_sq_norm(tstate.params)
            s['grad_sq_norm'] = pytree_sq_norm(grads)
            last_eval_step = t
        if 'bleu_every' in args and t % args['bleu_every'] == 0 and t != 0:
            s['bleu'] = tstate.model.bleu(tstate, test_ds.as_numpy_iterator())
        s['hypergrad'] = hypergrad
        s['lr'] = float(tstate.opt_state.hyperparams['learning_rate'])
        stats[t] = s
        pbar.set_postfix({'loss': round(s['loss'].item(), 3), 
                          'eval_loss': round(stats[last_eval_step]['eval_loss'].item(), 3) if last_eval_step is not None else 'N/A',
                          'lr': round(s['lr'], 3)})

    return dict(stats)
