from time import perf_counter
from collections import defaultdict
import tqdm
import jax

from meta_opt.nn import reset_model, train_step, eval
from meta_opt.workloads import get_workload
from meta_opt.utils.pytree_utils import pytree_sq_norm, pytree_proj
from meta_opt.utils.experiment_utils import get_opt_hyperparams

# -------------------------------------------------------------------------------------------------
# ------------------------------ Standard Optax Optimizers ----------------------------------------
# -------------------------------------------------------------------------------------------------

def train_standard_opt(cfg, optimizer):
    tstate, train_ds, test_ds, rng, args = get_workload(cfg, optimizer)

    stats = defaultdict(dict)
    args['optimizer_args'] = get_opt_hyperparams(tstate.opt_state)
    args['optimizer_name'] = 'standard'
    stats['args'] = args

    t0 = perf_counter()
    last_eval_step = None
    pbar = tqdm.tqdm(train_ds.as_numpy_iterator(), total=args['num_iters'])
    for t, batch in enumerate(pbar):

        tstate, (loss, grads) = train_step(tstate, batch)

        # update all the stats
        s = {}
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
