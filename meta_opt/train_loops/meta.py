from time import perf_counter
from collections import defaultdict
import tqdm
from copy import deepcopy

import jax
import optax

from meta_opt.meta_opt import MetaOpt
from meta_opt.nn import reset_model, train_step, eval
from meta_opt.workloads import get_workload
from meta_opt.utils.pytree_utils import pytree_sq_norm, pytree_proj
from meta_opt.utils.experiment_utils import get_opt_hyperparams

# -------------------------------------------------------------------------------------------------
# ----------------------------------- Our Meta Optimizer ------------------------------------------
# -------------------------------------------------------------------------------------------------

def train_meta_opt(cfg, 
                   counterfactual: bool, 
                   meta_optimizer, 
                   H: int, HH: int, 
                   m_method: str = 'scalar', 
                   initial_lr: float = 1e-4, grad_clip = 10, dtype=jax.numpy.float32,
                   cparams_initial = None, base_opt_type: str = 'sgd', disturbance_transformation: optax.GradientTransformation = None): 
    
    """
    note that if we aren't counterfactual, we have to rescale the number of iterations by HH to account for taking HH training steps every noncounterfactual meta step
    """
    cfg = deepcopy(cfg)
    
    if base_opt_type == 'sgd' or not counterfactual:
        optimizer = optax.chain(optax.add_decayed_weights(1e-5), optax.sgd(learning_rate=initial_lr))  
    else:
        print('using adam for base optimizer')
        optimizer = optax.adamw(learning_rate=initial_lr, weight_decay=1e-5)
    if disturbance_transformation is not None and counterfactual:
        print(f'using {disturbance_transformation} to rescale gradients being passed to meta-opt!')
        
    tstate, train_ds, test_ds, rng, args = get_workload(dict(cfg, **({'num_iters': cfg['num_iters'] // HH} if not counterfactual else {})), optimizer)
    meta_opt = MetaOpt(tstate.params, H=H, HH=HH, m_method=m_method, meta_optimizer=meta_optimizer, meta_grad_clip=grad_clip, dtype=dtype, grad_transformation=disturbance_transformation)
    
    if cparams_initial is not None: meta_opt.cstate = meta_opt.cstate.replace(cparams=cparams_initial)

    def check(t, k):  # to check conditions that happen every `n` steps, since `t` will increment by 1 if counterfactual and by `HH` otherwise
        n = args[k]
        if counterfactual: return t % n == 0
        else: return t % n < HH

    stats = defaultdict(dict)
    if not counterfactual: args['num_iters'] *= HH
    args['optimizer_name'] = 'meta'
    args['optimizer_args'] = {'initial_lr': initial_lr,
                              'm_method': m_method,
                              'meta_optimizer_args': get_opt_hyperparams(meta_opt.cstate.opt_state),
                              'disturbance_rescale_args': None if meta_opt.grad_transformation is None else get_opt_hyperparams(meta_opt.grad_transformation_state),
                              'H': H,
                              'HH': HH,
                              'grad_clip': grad_clip,
                              'dtype': dtype,
                              'cparams_initial_supplied': cparams_initial is not None,
                              'base_opt_type': base_opt_type,
                              }
    stats['args'] = args

    t0 = perf_counter()
    last_eval_step = None
    pbar = tqdm.tqdm(train_ds.as_numpy_iterator(), total=cfg['num_iters'])
    for t, batch in enumerate(pbar):
        if not counterfactual: t *= HH

        if counterfactual:
            tstate, (loss, grads) = train_step(tstate, batch)
            tstate = meta_opt.counterfactual_step(tstate, grads, batch)
        else:
            tstate, (loss, grads) = meta_opt.noncounterfactual_step(tstate, batch)

        # update all the stats
        s = {}
        s['timestamp'] = perf_counter() - t0
        s['loss'] = loss
        if check(t, 'eval_every') and t != 0:
            for k, v in eval(tstate, test_ds.as_numpy_iterator()).items(): s[f'eval_{k}'] = v
            s['param_sq_norm'] = pytree_sq_norm(tstate.params)
            s['grad_sq_norm'] = pytree_sq_norm(grads)
            if hasattr(tstate.model, 'radius'):
                proj_grads = pytree_proj(grads, tstate.params)
                s['proj_grad_sq_norm'] = pytree_sq_norm(proj_grads)
            last_eval_step = t
        if 'bleu_every' in args and check(t, 'bleu_every') and t != 0:
            s['bleu'], s['bleu_exemplars'] = tstate.model.bleu(tstate, test_ds.as_numpy_iterator())
            print(s['bleu'], s['bleu_exemplars'])
        # s['lr'] = float(tstate.opt_state.hyperparams['learning_rate'])

        # log the value of the Ms
        s['M'] = meta_opt.cstate.cparams['M']
        stats[t] = s
        pbar.set_postfix({'loss': round(s['loss'].item(), 3), 
                          'eval_loss': round(stats[last_eval_step]['eval_loss'].item(), 3) if last_eval_step is not None else 'N/A',
                        #   'lr': round(s['lr'], 5),
                          },)
        if not counterfactual: pbar.update(HH)

        if check(t, 'reset_every'):
            reset_rng, rng = jax.random.split(rng)
            tstate = reset_model(reset_rng, tstate)
            meta_opt = meta_opt.episode_reset()
            del reset_rng
            
            
    return dict(stats)
