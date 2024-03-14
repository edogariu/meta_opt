from typing import Tuple, Callable, Dict
import functools
from collections import defaultdict

import tensorflow as tf
import jax
import jax.numpy as jnp

import flax.linen as jnn
from flax import struct
from flax.training import train_state

class TrainState(train_state.TrainState):
    batch_stats: jnp.ndarray
    loss_fn: Callable[[Tuple[jnp.ndarray, jnp.ndarray]], float]
    metric_fns: Dict[str, Callable[[Tuple[jnp.ndarray, jnp.ndarray]], float]]
    model: jnn.Module = struct.field(pytree_node=False)
    example_input: jnp.ndarray
    rng: jnp.ndarray
    other_vars: Dict[str, jnp.ndarray]

def reset_model(rng, tstate: TrainState):
    init_rng, dropout_rng, rng = jax.random.split(rng, 3)
    variables = tstate.model.init({'params': init_rng, 'dropout': dropout_rng}, tstate.example_input, train=False)
    params, batch_stats = variables['params'], variables['batch_stats'] if 'batch_stats' in variables else {}  # initialize parameters by passing a template input
    other_vars = {k: v for k, v in variables.items() if k not in ['params', 'batch_stats']}
    opt_state = tstate.tx.init(params)
    tstate = tstate.replace(params=params, batch_stats=batch_stats, opt_state=opt_state, other_vars=other_vars, rng=rng)
    return tstate

def create_train_state(rng, model: jnn.Module, example_input: jnp.ndarray, optimizer, loss_fn, metric_fns={}):
    """Creates an initial `TrainState`."""
    tstate = TrainState.create(model=model, 
                               apply_fn=model.apply, 
                               params={}, 
                               batch_stats={},
                               tx=optimizer,
                               example_input=example_input, 
                               loss_fn=jax.tree_util.Partial(loss_fn), 
                               metric_fns={k: jax.tree_util.Partial(v) for k, v in metric_fns.items()},
                               other_vars={},
                               rng=None,)
    return reset_model(rng, tstate)


@jax.jit
def forward(tstate, batch):
    variables = {'params': tstate.params, 'batch_stats': tstate.batch_stats}
    variables.update(tstate.other_vars)
    yhat, updates = tstate.apply_fn(variables, batch['x'], train=False,)
    loss = tstate.loss_fn(yhat, batch['y'])
    return loss, yhat


# @jax.jit
print('trainstep ISNT JITTED!!!')
def train_step(tstate, batch):    
    
    if tstate.rng is not None:  # some rng hacking that is very anti-jax :)
        next_key, dropout_key = jax.random.split(tstate.rng)
        tstate = tstate.replace(rng=next_key)
    else: dropout_key = None
    
    print('lr at beginning:', tstate.opt_state.hyperparams['learning_rate'])
    
    # define grad fn
    def loss_fn(params):
        variables = {'params': params, 'batch_stats': tstate.batch_stats}
        variables.update(tstate.other_vars)
        yhat, updates = tstate.apply_fn(variables, 
                                        batch['x'], train=True, 
                                        rngs={'dropout': dropout_key}, mutable=['batch_stats'])
        loss = tstate.loss_fn(yhat, batch['y'])
        return loss, (yhat, updates)

    # get loss and grads
    (loss, (yhat, updates)), grads = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))(tstate.params)
    tstate = tstate.apply_gradients(grads=grads)
    print('lr after update:', tstate.opt_state.hyperparams['learning_rate'])
    tstate = tstate.replace(batch_stats=updates['batch_stats'])
    return tstate, (loss, grads)


# @jax.jit
def eval(tstate, dataset):
    eval_metrics = defaultdict(float)
    n = 0
    try:
        for batch in dataset:
            yhat, y = forward(tstate, batch)[1], batch['y']
            for k, v in tstate.metric_fns.items(): eval_metrics[k] += v(yhat, y)
            n += 1
    except Exception as e: 
        print(e)
        pass
    for k in eval_metrics.keys(): eval_metrics[k] /= n
    return dict(eval_metrics)


@jax.jit
def value_and_jacfwd(f, x):
    pushfwd = functools.partial(jax.jvp, f, (x,))
    basis = jnp.eye(x.size, dtype=x.dtype)
    y, jac = jax.vmap(pushfwd, out_axes=(None, 1))((basis,))
    return y, jac

@jax.jit
def value_and_jacrev(f, *x):
    y, pullback = jax.vjp(f, *x)
    basis = jnp.eye(y.size, dtype=y.dtype)
    jac = jax.vmap(pullback)(basis)
    return y, jac
