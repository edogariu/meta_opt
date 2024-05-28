from typing import Tuple, Callable, Dict
import functools
from collections import defaultdict

import tensorflow as tf
import jax
import jax.numpy as jnp

import flax.linen as jnn
from flax import struct
from flax.training import train_state

from meta_opt.utils.pytree_utils import pytree_sq_norm

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
                               loss_fn=jax.tree_util.Partial(jax.jit(loss_fn)), 
                               metric_fns={k: jax.tree_util.Partial(v) for k, v in metric_fns.items()},
                            #    metric_fns={k: jax.tree_util.Partial(jax.jit(v)) for k, v in metric_fns.items()},
                               other_vars={},
                               rng=None,)
    return reset_model(rng, tstate)


@jax.jit
def project(tstate):
    if hasattr(tstate.model, 'radius'): 
        div = jnp.maximum(1., ((pytree_sq_norm(tstate.params) ** 0.5) / tstate.model.radius))
        params = jax.tree_util.tree_map(lambda p: p / div, tstate.params)
        return tstate.replace(params=params)
    else: 
        return tstate

@jax.jit
def forward(tstate, batch):
    tstate = project(tstate)
    variables = {'params': tstate.params, 'batch_stats': tstate.batch_stats}
    variables.update(tstate.other_vars)
    yhat, updates = tstate.apply_fn(variables, batch['x'], train=False, mutable=['batch_stats'])
    loss = tstate.loss_fn(yhat, batch['y'])
    return loss, yhat

@jax.jit
# print('WARNING!!! TRAIN_STEP ISNT JITTED')
def train_step(tstate, batch):    
    
    if tstate.rng is not None:  # some rng hacking that is very anti-jax :)
        next_key, dropout_key = jax.random.split(tstate.rng)
        tstate = tstate.replace(rng=next_key)
    else: dropout_key = None
    
    tstate = project(tstate)
    
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
    # (loss, (yhat, updates)), grads = jax.value_and_grad(loss_fn, has_aux=True)(tstate.params)
    (loss, (yhat, updates)), grads = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))(tstate.params)
    tstate = tstate.apply_gradients(grads=grads)
    tstate = project(tstate)
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
        print('exception during eval!!!')
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


from typing import NamedTuple, Optional
import optax
import chex
import jax.numpy as jnp
import jax.tree_util as tu
from optax import tree_utils
from optax._src import base
from optax._src import utils
def dadapt_sgd(
    learning_rate: base.ScalarOrSchedule = 1.0,
    eps: float = 1e-8,
    estim_lr0: float = 1e-6,
    weight_decay: float = 0.,
) -> base.GradientTransformation:
  """Learning rate free AdamW by D-Adaptation.

  Adapts the baseline learning rate of AdamW automatically by estimating the
  initial distance to solution in the infinity norm.
  This method works best when combined with a learning rate schedule that
  treats 1.0 as the base (usually max) value.

  References:
    [Defazio & Mishchenko, 2023](https://arxiv.org/abs/2301.07733.pdf)

  Args:
    learning_rate: Learning rate scheduling parameter. The recommended schedule
      is a linear_schedule with init_value=1.0 and end_value=0, combined with a
      0-20% learning rate warmup.
    betas: Betas for the underlying AdamW Optimizer.
    eps: eps for the underlying AdamW Optimizer.
    estim_lr0: Initial (under-)estimate of the learning rate.
    weight_decay: AdamW style weight-decay. To use Regular Adam decay, chain
      with add_decayed_weights.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params: base.Params) -> optax.contrib.DAdaptAdamWState:
    exp_avg = tu.tree_map(lambda p: jnp.zeros(p.shape, jnp.float32), params)
    exp_avg_sq = tu.tree_map(lambda p: jnp.zeros(p.shape, jnp.float32), params)
    grad_sum = tu.tree_map(lambda p: jnp.zeros(p.shape, jnp.float32), params)
    estim_lr = jnp.asarray(estim_lr0, jnp.float32)
    numerator_weighted = jnp.zeros([], jnp.float32)
    count = jnp.zeros([], jnp.int32)
    return optax.contrib.DAdaptAdamWState(
        exp_avg, exp_avg_sq, grad_sum, estim_lr, numerator_weighted, count
    )

  def update_fn(
      updates: base.Updates,
      state: optax.contrib.DAdaptAdamWState,
      params: Optional[base.Params] = None,
  ) -> tuple[base.Updates, optax.contrib.DAdaptAdamWState]:
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)
    count = state.count
    beta1, beta2 = betas
    sb2 = beta2 ** (0.5)
    sched = learning_rate(count) if callable(learning_rate) else learning_rate
    grad_sum = state.grad_sum
    numerator_weighted = state.numerator_weighted
    bc = ((1 - beta2 ** (count + 1)) ** 0.5) / (1 - beta1 ** (count + 1))
    dlr = state.estim_lr * sched * bc
    s_weighted = tu.tree_map(
        lambda sk, eas: sk / (jnp.sqrt(eas) + eps), grad_sum, state.exp_avg_sq
    )
    numerator_acum = tree_utils.tree_vdot(updates, s_weighted)
    exp_avg = tu.tree_map(
        lambda ea, g: beta1 * ea + (1 - beta1) * dlr * g, state.exp_avg, updates
    )
    exp_avg_sq = tu.tree_map(
        lambda eas, g: beta2 * eas + (1 - beta2) * g * g,
        state.exp_avg_sq,
        updates,
    )
    grad_sum = tu.tree_map(
        lambda sk, g: sb2 * sk + (1 - sb2) * dlr * g, grad_sum, updates
    )
    grad_sum_l1 = tree_utils.tree_sum(tu.tree_map(jnp.abs, grad_sum))
    numerator_weighted = (
        sb2 * numerator_weighted + (1 - sb2) * dlr * numerator_acum
    )
    d_estimate = numerator_weighted / ((1 - sb2) * grad_sum_l1)
    estim_lr = jnp.maximum(state.estim_lr, d_estimate)
    p_update = tu.tree_map(
        lambda ea, eas, p: -weight_decay * dlr * p - ea / (jnp.sqrt(eas) + eps),
        exp_avg,
        exp_avg_sq,
        params,
    )
    new_state = optax.contrib.DAdaptAdamWState(
        exp_avg,
        exp_avg_sq,
        grad_sum,
        estim_lr,
        numerator_weighted,
        utils.safe_int32_increment(count),
    )
    return p_update, new_state

  return base.GradientTransformation(init_fn, update_fn)