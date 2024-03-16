import functools
from typing import Any, Callable, NamedTuple, Optional, Union

import chex
import jax
import jax.numpy as jnp

from optax._src import base
from optax._src import numerics
from optax._src import utils
from optax._src import wrappers
from optax._src import clipping
from optax._src import combine
from optax._src import factorized
from optax._src import transform



class MetaOptState(NamedTuple):
    grad_history: base.Updates
    cstate: MetaOptGPCState
    t: int
    counterfactual: bool
    
    # things for counterfactual updates; these won't be used for noncounterfactual
    tstate_history: base.Updates
    batch_history: base.Updates
    H: int
    HH: int
    
def tune_by_meta_opt(H: int, HH: int,
                 meta_optimizer,
                 m_method: str, 
                 counterfactual: bool,
                 grad_clip: float,
                 dtype,):
    
    assert m_method in ['scalar', 'diagonal', 'full']
    def init_fn(params):
        grad_history = jax.tree_map(lambda p: jnp.zeros((H, *p.shape)), initial_tstate.params)
        t = 0
        cstate = MetaOptGPCState.create(initial_params, m_method, H, HH, meta_optimizer=meta_optimizer, grad_clip=grad_clip, dtype=dtype)
        
        # if we dont do counterfactual steps, these will remain unused
        tstate_history = (None,) * (HH + 1)
        batch_history = None  # this will be size HH
        return MetaOptState(grad_history=grad_history, cstate=cstate, t=t, counterfactual=counterfactual, tstate_history=tstate_history, batch_history=batch_history, H=H, HH=HH)
    
    # if counterfactual:
    #     def update_fn(updates, state, params=None):
    #         # del params
    #         # mu = update_moment(updates, state.mu, b1, 1)
    #         # nu = update_moment_per_elem_norm(updates, state.nu, b2, 2)
    #         # count_inc = numerics.safe_int32_increment(state.count)
    #         # if nesterov:
    #         # mu_hat = jax.tree_util.tree_map(
    #         #     lambda m, g: b1 * m + (1 - b1) * g,
    #         #     bias_correction(mu, b1, numerics.safe_int32_increment(count_inc)),
    #         #     bias_correction(updates, b1, count_inc))
    #         # else:
    #         # mu_hat = bias_correction(mu, b1, count_inc)
    #         # # Dozat 2016 https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ
    #         # # Algorithm 2 further multiplies Adam's standard nu_hat by b2. It is
    #         # # unclear why. Other Nadam implementations also omit the extra b2 factor.
    #         # nu_hat = bias_correction(nu, b2, count_inc)
    #         # updates = jax.tree_util.tree_map(
    #         #     lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat)
    #         # mu = utils.cast_tree(mu, mu_dtype)
    #         return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)
    # else:
    #     def update_fn(updates, state, params=None):
            
    #     def noncounterfactual_step(self, tstate, batch):  
    #     # do HH train steps and update the controller if we have long enough histories. Otherwise, simply do HH train steps
    #     if self.t >= self.cstate.H:
    #         self.cstate, (tstate, loss, self.grad_history) = noncounterfactual_update(self.cstate, tstate, batch, self.grad_history)
    #     else: 
    #         for _ in range(self.cstate.HH): 
    #             tstate, loss, self.grad_history = jax.jit(_roll_forward)(tstate, batch, self.grad_history)
    #     self.t += self.cstate.HH
    #     return tstate, (loss, index_pytree(self.grad_history, -1))

    return base.GradientTransformation(init_fn, update_fn)
    



MaskOrFn = Optional[Union[Any, Callable[[base.Params], Any]]]

class ScaleByAdamState(NamedTuple):
  """State for the Adam algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  mu: base.Updates
  nu: base.Updates


def scale_by_adam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[chex.ArrayDType] = None,
    *,
    nesterov: bool = False
) -> base.GradientTransformation:
  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = jax.tree_util.tree_map(  # First moment
        lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
    nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = update_moment(updates, state.mu, b1, 1)
    nu = update_moment_per_elem_norm(updates, state.nu, b2, 2)
    count_inc = numerics.safe_int32_increment(state.count)
    if nesterov:
      mu_hat = jax.tree_util.tree_map(
          lambda m, g: b1 * m + (1 - b1) * g,
          bias_correction(mu, b1, numerics.safe_int32_increment(count_inc)),
          bias_correction(updates, b1, count_inc))
    else:
      mu_hat = bias_correction(mu, b1, count_inc)
    # Dozat 2016 https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ
    # Algorithm 2 further multiplies Adam's standard nu_hat by b2. It is
    # unclear why. Other Nadam implementations also omit the extra b2 factor.
    nu_hat = bias_correction(nu, b2, count_inc)
    updates = jax.tree_util.tree_map(
        lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat)
    mu = utils.cast_tree(mu, mu_dtype)
    return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)




def adam(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    *,
    nesterov: bool = False
) -> base.GradientTransformation:
    return combine.chain(
      transform.scale_by_adam(
          b1=b1,
          b2=b2,
          eps=eps,
          eps_root=eps_root,
          mu_dtype=mu_dtype,
          nesterov=nesterov,
      ),
      transform.scale_by_learning_rate(learning_rate),
  )

