import numpy as np
import jax
import jax.numpy as jnp
import optax

from .controllers.utils import append, slice_pytree

from .training.trainer import gradient_descent, value_and_jacfwd
from .meta_opt import MetaOptGPCState, compute_control

# --------------------------------------------------------------------------------------------------------------------
# --------------------   Gradient-based Adaptive Policy Selection (GAPS) ---------------------------------------------
# --------------------   (https://arxiv.org/pdf/2210.12320.pdf)          ---------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

"""
Implements the GAPS gradient estimator, specifically for meta-opt.  !!!!!  TODO ONLY FOR SCALAR M's !!!!!

some changes I (Evan) made:
    - we know `dg_dx = (1 - delta)` and `dg_dM = du_dM` because of the update equation `x_{t+1} = x_t - eta * grad(x_t) + M[1:h] * disturbances[t-h:t-1]`
    - we know `partial_u_x = 0` since our controller is not state-feedback
    - we know `partial_f_x` since we compute those grads anyway for GD
    - we know `partial_f_M = 0` since the cost depends only on the current parameters
"""

@jax.jit
def _estimate(buf, delta, dg_dM, partial_f_x):
    buf = jax.tree_util.tree_map(lambda b, g: append((1 - delta) * b, g.reshape(*b.shape[1:])), buf, dg_dM)
    G = jax.tree_util.tree_map(lambda p, b: (p.reshape(-1, 1) * b.sum(axis=0)).sum(axis=0), partial_f_x, buf)
    G = sum(jax.tree_leaves(G))
    return G, buf
        
class GAPSEstimator:
    """Only does the gradient estimation - no logging, no projection."""
    
    def __init__(self, B: int, H: int, initial_params):
        self.B = B
        self.H = H
        self.buffer = jax.tree_util.tree_map(lambda p: jnp.zeros((B, np.prod(p.shape), H)), initial_params)
        self.t = 0
            
    def estimate(self, delta, dg_dM, partial_f_x, partial_f_M):
        if self.t == 0:
            G = 0.
            self.buffer = jax.tree_util.tree_map(lambda b, g: append(b, g.reshape(*b.shape[1:])), self.buffer, dg_dM)
        else:
            G, self.buffer = _estimate(self.buffer, delta, dg_dM, partial_f_x)
        G += partial_f_M
        self.t += 1
        return G
    
    def reset(self):
        self.buffer = jax.tree_util.tree_map(lambda b: jnp.zeros_like(b), self.buffer)
        return self

    
# --------------------------------------------------------------------------------------------------------------------
# --------------------   DEFINE A META-OPT WRAPPER TO MAINTAIN PARAMS/GRADS  -----------------------------------------
# --------------------------------------------------------------------------------------------------------------------

@jax.jit
def update(cstate, grads):  # update the GPC controller
    updates, new_opt_state = cstate.tx.update(grads, cstate.opt_state, cstate.M)
    M = optax.apply_updates(cstate.M, updates)
    return cstate.replace(M=M, opt_state=new_opt_state)  

def _gd_and_loss(M, disturbances, batch, tstate, delta):
    control, (du_dM) = value_and_jacfwd(jax.tree_util.Partial(compute_control, disturbances=disturbances), M)  # use up to and including gradient from t-1
    params = jax.tree_map(lambda p, c: (1 - delta) * p + c, tstate.params, control)
    tstate = tstate.replace(params=params)
    tstate, (loss, grads) = gradient_descent(tstate, batch)
    return loss, (tstate, grads, du_dM)

_gd_and_loss = jax.value_and_grad(_gd_and_loss, has_aux=True)
_gd_and_loss = jax.jit(_gd_and_loss)

class MetaOptGAPS:
    grad_history: jnp.ndarray
    cstate: MetaOptGPCState
    delta: float
    t: int
    H: int
    B: int

    def __init__(self,
                 initial_tstate,
                 H: int, B: int, 
                 meta_lr: float, delta: float,
                 m_method: str):
        self.grad_history = jax.tree_map(lambda p: jnp.zeros((H + 1, *p.shape)), initial_tstate.params)
        self.delta = delta
        self.t = 0
        self.H = H
        self.B = B

        assert m_method in ['scalar',]
        self.cstate = MetaOptGPCState.create(initial_tstate, m_method, H, HH=-1, lr=meta_lr)
        self.grad_estimator = GAPSEstimator(B, H, initial_tstate.params)
        pass    

    def meta_step(self, 
                  tstate,  # tstate before a step of gd
                  batch,  # batch for one step of gd
                 ): 
        
        if self.t >= self.H + self.B:
            (loss, (tstate, grads, du_dM)), df_dM = _gd_and_loss(self.cstate.M, slice_pytree(self.grad_history, 0, self.H), batch, tstate, self.delta)
            du_dM = jax.tree_util.tree_map(lambda d: jnp.transpose(d, axes=(0, 2, 1)) if d.ndim > 2 else d, du_dM)
            m_grads = self.grad_estimator.estimate(self.delta, du_dM, grads, df_dM)
            self.cstate = update(self.cstate, m_grads)
        else:
            tstate, (loss, grads) = gradient_descent(tstate, batch)
            
        self.grad_history = jax.tree_map(lambda h, g: append(h, g), self.grad_history, grads)
        self.t += 1
        return tstate, (loss, grads)
    
    def episode_reset(self):
        self.grad_history = jax.tree_map(lambda p: jnp.zeros_like(p), self.grad_history)
        # self.grad_estimator = self.grad_estimator.reset()
        self.t = 0
        return self
    