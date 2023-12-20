from typing import Tuple
import jax
import jax.numpy as jnp
import optax
from flax import struct

from ._base import ControllerState
from .utils import dare_gain, quad_loss, append

# ----------------------------------------------------------------
# ---------------     define the controller state   --------------
# ----------------------------------------------------------------

class GPCState(ControllerState):
    K: jnp.ndarray  # state-feedback control matrix
    M: jnp.ndarray  # disturbance-feedback control matrix
    disturbance_history: jnp.ndarray  # disturbance history
    
    # system model
    A: jnp.ndarray
    B: jnp.ndarray
    
    state_dim: int = struct.field(pytree_node=False)
    control_dim: int = struct.field(pytree_node=False)
    H: int = struct.field(pytree_node=False)  # history of the controller
    HH: int = struct.field(pytree_node=False)  # history of the system
    t: int  # time counter (for decaying learning rate)
    lr: float
    decay_lr: bool = struct.field(pytree_node=False)
    
    @classmethod
    def create(cls, 
               A: jnp.ndarray,
               B: jnp.ndarray,
               Q: jnp.ndarray = None,
               R: jnp.ndarray = None,
               K: jnp.ndarray = None,
               H: int = 5,
               HH: int = 5,
               lr: float = 0.008,
               decay_lr: bool = True,):
        
        state_dim, control_dim = B.shape
        if Q is None: Q = jnp.eye(state_dim, dtype=jnp.float32)
        if R is None: R = jnp.eye(control_dim, dtype=jnp.float32)
        if K is None: K = dare_gain(A, B, Q, R)
        
        M = jnp.zeros((H, control_dim, state_dim))
        disturbance_history = jnp.zeros((H + HH, state_dim))  # Past H + HH noises ordered increasing in time
        tx = optax.inject_hyperparams(optax.sgd)(learning_rate=lr)
        opt_state = tx.init(M)
        
        return cls(K=K, M=M, disturbance_history=disturbance_history, A=A, B=B,
                   state_dim=state_dim, control_dim=control_dim, H=H, HH=HH, t=0,
                   lr=lr, decay_lr=decay_lr, tx=tx, opt_state=opt_state)
    
# ----------------------------------------------------------------
# ------------    define the controller interactions   -----------
# ----------------------------------------------------------------   
    
@jax.jit
def get_control(rng, cstate: ControllerState, state: jnp.ndarray) -> Tuple[ControllerState, jnp.ndarray]:
    control = cstate.K @ state + jnp.tensordot(cstate.M, jax.lax.dynamic_slice_in_dim(cstate.disturbance_history, -cstate.H, cstate.H), axes=([0, 2], [0, 1]))
    return cstate, control


@jax.jit
def update(cstate: ControllerState, 
           prev_state: jnp.ndarray,
           prev_cost: float,
           control: jnp.ndarray,
           next_state: jnp.ndarray,
           next_cost: float):
    
    pred_next_state = cstate.A @ prev_state + cstate.B @ control
    disturbance = next_state - pred_next_state
    disturbance_history = append(cstate.disturbance_history, disturbance)   

    # apply update
    if cstate.decay_lr: cstate.opt_state.hyperparams['learning_rate'] = cstate.lr / (cstate.t + 1)
    params = cstate.M
    grads = _grad_fn(cstate, params) 
    updates, new_opt_state = cstate.tx.update(grads, cstate.opt_state, params)
    M = optax.apply_updates(params, updates[0])

    return cstate.replace(M=M, opt_state=new_opt_state, disturbance_history=disturbance_history, t=cstate.t+1)   


def reset(cstate: ControllerState):
    disturbance_history = jnp.zeros((cstate.H + cstate.HH, cstate.state_dim))  # Past H + HH noises ordered increasing in time
    return cstate.replace(disturbance_history=disturbance_history)
    
# ----------------------------------------------------------------
# ------------------------    utilities   ------------------------
# ----------------------------------------------------------------  

def _loss_fn(cstate: GPCState, M: jnp.ndarray):
    """Surrogate cost function"""
    def _action(state, h):
        return cstate.K @ state + jnp.tensordot(M, jax.lax.dynamic_slice_in_dim(cstate.disturbance_history, h, cstate.H), axes=([0, 2], [0, 1]))
    def _evolve(state, h):
        return cstate.A @ state + cstate.B @ _action(state, h) + cstate.disturbance_history[h + cstate.H], None
    final_state, _ = jax.lax.scan(_evolve, jnp.zeros((cstate.state_dim,)), jnp.arange(cstate.HH - 1))  # NOTE i think this should be `HH`, not `H`
    return quad_loss(final_state, _action(final_state, cstate.HH - 1))

_grad_fn = jax.jit(jax.grad(_loss_fn, (1,)))
