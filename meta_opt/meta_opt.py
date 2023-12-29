from collections import deque
from typing import Tuple

import jax
import jax.numpy as jnp
import optax
from flax import struct

from .controllers._base import ControllerState
from .controllers.utils import append, slice_pytree

from .training.trainer import forward, gradient_descent


# --------------------------------------------------------------------------------------------------------------------
# --------------------   DEFINE THE GPC CONTROLLER TO USE IN META-OPT  -----------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

class MetaOptGPCState(ControllerState):
    M: jnp.ndarray  # pytree of disturbance-feedback control matrices
    
    H: int = struct.field(pytree_node=False)  # history of the controller, how many past disturbances to use for control
    HH: int = struct.field(pytree_node=False)  # history of the system, how many hallucination steps to take
    lr: float
    
    @classmethod
    def create(cls, 
               tstate,
               m_method: str,
               H: int,
               HH: int,
               lr: float = 0.001,
               use_adam: bool = False):

        if m_method == 'scalar': M = jnp.zeros((H,))
        elif m_method == 'diagonal': M = jax.tree_map(lambda p: jnp.zeros((H, *p.shape)), tstate.params)
        else: raise NotImplementedError(m_method)
        
        if not use_adam: tx = optax.sgd(learning_rate=lr)  # M optimizer
        else: tx = optax.adam(learning_rate=lr)
        opt_state = tx.init((M,))
        
        return cls(M=M,
                   H=H, HH=HH, 
                   lr=lr, tx=tx, opt_state=opt_state)

@jax.jit
def compute_control(M, disturbances):
    if isinstance(M, jnp.ndarray): control = jax.tree_map(lambda d: (jax.lax.expand_dims(M, range(1, d.ndim)) * d).sum(axis=0), disturbances)
    else: control = jax.tree_map(lambda m, d: (m * d).sum(axis=0), M, disturbances)
    return control

@jax.jit
def _hallucinate(M, tstate, disturbances, batch):
    tstate, _ = gradient_descent(tstate, batch)
    params = jax.tree_map(lambda p, c: p + c, tstate.params, compute_control(M, disturbances))
    tstate = tstate.replace(params=params)
    return tstate

def _compute_loss(M, H, HH, initial_tstate, 
                  disturbances,  # past H + HH disturbances
                  batches,  # past HH + 1 batches, starting at the one that would have been used to evolve `initial_params` and ending with the current one
                 ):
    # def _evolve(tstate, h):
    #     tstate, _ = gradient_descent(tstate, batches[h])
    #     params = jax.tree_map(lambda p, c: p + c, tstate.params, compute_control(M, slice_pytree(disturbances, h, H)))
    #     return tstate.replace(params=params), None
    # tstate, _ = jax.lax.scan(_evolve, initial_tstate, jnp.arange(HH))
    # loss = forward(tstate, batches[-1])
    # return loss

    tstate = initial_tstate
    for h in range(HH):
        tstate = _hallucinate(M, tstate, slice_pytree(disturbances, h, H), batches[h])
    loss = forward(tstate, batches[-1])
    return loss

_grad_fn = jax.grad(_compute_loss, (0,))

@jax.jit
def update(cstate,
           initial_tstate,  # tstate from HH steps ago
           disturbances,  # past H + HH disturbances
           batches,  # past HH + 1 batches, starting at the one that would have been used to evolve `initial_params` and ending with the current one
          ):
    
    grads = _grad_fn(cstate.M, cstate.H, cstate.HH, initial_tstate, disturbances, batches)
    
    # clip grads)
    K = 0.5
    if isinstance(cstate.M, jnp.ndarray): 
        grads = (jnp.clip(grads[0], -K, K),)
    else: grads = (jax.tree_map(lambda g: jnp.clip(g, -K, K), grads[0]),)
    
    updates, new_opt_state = cstate.tx.update(grads, cstate.opt_state, (cstate.M,))
    M = optax.apply_updates(cstate.M, updates[0])
    return cstate.replace(M=M, opt_state=new_opt_state)   



# --------------------------------------------------------------------------------------------------------------------
# --------------------   DEFINE A META-OPT WRAPPER TO MAINTAIN PARAMS/GRADS  -----------------------------------------
# --------------------------------------------------------------------------------------------------------------------

class MetaOpt:
    tstate_history: Tuple
    grad_history: jnp.ndarray
    batch_history: Tuple
    cstate: MetaOptGPCState
    delta: float
    t: int

    def __init__(self,
                 initial_tstate,
                 H: int, HH: int,
                 meta_lr: float, use_adam: bool, delta: float,
                 m_method: str):
        self.tstate_history = (None,) * HH
        self.grad_history = jax.tree_map(lambda p: jnp.zeros((H + HH, *p.shape)), initial_tstate.params)
        self.batch_history = (None,) * (HH + 1)
        self.delta = delta
        self.t = 0

        assert m_method in ['scalar', 'diagonal']
        self.cstate = MetaOptGPCState.create(initial_tstate, m_method, H, HH, lr=meta_lr, use_adam=use_adam)
        pass

    def meta_step(self, 
                  tstate,  # tstate after a step of gd
                  grads,  # grads from the step of gd that resulted in `tstate`
                  batch,  # batch from step of gd that resulted in `tstate`
                 ):      
        
        self.batch_history = append(self.batch_history, batch)
        self.grad_history = jax.tree_map(lambda h, g: append(h, g), self.grad_history, grads)

        if self.t >= self.cstate.H + self.cstate.HH:
            control = compute_control(self.cstate.M, slice_pytree(self.grad_history, self.cstate.HH, self.cstate.H))  # use past H disturbances
            params = jax.tree_map(lambda p, c: (1 - self.delta) * p + c, tstate.params, control)
            tstate = tstate.replace(params=params)
            self.cstate = update(self.cstate, self.tstate_history[0], self.grad_history, self.batch_history)
        
        self.tstate_history = append(self.tstate_history, tstate)
        self.t += 1
        return tstate
    
    def episode_reset(self):
        H, HH = self.cstate.H, self.cstate.HH
        self.grad_history = jax.tree_map(lambda p: jnp.zeros_like(p), self.grad_history)
        self.tstate_history = (None,) * HH
        self.batch_history = (None,) * (HH + 1)
        self.t = 0
        return self
