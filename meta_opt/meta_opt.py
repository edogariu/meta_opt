from collections import deque
from typing import Tuple, Dict

import jax
import jax.numpy as jnp
import optax
from flax import struct

from .controllers._base import ControllerState
from .controllers.utils import append, slice_pytree, index_pytree, add_pytrees, multiply_pytrees, multiply_pytree_by_scalar

from .nn.trainer import forward, gradient_descent


# --------------------------------------------------------------------------------------------------------------------
# --------------------   DEFINE THE GPC CONTROLLER TO USE IN META-OPT  -----------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

class MetaOptGPCState(ControllerState):
    cparams: Dict[str, jnp.ndarray]  # dict of pytrees of disturbance-feedback control matrices (which themselves may be in pytree form if they multiply a pytree)
    
    H: int = struct.field(pytree_node=False)  # history of the controller, how many past disturbances to use for control
    HH: int = struct.field(pytree_node=False)  # history of the system, how many hallucination steps to take
    num_params: int
    
    @classmethod
    def create(cls, 
               tstate,
               m_method: str,
               H: int,
               HH: int,
               meta_optimizer,
               grad_clip: float,
               ema_keys = [],
               ):
        # make controller
        if m_method == 'scalar': 
            M = jnp.zeros((H,))
            M_ema = {e: jnp.zeros(()) for e in ema_keys}
        elif m_method == 'diagonal': 
            M = jax.tree_map(lambda p: jnp.zeros((H, *p.shape)), tstate.params)
            M_ema = {e: jax.tree_map(jnp.zeros_like, tstate.params) for e in ema_keys}
        else: raise NotImplementedError(m_method)
        cparams = {'M': M, 'M_ema': M_ema}

        # make optimizer 
        tx = meta_optimizer
        if grad_clip is not None: tx = optax.chain(optax.clip(grad_clip), tx)  # clip grads
        opt_state = tx.init(cparams)
        
        param_counts = {k: sum(x.size for x in jax.tree_util.tree_leaves(v)) for k, v in cparams.items()}
        num_params = sum(param_counts.values())
        print(num_params, f'params in the controller {param_counts}')
        return cls(cparams=cparams,
                   H=H, HH=HH, num_params=num_params,
                   tx=tx, opt_state=opt_state)

@jax.jit
def compute_control(cparams, disturbances, emas):
    M, M_ema = cparams['M'], cparams['M_ema']
    if isinstance(M, jnp.ndarray): 
        control = jax.tree_map(lambda d: (jax.lax.expand_dims(M, range(1, d.ndim)) * d).sum(axis=0), disturbances)
        if len(emas) > 0:  # add the contribution from the emas
            control = add_pytrees(control, *map(multiply_pytree_by_scalar, M_ema.values(), emas.values()))
    else: 
        control = jax.tree_map(lambda m, d: (m * d).sum(axis=0), M, disturbances)
        if len(emas) > 0:
            control = add_pytrees(control, *map(multiply_pytrees, M_ema.values(), emas.values()))
    return control

@jax.jit
def _hallucinate(cparams, tstate, disturbances, emas, batch):
    tstate, _ = gradient_descent(tstate, batch)
    params = add_pytrees(tstate.params, compute_control(cparams, disturbances, emas))
    tstate = tstate.replace(params=params)
    return tstate

def _compute_loss(cparams, H, HH, initial_tstate, 
                  disturbances,  # past H + HH disturbances
                  initial_emas,  # dict of the `{momentum_coefficient: pytree_of_running_avgs}` sort
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
    emas = initial_emas
    for h in range(HH):
        # update emas or something like that, then hallucinate
        for beta, avg in emas.items(): emas[beta] = jax.tree_map(lambda v, g: beta * v + (1 - beta) * g, avg, index_pytree(disturbances, h + H - 1))  # update emas
        tstate = _hallucinate(cparams, tstate, slice_pytree(disturbances, h, H), emas, batches[h])
    loss = forward(tstate, batches[-1])
    return loss

_grad_fn = jax.grad(_compute_loss, (0,))

@jax.jit
def update(cstate,
           initial_tstate,  # tstate from HH steps ago
           disturbances,  # past H + HH disturbances
           initial_emas,  # dict of the `{momentum_coefficient: pytree_of_running_avgs}` sort
           batches,  # past HH + 1 batches, starting at the one that would have been used to evolve `initial_params` and ending with the current one
          ):
    
    grads = _grad_fn(cstate.cparams, cstate.H, cstate.HH, initial_tstate, disturbances, initial_emas, batches)    
    updates, new_opt_state = cstate.tx.update(grads[0], cstate.opt_state, cstate.cparams)
    cparams = optax.apply_updates(cstate.cparams, updates)
    return cstate.replace(cparams=cparams, opt_state=new_opt_state)   



# --------------------------------------------------------------------------------------------------------------------
# --------------------   DEFINE A META-OPT WRAPPER TO MAINTAIN PARAMS/GRADS  -----------------------------------------
# --------------------------------------------------------------------------------------------------------------------

class MetaOpt:
    tstate_history: Tuple
    grad_history: jnp.ndarray
    emas: jnp.ndarray  # emas to compute controls at the current timestep (i.e. this is NOT a history)
    batch_history: Tuple
    cstate: MetaOptGPCState
    t: int

    def __init__(self,
                 initial_tstate,
                 H: int, HH: int,
                 meta_optimizer,
                 m_method: str, 
                 grad_clip: float,
                 ema_keys = [], 
                 ):
        self.tstate_history = (None,) * (HH + 1)
        self.grad_history = jax.tree_map(lambda p: jnp.zeros((H + HH, *p.shape)), initial_tstate.params)
        self.emas = {k: jax.tree_map(jnp.zeros_like, initial_tstate.params) for k in ema_keys}
        self.batch_history = (None,) * (HH + 1)  # need one more because we hallucinate on `batch_history[:HH]` and compute stage loss via `batch_history[-1]`
        self.t = 0

        assert m_method in ['scalar', 'diagonal']
        self.cstate = MetaOptGPCState.create(initial_tstate, m_method, H, HH, meta_optimizer=meta_optimizer, ema_keys=ema_keys, grad_clip=grad_clip)
        pass

    def meta_step(self, 
                  tstate,  # tstate after a step of gd
                  grads,  # grads from the step of gd that resulted in `tstate`
                  batch,  # batch from step of gd that resulted in `tstate`
                 ):      
        
        self.batch_history = append(self.batch_history, batch)        

        # clip disturbances
        K = 5.0; grads = jax.tree_map(lambda g: jnp.clip(g, -K, K), grads)
                     
        self.grad_history = jax.tree_map(append, self.grad_history, grads)
        for beta, avg in self.emas.items(): self.emas[beta] = jax.tree_map(lambda v, g: beta * v + (1 - beta) * g, avg, grads)  # update emas

        if self.t >= self.cstate.H + self.cstate.HH:
            control = compute_control(self.cstate.cparams, slice_pytree(self.grad_history, self.cstate.HH, self.cstate.H), self.emas)  # use past H disturbances
            tstate = tstate.replace(params=add_pytrees(tstate.params, control))
            
            # compute the states of the ema buffers from `HH` steps ago by reversing the running averages
            initial_emas = {}
            for j in range(1, self.cstate.HH + 1):
                grad = index_pytree(self.grad_history, self.cstate.H + self.cstate.HH - j)
                for beta, avg in self.emas.items():
                    if j == 1: initial_emas[beta] = avg
                    initial_emas[beta] = jax.tree_map(lambda v, g: (v - (1 - beta) * g) / beta, initial_emas[beta], grad)  # reverse the emas
            self.cstate = update(self.cstate, self.tstate_history[0], self.grad_history, initial_emas, self.batch_history)
            
        self.tstate_history = append(self.tstate_history, tstate)
        self.t += 1
        return tstate
    
    def episode_reset(self):
        H, HH = self.cstate.H, self.cstate.HH
        self.grad_history = jax.tree_map(jnp.zeros_like, self.grad_history)
        self.tstate_history = (None,) * (HH + 1)
        self.batch_history = (None,) * (HH + 1)
        self.t = 0
        self.cstate = self.cstate.replace(opt_state=self.cstate.tx.init(self.cstate.cparams))
        return self
