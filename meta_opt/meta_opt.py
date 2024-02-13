from collections import deque
from typing import Tuple, Dict

import jax
import jax.numpy as jnp
import optax
from flax import struct

from .controllers._base import ControllerState
from .controllers.utils import append, slice_pytree, index_pytree, add_pytrees, multiply_pytrees, multiply_pytree_by_scalar

from .nn.trainer import forward, train_step


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

# -------------------------------------------------------------------------------------------------------------------------------
# ----------------------------        the counterfactual way --------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------

def _hallucinate(cparams, tstate, disturbances, emas, batch):
    tstate, _ = train_step(tstate, batch)
    params = add_pytrees(tstate.params, compute_control(cparams, disturbances, emas))
    tstate = tstate.replace(params=params)
    return tstate

# @jax.jit
def _compute_loss_counterfactual(cparams, H, HH, initial_tstate, 
                                disturbances,  # past H + HH disturbances
                                initial_emas,  # dict of the `{momentum_coefficient: pytree_of_running_avgs}` sort
                                batches,  # past HH batches, starting at the one that would have been used to evolve `initial_params`
                                curr_batch,  #  the current one
                                ):
    # # the scanning way
    # def _evolve(carry, batch):
    #     tstate, emas, h = carry
    #     for beta, avg in emas.items(): emas[beta] = jax.tree_map(lambda v, g: beta * v + (1 - beta) * g, avg, index_pytree(disturbances, h + H - 1))  # update emas
    #     tstate, _ = train_step(tstate, batch)
    #     params = add_pytrees(tstate.params, compute_control(cparams, slice_pytree(disturbances, h, H), emas))
    #     tstate = tstate.replace(params=params)
    #     return (tstate, emas, h + 1), None
    
    # (tstate, _, _), _ = jax.lax.scan(_evolve, (initial_tstate, initial_emas, 0), batches)
    # loss, _ = forward(tstate, curr_batch)

    # # the original way
    # tstate = initial_tstate
    # emas = initial_emas
    # for h in range(HH):
    #     # update emas or something like that, then hallucinate
    #     for beta, avg in emas.items(): emas[beta] = jax.tree_map(lambda v, g: beta * v + (1 - beta) * g, avg, index_pytree(disturbances, h + H - 1))  # update emas
    #     tstate = _hallucinate(cparams, tstate, slice_pytree(disturbances, h, H), emas, {'x': batches['x'][h], 'y': batches['y'][h]})
    # loss, _ = forward(tstate, curr_batch)
    
    # the memory-optimized way?
    tstate = initial_tstate
    emas = initial_emas
    for h in range(HH):
        # update emas or something like that, then hallucinate
        for beta, avg in emas.items(): emas[beta] = jax.tree_map(lambda v, g: beta * v + (1 - beta) * g, avg, index_pytree(disturbances, h + H - 1))  # update emas
        temp, _ = train_step(tstate, {'x': batches['x'][h], 'y': batches['y'][h]})
        del tstate
        params = add_pytrees(temp.params, compute_control(cparams, slice_pytree(disturbances, h, H), emas))
        tstate = temp.replace(params=params)
        del params, temp
    loss, _ = forward(tstate, curr_batch)
    
    return loss

_grad_fn_counterfactual = jax.grad(_compute_loss_counterfactual, (0,))

@jax.jit
def counterfactual_update(cstate,
           initial_tstate,  # tstate from HH steps ago
           disturbances,  # past H + HH disturbances
           initial_emas,  # dict of the `{momentum_coefficient: pytree_of_running_avgs}` sort
           batches,  # past HH batches, starting at the one that would have been used to evolve `initial_params`
           curr_batch,  #  the current one
          ):
    grads = _grad_fn_counterfactual(cstate.cparams, cstate.H, cstate.HH, initial_tstate, disturbances, initial_emas, batches, curr_batch)    
    updates, new_opt_state = cstate.tx.update(grads[0], cstate.opt_state, cstate.cparams)
    cparams = optax.apply_updates(cstate.cparams, updates)
    return cstate.replace(cparams=cparams, opt_state=new_opt_state)   

# -------------------------------------------------------------------------------------------------------------------------------



# -------------------------------------------------------------------------------------------------------------------------------
# ----------------------------        the non-counterfactual way ----------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------

def _roll_forward(tstate, batch, disturbances, emas):
    tstate, (loss, grads) = train_step(tstate, batch)
    # clip disturbances (K = 10 is very soft)
    K = 10; grads = jax.tree_map(lambda g: jnp.clip(g, -K, K), grads)
    
    # update disturbances and emas
    disturbances = jax.tree_map(append, disturbances, grads)
    for beta, avg in emas.items(): emas[beta] = jax.tree_map(lambda v, g: beta * v + (1 - beta) * g, avg, grads)
    return tstate, loss, disturbances, emas
    

def _compute_loss_noncounterfactual(cparams,
                                    tstate, batch,  # current state and batch
                                    disturbances,  # past H grads, including the one that we used to get to `tstate`
                                    emas,  # dict of the `{momentum_coefficient: pytree_of_running_avgs}` sort
                                    H, HH):
    
    # TODO compare this w jax.lax.scan
    for _ in range(HH):
        # roll forward and apply control
        tstate, loss, disturbances, emas = _roll_forward(tstate, batch, disturbances, emas)
        tstate = tstate.replace(params=add_pytrees(tstate.params, compute_control(cparams, disturbances, emas)))
    
    # compute stage loss
    loss, _ = forward(tstate, batch)
    
    return loss, (tstate, disturbances, emas)
    
@jax.jit
def noncounterfactual_update(cstate,
                            tstate, batch,  # current state and batch
                            disturbances,  # past H grads, including the one that we used to get to `tstate`
                            emas,  # dict of the `{momentum_coefficient: pytree_of_running_avgs}` sort
                            ):
    (loss, (tstate, disturbances, emas)), grads = jax.value_and_grad(_compute_loss_noncounterfactual, (0,), has_aux=True)(cstate.cparams, tstate, batch, disturbances, emas, cstate.H, cstate.HH)
    updates, new_opt_state = cstate.tx.update(grads[0], cstate.opt_state, cstate.cparams)
    cparams = optax.apply_updates(cstate.cparams, updates)
    return cstate.replace(cparams=cparams, opt_state=new_opt_state), (tstate, loss, disturbances, emas)

# -------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------
# --------------------   DEFINE A META-OPT WRAPPER TO MAINTAIN PARAMS/GRADS  -----------------------------------------
# --------------------------------------------------------------------------------------------------------------------

class MetaOpt:
    grad_history: jnp.ndarray
    emas: jnp.ndarray  # emas to compute controls at the current timestep (i.e. this is NOT a history)
    cstate: MetaOptGPCState
    t: int
    
    # things for counterfactual updates; these won't be used for noncounterfactual
    tstate_history: Tuple
    batch_history: Tuple
    H: int
    HH: int

    def __init__(self,
                 initial_tstate,
                 H: int, HH: int,
                 meta_optimizer,
                 m_method: str, 
                 grad_clip: float,
                 ema_keys = [], 
                 ):
        self.grad_history = jax.tree_map(lambda p: jnp.zeros((H, *p.shape)), initial_tstate.params)
        self.emas = {k: jax.tree_map(jnp.zeros_like, initial_tstate.params) for k in ema_keys}
        self.t = 0

        assert m_method in ['scalar', 'diagonal']
        self.cstate = MetaOptGPCState.create(initial_tstate, m_method, H, HH, meta_optimizer=meta_optimizer, ema_keys=ema_keys, grad_clip=grad_clip)
        
        # if we dont do counterfactual steps, these will remain unused
        self.H, self.HH = H, HH
        self.tstate_history = (None,) * (HH + 1)
        self.batch_history = None  # this will be size HH
        pass

    def noncounterfactual_step(self, tstate, batch):  
        # do HH train steps and update the controller if we have long enough histories. Otherwise, simply do HH train steps
        if self.t >= self.cstate.H:
            self.cstate, (tstate, loss, self.grad_history, self.emas) = noncounterfactual_update(self.cstate, tstate, batch, self.grad_history, self.emas)
        else: 
            for _ in range(self.cstate.HH): 
                tstate, loss, self.grad_history, self.emas = jax.jit(_roll_forward)(tstate, batch, self.grad_history, self.emas)
        self.t += self.cstate.HH
        return tstate, (loss, index_pytree(self.grad_history, -1))
    
    
    def counterfactual_step(self, 
                        tstate,  # tstate after a step of gd
                        grads,  # grads from the step of gd that resulted in `tstate`
                        batch,  # batch from step of gd that resulted in `tstate`
                        ):      
        # lazy initialize the history if it is still `None``
        if self.batch_history is None: self.batch_history = {k: jnp.repeat(v[None], self.HH, axis=0) for k, v in batch.items()}

        # clip disturbances (K = 10 is very soft)
        K = 10; grads = jax.tree_map(lambda g: jnp.clip(g, -K, K), grads)
                     
        self.grad_history = jax.tree_map(append, self.grad_history, grads)
        for beta, avg in self.emas.items(): self.emas[beta] = jax.tree_map(lambda v, g: beta * v + (1 - beta) * g, avg, grads)  # update emas

        if self.t >= self.cstate.H + self.cstate.HH:
            control = compute_control(self.cstate.cparams, slice_pytree(self.grad_history, self.cstate.HH, self.cstate.H), self.emas)  # use past H disturbances
            tstate = tstate.replace(params=add_pytrees(tstate.params, control))
            
            # compute the states of the ema buffers from `HH` steps ago by reversing the running averages
            initial_emas = {}
            if len(self.emas) > 0:
                for j in range(1, self.cstate.HH + 1):
                    grad = index_pytree(self.grad_history, self.cstate.H + self.cstate.HH - j)
                    for beta, avg in self.emas.items():
                        if j == 1: initial_emas[beta] = avg
                        initial_emas[beta] = jax.tree_map(lambda v, g: (v - (1 - beta) * g) / beta, initial_emas[beta], grad)  # reverse the emas
            self.cstate = counterfactual_update(self.cstate, self.tstate_history[0], self.grad_history, initial_emas, self.batch_history, batch)
            
        self.tstate_history = append(self.tstate_history, tstate)
        for k in self.batch_history.keys(): self.batch_history[k] = append(self.batch_history[k], batch[k]) 
        self.t += 1
        return tstate
    
    def episode_reset(self):
        H, HH = self.cstate.H, self.cstate.HH
        self.grad_history = jax.tree_map(jnp.zeros_like, self.grad_history)
        self.t = 0
        self.cstate = self.cstate.replace(opt_state=self.cstate.tx.init(self.cstate.cparams))
        self.tstate_history = (None,) * (HH + 1)
        self.batch_history = None
        return self
