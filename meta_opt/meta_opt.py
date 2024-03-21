from collections import deque
from typing import Tuple, Dict

import jax
import jax.numpy as jnp
import optax
from flax import struct

from meta_opt.utils.pytree_utils import append, slice_pytree, index_pytree, add_pytrees, multiply_pytrees, multiply_pytree_by_scalar
from meta_opt.nn import forward, train_step

K = 1.0  # for clipping things

# --------------------------------------------------------------------------------------------------------------------
# --------------------   DEFINE THE GPC CONTROLLER TO USE IN META-OPT  -----------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

class MetaOptGPCState(struct.PyTreeNode):
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState = struct.field(pytree_node=True)
    cparams: Dict[str, jnp.ndarray]  # dict of pytrees of disturbance-feedback control matrices (which themselves may be in pytree form if they multiply a pytree)
    
    H: int = struct.field(pytree_node=False)  # history of the controller, how many past disturbances to use for control
    HH: int = struct.field(pytree_node=False)  # history of the system, how many hallucination steps to take
    
    @classmethod
    def create(cls, 
               params,
               m_method: str,
               H: int,
               HH: int,
               meta_optimizer,
               grad_clip: float,
               dtype
               ):
        # make controller
        if m_method == 'scalar': M = jnp.zeros((H,), dtype=dtype)
        elif m_method == 'diagonal': M = jax.tree_map(lambda p: jnp.zeros((H, jnp.prod(jnp.array(p.shape))), dtype=dtype), params)
        elif m_method == 'full': M = jax.tree_map(lambda p: jnp.zeros((H, jnp.prod(jnp.array(p.shape)), jnp.prod(jnp.array(p.shape))), dtype=dtype), params)
        else: raise NotImplementedError(m_method)
        cparams = {'M': M}

        # make optimizer 
        tx = meta_optimizer
        if grad_clip is not None: tx = optax.chain(optax.clip(grad_clip), tx)  # clip grads
        opt_state = tx.init(cparams)
        
        return cls(cparams=cparams,
                   H=H, HH=HH,
                   tx=tx, opt_state=opt_state)


@jax.jit
def _compute_control_scalar(M, disturbances): return jax.tree_map(lambda d: (jax.lax.expand_dims(M, range(1, d.ndim)) * d).sum(axis=0), disturbances)

@jax.jit
def _compute_control_full(M, disturbances): 
    def fn(m, d):
        s = d[0].shape
        d = d.reshape(d.shape[0], -1)
        if m.ndim == 2: return (m * d).sum(axis=0).reshape(s)
        else: return jax.lax.batch_matmul(m, jnp.expand_dims(d, axis=-1)).sum(axis=0).reshape(s)
        
    return jax.tree_map(fn, M, disturbances)

def compute_control(cparams, disturbances):
    M = cparams['M']
    if isinstance(M, jnp.ndarray): return _compute_control_scalar(M, disturbances)
    else: return _compute_control_full(M, disturbances)

# -------------------------------------------------------------------------------------------------------------------------------
# ----------------------------        the counterfactual way --------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------

def _hallucinate(cparams, tstate, disturbances, batch):
    tstate, _ = train_step(tstate, batch)
    params = add_pytrees(tstate.params, compute_control(cparams, disturbances))
    tstate = tstate.replace(params=params)
    return tstate

def _compute_loss_counterfactual(cparams, H, HH, initial_tstate, 
                                disturbances,  # past H + HH disturbances
                                batches,  # past HH batches, starting at the one that would have been used to evolve `initial_params`
                                curr_batch,  #  the current one
                                ):
    # # the scanning way
    # def _evolve(carry, batch):
    #     tstate, h = carry
    #     temp, _ = train_step(tstate, batch)
    #     del tstate
    #     params = add_pytrees(temp.params, compute_control(cparams, slice_pytree(disturbances, h, H)))
    #     tstate = temp.replace(params=params)
    #     return (tstate, h + 1), None
    
    # x = jnp.stack(batches['x'], axis=0)
    # y = jnp.stack(batches['y'], axis=0)
    # (tstate, _), _ = jax.lax.scan(_evolve, (initial_tstate, 0), {'x': x, 'y': y})
    # loss, _ = forward(tstate, curr_batch)

    # # the original way
    # tstate = initial_tstate
    # for h in range(HH):
    #     tstate = _hallucinate(cparams, tstate, slice_pytree(disturbances, h, H), {'x': batches['x'][h], 'y': batches['y'][h]})
    # loss, _ = forward(tstate, curr_batch)
    
    # the memory-optimized way?
    tstate = initial_tstate
    for h in range(HH):
        temp, _ = train_step(tstate, {'x': batches['x'][h], 'y': batches['y'][h]})
        del tstate
        params = add_pytrees(temp.params, compute_control(cparams, slice_pytree(disturbances, h, H)))
        tstate = temp.replace(params=params)
        del params, temp
    loss, _ = forward(tstate, curr_batch)
    
    return loss

# _grad_fn_counterfactual = jax.jit(jax.grad(_compute_loss_counterfactual, (0,)), static_argnames=['H', 'HH'])
_grad_fn_counterfactual = jax.grad(_compute_loss_counterfactual, (0,))

@jax.jit
def counterfactual_update(cstate,
           initial_tstate,  # tstate from HH steps ago
           disturbances,  # past H + HH disturbances
           batches,  # past HH batches, starting at the one that would have been used to evolve `initial_params`
           curr_batch,  #  the current one
          ):
    grads = _grad_fn_counterfactual(cstate.cparams, cstate.H, cstate.HH, initial_tstate, disturbances, batches, curr_batch)    
    updates, new_opt_state = cstate.tx.update(grads[0], cstate.opt_state, cstate.cparams)
    cparams = optax.apply_updates(cstate.cparams, updates)
    return cstate.replace(cparams=cparams, opt_state=new_opt_state)   

# -------------------------------------------------------------------------------------------------------------------------------



# -------------------------------------------------------------------------------------------------------------------------------
# ----------------------------        the non-counterfactual way ----------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------

@jax.jit
def _roll_forward(tstate, batch, disturbances):
    tstate, (loss, grads) = train_step(tstate, batch)
    # clip disturbances (K = 10 is very soft)
    grads = jax.tree_map(lambda g: jnp.clip(g, -K, K), grads)
    disturbances = jax.tree_map(append, disturbances, grads)
    return tstate, loss, disturbances
    

def _compute_loss_noncounterfactual(cparams,
                                    tstate, batch,  # current state and batch
                                    disturbances,  # past H grads, including the one that we used to get to `tstate`
                                    H, HH):
    
    # TODO compare this w jax.lax.scan
    for h in range(HH):
        # roll forward and apply control
        tstate, loss, disturbances = _roll_forward(tstate, batch, disturbances)
        tstate = tstate.replace(params=add_pytrees(tstate.params, compute_control(cparams, disturbances)))
    
    # compute stage loss
    loss, _ = forward(tstate, batch)
    
    return loss, (tstate, disturbances)
    
@jax.jit
def noncounterfactual_update(cstate,
                            tstate, batch,  # current state and batch
                            disturbances,  # past H grads, including the one that we used to get to `tstate`
                            ):
    (loss, (tstate, disturbances)), grads = jax.value_and_grad(_compute_loss_noncounterfactual, (0,), has_aux=True)(cstate.cparams, tstate, batch, disturbances, cstate.H, cstate.HH)
    updates, new_opt_state = cstate.tx.update(grads[0], cstate.opt_state, cstate.cparams)
    cparams = optax.apply_updates(cstate.cparams, updates)
    return cstate.replace(cparams=cparams, opt_state=new_opt_state), (tstate, loss, disturbances)

# -------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------
# --------------------   DEFINE A META-OPT WRAPPER TO MAINTAIN PARAMS/GRADS  -----------------------------------------
# --------------------------------------------------------------------------------------------------------------------

@jax.jit
def prologue(cstate, grad_history, batch_history, tstate, grads, batch):
    if batch_history is None: batch_history = {k: [v for _ in range(cstate.HH)] for k, v in batch.items()}
    # clip disturbances (K = 10 is very soft)
    grads = jax.tree_map(lambda g: jnp.clip(g, -K, K), grads)
    grad_history = jax.tree_map(append, grad_history, grads)
    control = compute_control(cstate.cparams, slice_pytree(grad_history, cstate.HH, cstate.H))  # use past H disturbances
    tstate = tstate.replace(params=add_pytrees(tstate.params, control))
    return grad_history, batch_history, control, tstate


@jax.jit
def epilogue(tstate_history, batch_history, tstate, batch):
    tstate_history = append(tstate_history, tstate)
    for k in batch_history.keys(): batch_history[k] = append(batch_history[k], batch[k]) 
    return tstate_history, batch_history

class MetaOpt:
    grad_history: jnp.ndarray
    cstate: MetaOptGPCState
    t: int
    
    # things for counterfactual updates; these won't be used for noncounterfactual
    tstate_history: Tuple
    batch_history: Tuple
    H: int
    HH: int

    def __init__(self,
                 initial_params,
                 H: int, HH: int,
                 meta_optimizer,
                 m_method: str, 
                 grad_clip: float,
                 dtype,
                 ):
        self.grad_history = jax.tree_map(lambda p: jnp.zeros((H, *p.shape)), initial_params)
        self.t = 0

        assert m_method in ['scalar', 'diagonal', 'full']
        self.cstate = MetaOptGPCState.create(initial_params, m_method, H, HH, meta_optimizer=meta_optimizer, grad_clip=grad_clip, dtype=dtype)
        
        # if we dont do counterfactual steps, these will remain unused
        self.H, self.HH = H, HH
        self.tstate_history = (None,) * (HH + 1)
        self.batch_history = None  # this will be size HH
        pass

    def noncounterfactual_step(self, tstate, batch):  
        # do HH train steps and update the controller if we have long enough histories. Otherwise, simply do HH train steps
        if self.t >= self.cstate.H:
            self.cstate, (tstate, loss, self.grad_history) = noncounterfactual_update(self.cstate, tstate, batch, self.grad_history)
        else: 
            for _ in range(self.cstate.HH): 
                tstate, loss, self.grad_history = _roll_forward(tstate, batch, self.grad_history)
            control = compute_control(self.cstate.cparams, slice_pytree(self.grad_history, self.cstate.HH, self.cstate.H))  # use past H disturbances
            tstate = tstate.replace(params=add_pytrees(tstate.params, control))
        self.t += self.cstate.HH
        return tstate, (loss, index_pytree(self.grad_history, -1))
    
    
    def counterfactual_step(self, 
                        tstate,  # tstate after a step of gd
                        grads,  # grads from the step of gd that resulted in `tstate`
                        batch,  # batch from step of gd that resulted in `tstate`
                        ):       
    
        # # lazy initialize the history if it is still `None``
        # if self.batch_history is None: self.batch_history = {k: [v for _ in range(self.HH)] for k, v in batch.items()}

        # # clip disturbances (K = 10 is very soft)
        # K = 10; grads = jax.tree_map(lambda g: jnp.clip(g, -K, K), grads)
                     
        # self.grad_history = jax.tree_map(append, self.grad_history, grads)
        # control = compute_control(self.cstate.cparams, slice_pytree(self.grad_history, self.cstate.HH, self.cstate.H))  # use past H disturbances
        # tstate = tstate.replace(params=add_pytrees(tstate.params, control))
        
        self.grad_history, self.batch_history, control, tstate = prologue(self.cstate, self.grad_history, self.batch_history, tstate, grads, batch)
        
        if self.t >= self.cstate.H + self.cstate.HH:
            self.cstate = counterfactual_update(self.cstate, self.tstate_history[0], self.grad_history, self.batch_history, batch)
        
        self.tstate_history, self.batch_history = epilogue(self.tstate_history, self.batch_history, tstate, batch)
        
        # import sys
        # b, g, t = sys.getsizeof(self.batch_history), sys.getsizeof(self.grad_history), sys.getsizeof(self.tstate_history)
        # print(f'time={self.t}:     \n\tbatch_history={b}   \n\tgrad_history={g}      \n\ttstate_history={t}\n')
            
        # self.tstate_history = append(self.tstate_history, tstate)
        # for k in self.batch_history.keys(): self.batch_history[k] = append(self.batch_history[k], batch[k]) 
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
