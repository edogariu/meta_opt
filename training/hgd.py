import jax
import jax.numpy as jnp
from flax import struct

# implements hypergradient descent

class HGDState(struct.PyTreeNode):
    prev_grads: jnp.ndarray
    hypergrad: float = struct.field(pytree_node=False)  # to keep track of for plotting
    hypergrad_lr: float = struct.field(pytree_node=False)
    
    @classmethod
    def create(cls,
               hypergrad_lr: float):
        return cls(prev_grads=None, hypergrad=0., hypergrad_lr=hypergrad_lr)

# @jax.jit
def hypergrad_step(cstate, grads):
    if cstate.prev_grads is not None: hypergrad = -sum([(g1 * g2).sum() for g1, g2 in zip(jax.tree_util.tree_leaves(grads), jax.tree_util.tree_leaves(cstate.prev_grads))])
    else: hypergrad = cstate.hypergrad
    return cstate.replace(prev_grads=grads, hypergrad=hypergrad), hypergrad
    