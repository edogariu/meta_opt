from typing import Tuple
import jax
import jax.numpy as jnp

# --------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------   Some Jax Utils  --------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

@jax.jit
def _append(arr, val):
    """
    rightmost recent appending, i.e. arr = (val_{t-h}, ..., val_{t-1}, val_t)
    """
    # if not isinstance(val, jnp.ndarray):
    #     val = jnp.array(val, dtype=arr.dtype)
    arr = arr.at[0].set(val)
    arr = jnp.roll(arr, -1, axis=0)
    return arr

def append(arr, val):  # handle tuples and lists as well
    if isinstance(arr, jnp.ndarray): return _append(arr, val)
    elif isinstance(arr, list): 
        arr.append(val)
        return arr[1:]
    elif isinstance(arr, Tuple):
        arr = arr[1:] + (val,)
        return arr
    else:
        raise NotImplementedError(arr.__class__)
    
def _slice_pytree(pytree, start_idx, slice_size):
    """
    Slice a pytree leafwise
    """
    return jax.tree_map(lambda p: jax.lax.dynamic_slice_in_dim(p, start_idx, slice_size), pytree)
slice_pytree = jax.jit(_slice_pytree, static_argnums=(2,))

@jax.jit
def add_pytrees(*pytrees):
    """
    add pytrees elementwise
    """
    return jax.tree_map(lambda *p: sum(p), *pytrees)

@jax.jit
def sq_norm_pytree(pytree):
    """
    flattens pytree to a vector and computes the squared norm 
    """
    return sum(jax.tree_util.tree_flatten(jax.tree_map(lambda p: (p * p).sum(), pytree))[0])
