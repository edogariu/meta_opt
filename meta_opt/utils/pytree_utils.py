from typing import Tuple
import jax
import jax.numpy as jnp

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

# @jax.jit
def append(arr, val):  # handle tuples as well
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
def index_pytree(pytree, idx):
    return jax.tree_map(lambda p: jax.lax.dynamic_index_in_dim(p, idx)[0], pytree)

@jax.jit
def add_pytrees(*pytrees):
    """
    add pytrees elementwise
    """
    return jax.tree_map(lambda *p: sum(p), *pytrees)

@jax.jit
def multiply_pytrees(*pytrees):
    """
    multiply pytrees elementwise
    """
    return jax.tree_map(lambda *p: jnp.prod(jnp.stack(p, axis=0), axis=0), *pytrees)

@jax.jit
def multiply_pytree_by_scalar(scalar, pytree):
    """
    multiply pytree with a scalar elementwise
    """
    return jax.tree_map(lambda p: scalar * p, pytree)

@jax.jit
def pytree_sq_norm(pytree):
    return sum(jax.tree_util.tree_flatten(jax.tree_map(lambda p: (p * p).sum(), pytree))[0])

@jax.jit
def pytree_proj(pytree, p):
    flat_pytree, unflatten_func = jax.flatten_util.ravel_pytree(pytree)
    flat_p, _ = jax.flatten_util.ravel_pytree(p)
    flat_p_normalized = flat_p / jnp.linalg.norm(flat_p)
    # print(jnp.linalg.norm(flat_pytree), jnp.linalg.norm(flat_p_normalized * jnp.linalg.norm(flat_pytree) + flat_pytree))
    flat_proj = flat_pytree + jnp.linalg.norm(flat_pytree) * flat_p_normalized
    return unflatten_func(flat_proj)
