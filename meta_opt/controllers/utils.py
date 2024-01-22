from typing import List, Tuple
from scipy.linalg import solve_discrete_are
import jax
import jax.numpy as jnp

@jax.jit
def quad_loss(x: jnp.ndarray, u: jnp.ndarray) -> float:
    """
    Quadratic loss.

    Args:
        x (jnp.ndarray):
        u (jnp.ndarray):

    Returns:
        Real
    """
    return jnp.sum(x.T @ x + u.T @ u)

def clip(x: jnp.ndarray, max_norm: float):
    norm = jnp.linalg.norm(x)  
    x *= jnp.minimum(1., max_norm / norm)
    return x

def dare_gain(A: jnp.ndarray,
              B: jnp.ndarray,
              Q: jnp.ndarray = None,
              R: jnp.ndarray = None):
    if Q is None: Q = jnp.eye(B.shape[0])
    if R is None: R = jnp.eye(B.shape[1])
    P = solve_discrete_are(A, B, Q, R)
    K = jnp.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)  # compute LQR gain
    return -K

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

def append(arr, val):  # handle tuples as well
    if isinstance(arr, jnp.ndarray): return _append(arr, val)
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

# @jax.jit
def index_pytree(pytree, idx):
    return jax.tree_map(lambda p: jax.lax.dynamic_index_in_dim(p, idx)[0], pytree)

# @jax.jit
def add_pytrees(*pytrees):
    """
    add pytrees elementwise
    """
    return jax.tree_map(lambda *p: sum(p), *pytrees)

# @jax.jit
def multiply_pytrees(*pytrees):
    """
    multiply pytrees elementwise
    """
    return jax.tree_map(lambda *p: jnp.prod(jnp.stack(p, axis=0), axis=0), *pytrees)

# @jax.jit
def multiply_pytree_by_scalar(scalar, pytree):
    """
    multiply pytree with a scalar elementwise
    """
    return jax.tree_map(lambda p: scalar * p, pytree)
