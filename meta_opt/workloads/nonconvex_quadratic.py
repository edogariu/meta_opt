from typing import Any, Callable, NamedTuple, Optional, Union, Tuple, List
from copy import deepcopy

import tensorflow as tf; tf.config.experimental.set_visible_devices([], "GPU")

import jax
import jax.numpy as jnp
import flax.linen as jnn
import optax

from .utils import cross_entropy, accuracy
from meta_opt.utils.pytree_utils import pytree_sq_norm

# ------------------------------------------------------------------
# ------------------------- Dataset --------------------------------
# ------------------------------------------------------------------
def load_ncq(cfg) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[int], Callable, Callable]:    
    num_iters, seed = cfg['num_iters'], cfg['seed']
    def loss_fn(yhat, y): return yhat
    train_ds = tf.data.Dataset.random(seed=seed).take(num_iters).map(lambda sample: {'x': jnp.ones((3, 5)), 'y': jnp.zeros((3, 5))}).cache()  # empty dataset of correct format
    test_ds = tf.data.Dataset.random(seed=seed).take(1).map(lambda sample: {'x': jnp.ones((5,)), 'y': jnp.zeros((5,))}).batch(3).cache()  # empty dataset of correct format
    return train_ds, test_ds, 0, loss_fn, {'loss': loss_fn}  # train dataset, test dataset, unbatched input dimensions, loss function, eval metrics

# ------------------------------------------------------------------
# ------------------------------ Model -----------------------------
# ------------------------------------------------------------------
class NCQ(jnn.Module):
    dim: int
    std: float
    A: jnp.ndarray
    radius: float
    
    @jnn.compact
    def __call__(self, x: jnp.ndarray, train=False):  # forward pass. x and train are dummy variables
        init = lambda s: jnn.initializers.normal(stddev=self.std)(self.make_rng('params'), s)
        theta = self.variable('params', 'theta', init, (self.dim,)).value
        loss = 0.5 * (theta.T @ self.A @ theta).reshape(1)
        loss = loss[0]
        return loss
    
    
# ------------------------------------------------------------------
# ------------------------------ Optimizer -------------------------
# ------------------------------------------------------------------
def add_projection(
    radius: Union[float, jax.Array] = 1.0,
    mask: Optional[Union[Any, Callable[[optax._src.base.Params], Any]]] = None
) -> optax._src.base.GradientTransformation:
    """

    Args:
    radius: A scalar radius for the ball onto which to project.
    mask: A tree with same structure as (or a prefix of) the params PyTree,
        or a Callable that returns such a pytree given the params/updates.
        The leaves should be booleans, `True` for leaves/subtrees you want to
        apply the transformation to, and `False` for those you want to skip.

    Returns:
    A `GradientTransformation` object.
    """

    def init_fn(params):
        del params
        return optax._src.base.EmptyState()

    def update_fn(updates, state, params):
        if params is None: raise ValueError(optax._src.base.NO_PARAMS_MSG)
        
        new_params = jax.tree_util.tree_map(lambda g, p: g + p, updates, params)
        div = jnp.maximum(1., (pytree_sq_norm(new_params) / radius))
        updates = jax.tree_util.tree_map(lambda n, p: (n / div) - p, new_params, params)
        return updates, state

    # If mask is not `None`, apply mask to the gradient transformation.
    # E.g. it is common to skip weight decay on bias units and batch stats.
    if mask is not None:
        return wrappers.masked(
            optax._src.base.GradientTransformation(init_fn, update_fn), mask)
    return optax._src.base.GradientTransformation(init_fn, update_fn)
        