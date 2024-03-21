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
    def loss_fn(yhat, y): 
        A, theta = yhat['A'], yhat['theta']
        norm = jnp.linalg.norm(theta)
        n = theta / jnp.maximum(1.0, norm)
        loss = 0.5 * (n.T @ A @ n).reshape(1)
        loss = loss[0]
        return loss
    
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
        return {'A': self.A, 'theta': theta}    
        