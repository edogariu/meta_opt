import functools
from typing import Callable

import numpy as np
import jax
import optax

from meta_opt.optimizers.base import OptimizerConfig
from ._functions import Hypersphere, \
                        Hyperellipsoid, \
                        Rosenbrock, \
                        Rastrigin, \
                        Schwefel, \
                        Griewank, \
                        Ackley, \
                        Michalewicz, \
                        EggHolder, \
                        Keane, \
                        Rana, \
                        DeJong3, \
                        StyblinskiTang

FNS = [Hypersphere, Hyperellipsoid, Rosenbrock, Rastrigin, Schwefel, Griewank, Ackley, Michalewicz, EggHolder, Keane, Rana, DeJong3, StyblinskiTang]

def normalize(fn):
    """returns a callable x -> loss such that x \in [-1, 1]^d and loss \in [0, 1]"""
    b = np.array(fn.suggested_bounds())
    x_scale = b[1] - b[0]
    x_shift = b[0]
    min, max = fn.minimum(), fn.maximum()
    if min is None:
        # print(fn.name(), 'had no minimum!')
        min = 0.
    else:
        min = min[0]
    if max is None: 
        # print(fn.name(), 'had no maximum!')
        max = 1.
    else:
        max = max[0]
    y_scale = (1e-6 + max - min)
    y_shift = min

    def out_fn(x):
        # x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        # x = x_scale * x + x_shift  # [0, 1] -> any
        loss = fn(x, validate=False)
        # loss = (loss - y_shift) / y_scale  # any -> [0, 1]
        return loss
    return jax.tree_util.Partial(jax.jit(out_fn))

# @functools.partial(jax.jit, static_argnums=[3,])
def train(iterate, opt, loss_fn, num_iters) -> float:
    opt_state = opt.init(iterate)
    if hasattr(opt_state[0], 'cost_fn_history'):
        opt_state = (opt_state[0].replace(cost_fn_history=(loss_fn,) * opt_state[0].HH), opt_state[1])
    def train_step(carry, _):
        (x, opt_state) = carry
        loss, grads = jax.value_and_grad(loss_fn)(x)
        updates, opt_state = opt.update(grads, opt_state, x, cost_fn=loss_fn)
        x = optax.apply_updates(x, updates)
        return (x, opt_state), loss
    _, losses = jax.lax.scan(train_step, (iterate, opt_state), length=num_iters)
    return losses


@functools.partial(jax.jit, static_argnames=('d', 'num_iters'))
def _test_opt(seed: int,
              make_optimizer_cfg: Callable[[], OptimizerConfig],
              d: int,
              num_iters: int,):    
    rng = jax.random.PRNGKey(seed)
    optimizer_cfg = make_optimizer_cfg()

    rets = {}
    for fn_cls in FNS:
        # get loss fn and initialize iterate and opt
        rng, init_rng = jax.random.split(rng)
        iterate = jax.random.normal(init_rng, (d,)) / 2
        opt = optimizer_cfg.make_jax()
        losses = train(iterate=iterate, 
                       opt=opt, 
                       loss_fn=normalize(fn_cls(n_dimensions=d)), 
                       num_iters=num_iters)
        rets[fn_cls.__name__] = losses
    return rets

def test_opt(seed: int,
             optimizer_cfg: OptimizerConfig,
             d: int,
             num_iters: int,):
    """Runs a suite of benchmark optimization functions (forked from https://gitlab.com/luca.baronti/python_benchmark_functions and defined in `_functions.py`).

    Args:
        seed (int): seed for the random iterate initializations
        optimizer_cfg (OptimizerConfig): optimizer config to test
        d (int): dimension of the optimization problem
        num_iters (int): number of iterations to optimize for.

    Returns:
        Dict[str, jax.Array]: loss arrays (each of size num_iters) for each function in the suite
    """
    return _test_opt(seed=seed, make_optimizer_cfg=jax.tree_util.Partial(lambda: optimizer_cfg), d=d, num_iters=num_iters)