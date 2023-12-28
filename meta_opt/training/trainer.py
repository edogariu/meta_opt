from typing import Tuple, List, Callable
import functools

import jax
import jax.numpy as jnp

import flax.linen as jnn
from flax import struct
from flax.training import train_state

class TrainState(train_state.TrainState):
    # things that dont change
    loss_fn: Callable[[Tuple[jnp.ndarray, jnp.ndarray]], float]
    model: jnn.Module = struct.field(pytree_node=False)
    input_dims: List[int] = struct.field(pytree_node=False)  # dimensions that the model takes as input

def reset_model(rng, tstate: TrainState):
    params = tstate.model.init(rng, jnp.ones([1, *tstate.input_dims]))['params'] # initialize parameters by passing a template input
    opt_state = tstate.tx.init(params)
    tstate = tstate.replace(params=params, opt_state=opt_state)
    return tstate

def create_train_state(rng, model: jnn.Module, input_dims: List[int], optimizer, loss_fn):
    """Creates an initial `TrainState`."""
    tstate = TrainState.create(model=model, 
                               apply_fn=model.apply, 
                               params={}, 
                               tx=optimizer,
                               loss_fn=jax.tree_util.Partial(loss_fn), 
                               input_dims=input_dims)
    return reset_model(rng, tstate)




@jax.jit
def forward_and_backward(tstate, batch):
    y = batch['y']

    # define grad fn
    def loss_fn(params):
        yhat = tstate.apply_fn({'params': params}, batch['x'])
        loss = tstate.loss_fn(yhat, y)
        return loss
    grad_fn = jax.value_and_grad(loss_fn)

    # get loss and grads
    loss, grads = grad_fn(tstate.params)

    return tstate, (loss, grads)

@jax.jit
def forward(tstate, batch):
    yhat = tstate.apply_fn({'params': tstate.params}, batch['x'])
    loss = tstate.loss_fn(yhat, batch['y'])
    return loss

@jax.jit
def apply_gradients(tstate, grads):
    tstate = tstate.apply_gradients(grads=grads)
    return tstate

@jax.jit
def gradient_descent(tstate, batch):
    tstate, (loss, grads) = forward_and_backward(tstate, batch)
    tstate = apply_gradients(tstate, grads)
    return tstate, (loss, grads)

@jax.jit
def p_gradient_descent(params, batch, tstate):
    y = batch['y']

    # define grad fn
    def loss_fn(p):
        yhat = tstate.apply_fn({'params': p}, batch['x'])
        loss = tstate.loss_fn(yhat, y)
        return loss
    grad_fn = jax.value_and_grad(loss_fn)

    # get loss and grads
    loss, grads = grad_fn(params)
    tstate = tstate.apply_gradients(grads=grads)
    params = tstate.params
    return params, (loss, grads, tstate)
    

@jax.jit
def value_and_jacfwd(f, x):
    pushfwd = functools.partial(jax.jvp, f, (x,))
    basis = jnp.eye(x.size, dtype=x.dtype)
    y, jac = jax.vmap(pushfwd, out_axes=(None, 1))((basis,))
    return y, jac

@jax.jit
def value_and_jacrev(f, *x):
    y, pullback = jax.vjp(f, *x)
    basis = jnp.eye(y.size, dtype=y.dtype)
    jac = jax.vmap(pullback)(basis)
    return y, jac
