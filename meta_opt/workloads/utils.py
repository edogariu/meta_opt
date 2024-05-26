import jax
import jax.numpy as jnp
import optax  

# ------------------------------------------------------------------
# ----------------------- Loss Functions ---------------------------
# ------------------------------------------------------------------
def cross_entropy(yhat, y):
    """Assumes `yhat` is logits and `y` is integer labels"""
    return optax.softmax_cross_entropy_with_integer_labels(logits=yhat, labels=y).mean()

def mse(yhat, y):
    return optax.l2_loss(predictions=yhat, targets=y).mean()

def accuracy(yhat, y):
    return (jnp.argmax(yhat, -1) == y).mean()