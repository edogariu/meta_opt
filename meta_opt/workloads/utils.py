import jax.numpy as jnp
import optax  

from meta_opt.workloads._wmt.train import compute_weighted_accuracy, compute_weighted_cross_entropy

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

@jax.jit
def weighted_accuracy(logits, targets):
    weights = jnp.where(targets > 0, 1, 0).astype(jnp.float32)
    acc, n = compute_weighted_accuracy(logits, targets, weights=weights)
    return acc / n

@jax.jit
def weighted_cross_entropy(logits, targets, label_smoothing=0.0):
    weights = jnp.where(targets > 0, 1, 0).astype(jnp.float32)
    loss, weight_sum = compute_weighted_cross_entropy(logits, targets, weights, label_smoothing=label_smoothing)
    mean_loss = loss / weight_sum
    return mean_loss