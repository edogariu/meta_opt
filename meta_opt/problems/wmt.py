
from typing import Tuple, List, Callable

import tensorflow as tf
import jax
import jax.numpy as jnp

from ._config import get_config as _get_config
from ._input_pipeline import get_wmt_datasets as _get_wmt_datasets
from ._models import Transformer as _Transformer
from ._models import TransformerConfig as _TransformerConfig

# this is  flax's example WMT transformer problem taken from https://github.com/google/flax/blob/main/examples/wmt/train.py

# ------------------------------------------------------------------
# ------------------------ Loss/Accuracy Fns -----------------------
# ------------------------------------------------------------------

def _compute_weighted_cross_entropy(
    logits, targets, weights=None, label_smoothing=0.0
):
    """Compute weighted cross entropy and entropy for log probs and targets.

    Args:
    logits: [batch, length, num_classes] float array.
    targets: categorical targets [batch, length] int array.
    weights: None or array of shape [batch, length].
    label_smoothing: label smoothing constant, used to determine the on and off
        values.

    Returns:
    Tuple of scalar loss and batch normalizing factor.
    """
    if logits.ndim != targets.ndim + 1:
        raise ValueError(
            "Incorrect shapes. Got shape %s logits and %s targets"
            % (str(logits.shape), str(targets.shape))
        )
    vocab_size = logits.shape[-1]
    confidence = 1.0 - label_smoothing
    low_confidence = (1.0 - confidence) / (vocab_size - 1)
    normalizing_constant = -(
        confidence * jnp.log(confidence)
        + (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20)
    )
    soft_targets = common_utils.onehot(
        targets, vocab_size, on_value=confidence, off_value=low_confidence
    )

    loss = -jnp.sum(soft_targets * nn.log_softmax(logits), axis=-1)
    loss = loss - normalizing_constant

    normalizing_factor = np.prod(targets.shape)
    if weights is not None:
        loss = loss * weights
        normalizing_factor = weights.sum()

    return loss.sum() / normalizing_factor


def _compute_weighted_accuracy(logits, targets, weights=None):
    """Compute weighted accuracy for log probs and targets.

    Args:
    logits: [batch, length, num_classes] float array.
    targets: categorical targets [batch, length] int array.
    weights: None or array of shape [batch, length]

    Returns:
    Tuple of scalar loss and batch normalizing factor.
    """
    if logits.ndim != targets.ndim + 1:
        raise ValueError(
            "Incorrect shapes. Got shape %s logits and %s targets"
            % (str(logits.shape), str(targets.shape))
        )
    loss = jnp.equal(jnp.argmax(logits, axis=-1), targets)
    normalizing_factor = np.prod(logits.shape[:-1])
    if weights is not None:
        loss = loss * weights
        normalizing_factor = weights.sum()

    return loss.sum() / normalizing_factor

# ------------------------------------------------------------------
# ------------------------------ Dataset ---------------------------
# ------------------------------------------------------------------

def load_wmt(num_iters: int, batch_size: int) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[int], Callable, Callable]:
    """Load WMT DE/EN train and test datasets into memory."""
    train_ds, test_ds = _get_wmt_datasets(dataset_name='wmt14_translate/de-en',
                                          eval_dataset_name='wmt14_translate/de-en',
                                          batch_size=batch_size,
                                          vocab_size=32000,
                                          reverse_translation=False,
                                          vocab_path=None)

    # check normalization and shit to ensure dataset is good to go
        
    # num_epochs = 1 + (num_iters * batch_size) // len(train_ds)
    # train_ds = train_ds.repeat(num_epochs).shuffle(1024)
    # train_ds = train_ds.batch(batch_size, drop_remainder=True).take(num_iters).prefetch(1)
    # test_ds = test_ds.shuffle(1024)
    # test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)
    
    return train_ds, test_ds, [32, 32, 3], _compute_weighted_cross_entropy, c_ompute_weighted_accuracy  # train dataset, test dataset, unbatched input dimensions, loss function, accuracy fn



# ------------------------------------------------------------------
# ------------------------------ Models ----------------------------
# ------------------------------------------------------------------

# def make_transformer(...):
#     return _Transformer(_TransformerConfig(...)) 

    
@jax.jit
def forward_and_backward(tstate, batch):
    inputs, inputs_positions, targets_positions, inputs_segementation, targets_segmentation, targets = (batch[k] for k in ('inputs',
                                                                                                                           'inputs_positions',
                                                                                                                           'targets_positions',
                                                                                                                           'inputs_segementation',
                                                                                                                           'targets_segmentation',
                                                                                                                           'targets'))
    
    if tstate.rng is not None:
        next_key, dropout_key = jax.random.split(tstate.rng)
        tstate = tstate.replace(rng=next_key)
    else:
        dropout_key = None

    # define grad fn
    def loss_fn(params):
        yhat = _Transformer(tstate.model.replace(deterministic=False)).apply(
            {"params": params},
            inputs,
            targets,
            inputs_positions=inputs_positions,
            targets_positions=targets_positions,
            inputs_segmentation=inputs_segmentation,
            targets_segmentation=targets_segmentation,
            rngs={"dropout": dropout_key},
        )
        loss = tstate.loss_fn(yhat, targets)
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn)

    # get loss and grads
    loss, grads = grad_fn(tstate.params)

    return tstate, (loss, grads)

@jax.jit
def forward(tstate, batch):
    inputs, inputs_positions, targets_positions, inputs_segementation, targets_segmentation, targets = (batch[k] for k in ('inputs',
                                                                                                                           'inputs_positions',
                                                                                                                           'targets_positions',
                                                                                                                           'inputs_segementation',
                                                                                                                           'targets_segmentation',
                                                                                                                           'targets'))
    yhat = _Transformer(tstate.model.replace(deterministic=True)).apply(
            {"params": tstate.params},
            inputs,
            targets,
            inputs_positions=inputs_positions,
            targets_positions=targets_positions,
            inputs_segmentation=inputs_segmentation,
            targets_segmentation=targets_segmentation,
        )
    loss = tstate.loss_fn(yhat, targets)
    return loss

@jax.jit
def eval(tstate, batch):
    inputs, inputs_positions, targets_positions, inputs_segementation, targets_segmentation, targets = (batch[k] for k in ('inputs',
                                                                                                                           'inputs_positions',
                                                                                                                           'targets_positions',
                                                                                                                           'inputs_segementation',
                                                                                                                           'targets_segmentation',
                                                                                                                           'targets'))
    yhat = _Transformer(tstate.model.replace(deterministic=True)).apply(
            {"params": tstate.params},
            inputs,
            targets,
            inputs_positions=inputs_positions,
            targets_positions=targets_positions,
            inputs_segmentation=inputs_segmentation,
            targets_segmentation=targets_segmentation,
        )
    loss = tstate.loss_fn(yhat, targets)
    acc = tstate.acc_fn(yhat, targets)
    return loss, acc
