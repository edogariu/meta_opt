
import os
from typing import Tuple, List, Callable

import numpy as np
import tensorflow as tf
import jax
import jax.numpy as jnp
import flax.linen as jnn
from flax.training import common_utils

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

    loss = -jnp.sum(soft_targets * jnn.log_softmax(logits), axis=-1)
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

def load_wmt(num_iters: int, batch_size: int, dataset_dir: str = './datasets') -> Tuple[tf.data.Dataset, tf.data.Dataset, List[int], Callable, Callable]:
    """Load WMT DE/EN train and test datasets into memory."""
    (train_ds, test_ds), tokenizer = _get_wmt_datasets(dataset_name='wmt14_translate/de-en',
                                                       eval_dataset_name='wmt14_translate/de-en',
                                                       dataset_dir=dataset_dir,
                                                       batch_size=batch_size,
                                                       num_iters=num_iters,
                                                       vocab_size=32000,
                                                       reverse_translation=False,
                                                       vocab_path=os.path.join(dataset_dir, 'wmt_sentencepiece_model'))
    example_input = {'inputs': jnp.zeros((batch_size, 256), dtype=int),
                     'targets': jnp.zeros((batch_size, 256), dtype=int),
                     'inputs_positions': jnp.zeros((batch_size, 256), dtype=int),
                     'targets_positions': jnp.zeros((batch_size, 256), dtype=int),
                     'inputs_segmentation': jnp.zeros((batch_size, 256), dtype=int),
                     'targets_segmentation': jnp.zeros((batch_size, 256), dtype=int),
                     }

    return train_ds, test_ds, example_input, _compute_weighted_cross_entropy, _compute_weighted_accuracy  # train dataset, test dataset, unbatched input dimensions, loss function, accuracy fn



# ------------------------------------------------------------------
# ------------------------------ Models ----------------------------
# ------------------------------------------------------------------

def make_transformer(num_heads: int,
                     num_layers: int,
                     emb_dim: int,
                     qkv_dim: int,
                     mlp_dim: int,
                     max_len: int = 256,
                     vocab_size: int = 32000,
                     dropout_rate: float = 0.1,
                     attention_dropout_rate: float = 0.1,
                     deterministic: bool = False,
                     decode: bool = False):
    
    cfg = {'vocab_size': vocab_size,
            'output_vocab_size': vocab_size,
            'share_embeddings': True,
            'logits_via_embedding': True,

            'num_heads': num_heads,
            'num_layers': num_layers,
            'emb_dim': emb_dim,
            'qkv_dim': qkv_dim,
            'mlp_dim': mlp_dim,
            'max_len': max_len,
            'dropout_rate': dropout_rate,
            'attention_dropout_rate': attention_dropout_rate,
            'deterministic': deterministic,
            'decode': decode
            }
    return _Transformer(_TransformerConfig(**cfg)) 
