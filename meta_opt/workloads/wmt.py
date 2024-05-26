import os
import shutil
from typing import Tuple, Callable, List
from collections import defaultdict

from absl import logging
logging.set_verbosity('error')
import tensorflow as tf; tf.config.experimental.set_visible_devices([], "GPU")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel('ERROR')

import tensorflow_datasets as tfds
import tensorflow_text as tftxt

import numpy as np
import jax
import jax.numpy as jnp
import ml_collections
import flax.linen as jnn
import optax

from meta_opt.workloads._wmt.input_pipeline import get_wmt_datasets
from meta_opt.workloads._wmt.models import Transformer, TransformerConfig
from meta_opt.workloads._wmt.train import initialize_cache, predict_step, tohost, per_host_sum_pmap, preferred_dtype, create_learning_rate_schedule
from meta_opt.workloads._wmt.bleu import bleu_partial, complete_bleu
from meta_opt.workloads._wmt.decode import EOS_ID
from meta_opt.workloads._wmt.default import get_small_config, get_base_config, get_big_config, get_miniscule_config, get_base_short_config
from meta_opt.workloads._wmt.train import compute_weighted_accuracy, compute_weighted_cross_entropy

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

# ================================================================================================
# this whole thing is taken and adapted from https://github.com/google/flax/tree/main/examples/wmt 
# ================================================================================================

def get_model_config(experiment_config):
    assert 'transformer_size' in experiment_config
    k = experiment_config['transformer_size']
    d = {'small': get_small_config,
         'base': get_base_config,
         'big': get_big_config,
         'miniscule': get_miniscule_config,
         'base_short': get_base_short_config,
    }
    assert k in d, f'key {k} is not a valid size'
    return d[k]()

# ------------------------------------------------------------------
# ------------------------- Dataset --------------------------------
# ------------------------------------------------------------------
def load_wmt(cfg, dataset_dir: str = './datasets') -> Tuple[tf.data.Dataset, tf.data.Dataset, List[int], Callable, Callable]:
    """Load WMT train and test datasets into memory."""
    num_iters, batch_size, num_eval_iters, full_batch, main_dir = cfg['num_iters'], cfg['batch_size'], cfg['num_eval_iters'], cfg['full_batch'], cfg['directory']
    tf.random.set_seed(cfg['seed'])
    assert os.path.isdir(f'{main_dir}/meta_opt/workloads/_wmt'), 'WMT library is in the wrong place'
    if not os.path.isfile(f'{main_dir}/meta_opt/workloads/_wmt/default.py'): 
        shutil.copyfile(f'{main_dir}/meta_opt/workloads/_wmt/configs/default.py', f'{main_dir}/meta_opt/workloads/_wmt/default.py')
        print('copied `default.py` into correct place!')
    
    config = get_model_config(cfg)
    config.num_train_steps = num_iters
    config.num_eval_steps = num_eval_iters
    config.seed = cfg['seed']
    config.per_device_batch_size = batch_size
    config.vocab_path = os.path.join(cfg['directory'], 'datasets', 'tokenizer.pth')
    
    train_ds, eval_ds, _, tokenizer = get_wmt_datasets(config, n_devices=1, vocab_path=os.path.join(cfg['directory'], 'datasets'), reverse_translation=config.reverse_translation)
    config.vocab_size = int(tokenizer.vocab_size())
    
    train_ds = train_ds.map(lambda sample: {'x': sample,
                                            'y': sample['targets']})
    eval_ds = eval_ds.map(lambda sample: {'x': sample,
                                            'y': sample['targets']})
    
    train_ds = train_ds.shuffle(1024)
    if full_batch: train_ds = train_ds.take(1).cache().repeat(num_iters)
    else: train_ds = train_ds.take(num_iters).prefetch(tf.data.AUTOTUNE)
    if num_eval_iters > 0: eval_ds = eval_ds.take(num_eval_iters)
    
    input_shape = (config.per_device_batch_size, config.max_target_length)
    example_input = jnp.ones(input_shape, jnp.float32)
    
    loss_fn = lambda yhat, y: weighted_cross_entropy(yhat, y, label_smoothing=config.label_smoothing)
    metrics = {'loss': loss_fn,
               'acc': weighted_accuracy}
    
    return train_ds, eval_ds, example_input, loss_fn, metrics, tokenizer  # train dataset, test dataset, unbatched input dimensions, loss function, eval metrics, tokenizer
    
    
# ------------------------------------------------------------------
# ------------------------------ Models ----------------------------
# ------------------------------------------------------------------


def _wmt_init(model, rngs, example_input, train: bool):
    cfg = model.config.replace(deterministic=True, decode=False)  # eval mode
    variables = jax.jit(Transformer(cfg).init)(rngs['params'], example_input, example_input)
    return variables

def _wmt_apply(model, variables, inputs, train: bool, rngs={}, mutable=[]):
    
    # X_position and X_segmentation are needed only when using "packed examples"
    # where multiple sequences are packed into the same example with this
    # metadata.
    # if such features are not present they are ignored and the example is treated
    # like a normal, unpacked sequence example.
    train_keys = [
        "inputs",
        "targets",
        "inputs_position",
        "targets_position",
        "inputs_segmentation",
        "targets_segmentation",
    ]
    (
        inputs,
        targets,
        inputs_positions,
        targets_positions,
        inputs_segmentation,
        targets_segmentation,
    ) = (inputs.get(k, None) for k in train_keys)
    
    cfg = model.config.replace(deterministic=False, decode=False) if train else model.config.replace(deterministic=True, decode=False)  # train or eval mode
    logits = Transformer(cfg).apply(
        variables,
        inputs,
        targets,
        inputs_positions=inputs_positions,
        targets_positions=targets_positions,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=targets_segmentation,
        rngs=rngs,
        mutable=mutable,
    )        
    return logits

def _decode_tokens(toks, eos_id, tokenizer):
    valid_toks = toks[: np.argmax(toks == eos_id) + 1].astype(np.int32)
    return tokenizer.detokenize(valid_toks).numpy().decode("utf-8")

def _wmt_bleu(model, tstate, predict_ds):
    """Translates the `predict_ds` and calculates the BLEU score, among other metrics."""
    
    logging.info("Translating evaluation dataset.")
    sources, references, predictions = [], [], []
    cfg = model.config.replace(deterministic=True, decode=True)
    for pred_batch in predict_ds:
        # pred_batch = jax.tree_util.tree_map(lambda x: x._numpy(), pred_batch['x'])  # pylint: disable=protected-access
        pred_batch = pred_batch['x']
        # Handle final odd-sized batch by padding instead of dropping it.
        cur_pred_batch_size = pred_batch["inputs"].shape[0]
        cache = initialize_cache(pred_batch["inputs"], cfg.max_len, cfg)  # predict mode
        predicted = predict_step(pred_batch['inputs'], tstate.params, cache, eos_id=model.eos_id, max_decode_len=cfg.max_len, config=cfg, beam_size=4)
        inputs = pred_batch["inputs"]
        targets = pred_batch["targets"]
        # Iterate through non-padding examples of batch.
        for i, s in enumerate(predicted[:cur_pred_batch_size]):
            sources.append(_decode_tokens(inputs[i], model.eos_id, model.tokenizer))
            references.append(_decode_tokens(targets[i], model.eos_id, model.tokenizer))
            predictions.append(_decode_tokens(s, model.eos_id, model.tokenizer))
    logging.info(
        "Translation: %d predictions %d references %d sources.",
        len(predictions),
        len(references),
        len(sources),
    )

    # Calculate BLEU score for translated eval corpus against reference.
    bleu_matches = bleu_partial(references, predictions)
    all_bleu_matches = per_host_sum_pmap(bleu_matches)
    bleu_score = complete_bleu(*all_bleu_matches)
    # Save translation samples for tensorboard.
    exemplars = ""
    for n in np.random.choice(np.arange(len(predictions)), 8):
        exemplars += f"{sources[n]}\n\n{references[n]}\n\n{predictions[n]}\n\n"
    # print(exemplars)
    
    return bleu_score, exemplars


class WMT(jnn.Module):
    config: ml_collections.ConfigDict
    tokenizer: tftxt.SentencepieceTokenizer
    eos_id: int
    
    def __init__(self, experiment_config, tokenizer, size: str):
        num_iters, batch_size, num_eval_iters = experiment_config['num_iters'], experiment_config['batch_size'], experiment_config['num_eval_iters']
        
        config = get_model_config(experiment_config)
        
        vocab_path = os.path.join(experiment_config['directory'], 'datasets')
        if vocab_path is None:
            vocab_path = os.path.join(vocab_path, "sentencepiece_model")
            config.vocab_path = vocab_path
        config.num_train_steps = num_iters
        config.num_eval_steps = num_eval_iters
        config.seed = experiment_config['seed']
        config.per_device_batch_size = batch_size
        vocab_size = int(tokenizer.vocab_size())
        config.vocab_size = vocab_size
        self.config = TransformerConfig(
            vocab_size=vocab_size,
            output_vocab_size=vocab_size,
            share_embeddings=config.share_embeddings,
            logits_via_embedding=config.logits_via_embedding,
            dtype=preferred_dtype(config),
            emb_dim=config.emb_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            qkv_dim=config.qkv_dim,
            mlp_dim=config.mlp_dim,
            max_len=max(config.max_target_length, config.max_eval_target_length),
            dropout_rate=config.dropout_rate,
            attention_dropout_rate=config.attention_dropout_rate,
            deterministic=False,
            decode=False,
            kernel_init=jnn.initializers.xavier_uniform(),
            bias_init=jnn.initializers.normal(stddev=1e-6),
        )
        self.eos_id = EOS_ID
        self.tokenizer = tokenizer
        pass
    
WMT.init = _wmt_init
WMT.apply = _wmt_apply
WMT.bleu = _wmt_bleu


def rsqrt(lr: float = 0.002, warmup_steps: int = 1000, b1: float = 0.9, b2: float = 0.98, weight_decay: float = 0.0):
    lr_schedule = create_learning_rate_schedule(lr, warmup_steps)
    opt = optax.inject_hyperparams(optax.adamw)(
          learning_rate=lr_schedule,
          b1=b1,
          b2=b2,
          eps=1e-8,
          weight_decay=weight_decay,
      )
    return opt
