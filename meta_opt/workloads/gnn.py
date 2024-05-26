from algorithmic_efficiency.workloads.ogbg.ogbg_jax.workload import OgbgWorkload 
from algorithmic_efficiency.workloads.ogbg.input_pipeline import _load_dataset, _get_batch_iterator

from typing import Tuple, Callable, List

import tensorflow as tf; tf.config.experimental.set_visible_devices([], "GPU")

import jax
import jax.numpy as jnp

_B = OgbgWorkload()

class DS:
    def __init__(self, dataset):
        self.ds = dataset

    def __iter__(self):
        return iter(self.ds)

    def __next__(self): 
        ret = next(self.ds)
        ret['x'] = ret['inputs']
        ret['y'] = jnp.where(ret['weights'], ret['targets'], -1)
        return ret

# ------------------------------------------------------------------
# ------------------------- Dataset --------------------------------
# ------------------------------------------------------------------
def load_gnn(cfg, dataset_dir: str = './datasets', seed: int = 0) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[int], Callable, Callable]:
    """Load MNIST train and test datasets into memory."""
    num_iters, batch_size, num_eval_iters, full_batch = cfg['num_iters'], cfg['batch_size'], cfg['num_eval_iters'], cfg['full_batch']
    
    data_rng = jax.random.PRNGKey(seed)

    # get the example input
    example_ds = _load_dataset('train', True, data_rng, dataset_dir)
    example_input = next(_get_batch_iterator(iter(example_ds), 4))['inputs']

    if full_batch: 
        train_ds = _load_dataset('train', False, data_rng, dataset_dir)
        train_ds = train_ds.shuffle(1024).take(batch_size).cache().repeat(num_iters).ignore_errors(log_warning=False)
        train_ds = _get_batch_iterator(iter(train_ds), batch_size) 
        train_ds = DS(train_ds)
        test_ds = None
    else:
        train_ds = _load_dataset('train', False, data_rng, dataset_dir)
        test_ds = _load_dataset('test', False, data_rng, dataset_dir)
        if num_eval_iters != -1: 
            raise NotImplementedError('cant use less eval iters for GNN')
        
        num_epochs = int(1 + (num_iters * batch_size) / len(train_ds))
        train_ds = train_ds.repeat(num_epochs).shuffle(1024).take(num_iters * batch_size)
        train_ds = _get_batch_iterator(iter(train_ds), batch_size) 
        test_ds = test_ds.shuffle(1024)
        test_ds = _get_batch_iterator(iter(test_ds), batch_size) 

        train_ds = DS(train_ds)
        test_ds = DS(test_ds)

    def loss_fn(yhat, y):
        ret =  _B.loss_fn(yhat, y)
        return ret['summed'] / ret['n_valid_examples']
    
    @jax.jit
    def accuracy(yhat, y):
        print(yhat)
        mask = (yhat != -1)
        preds = y > 0
        return (preds == yhat).astype(jnp.float32)[mask].mean()
    
    return train_ds, test_ds, example_input, loss_fn, {'loss': loss_fn, 'acc': accuracy}  # train dataset, test dataset, unbatched input dimensions, loss function, eval metrics


# Forked from the init2winit implementation here
# https://github.com/google/init2winit/blob/master/init2winit/model_lib/gnn.py.
from typing import Optional, Tuple

from flax import linen as nn
import jax.numpy as jnp
import jraph


def _make_embed(latent_dim, name):

  def make_fn(inputs):
    return nn.Dense(features=latent_dim, name=name)(inputs)

  return make_fn


def _make_mlp(hidden_dims, dropout, activation_fn):
  """Creates a MLP with specified dimensions."""

  def make_fn(*inputs):
    x = jnp.concatenate([inp.reshape(-1) for inp in inputs], axis=0).reshape(1, -1)
    for dim in hidden_dims:
      x = nn.Dense(features=dim)(x)
      x = nn.LayerNorm()(x)
      x = activation_fn(x)
      x = dropout(x)
    return x
      
  # @jraph.concatenated_args
  # def make_fn(inputs):
  #   x = inputs
  #   for dim in hidden_dims:
  #     x = nn.Dense(features=dim)(x)
  #     x = nn.LayerNorm()(x)
  #     x = activation_fn(x)
  #     x = dropout(x)
  #   return x

  return make_fn


class GNN(nn.Module):
  """Defines a graph network.
  The model assumes the input data is a jraph.GraphsTuple without global
  variables. The final prediction will be encoded in the globals.
  """
  num_outputs: int = _B._num_outputs
  latent_dim: int = 32  # 256
  hidden_dims: Tuple[int] = (32,)  # 256
  # If None, defaults to 0.1.
  dropout_rate: Optional[float] = 0.1
  num_message_passing_steps: int = 2  # 5
  activation_fn_name: str = 'relu'

  @nn.compact
  def __call__(self, graph, train):
    if self.dropout_rate is None:
      dropout_rate = 0.1
    else:
      dropout_rate = self.dropout_rate
    dropout = nn.Dropout(rate=dropout_rate, deterministic=not train)

    graph = graph._replace(
        globals=jnp.zeros([graph.n_node.shape[0], self.num_outputs]))

    embedder = jraph.GraphMapFeatures(
        embed_node_fn=_make_embed(self.latent_dim, name='node_embedding'),
        embed_edge_fn=_make_embed(self.latent_dim, name='edge_embedding'))
    graph = embedder(graph)

    if self.activation_fn_name == 'relu':
      activation_fn = nn.relu
    elif self.activation_fn_name == 'gelu':
      activation_fn = nn.gelu
    elif self.activation_fn_name == 'silu':
      activation_fn = nn.silu
    else:
      raise ValueError(
          f'Invalid activation function name: {self.activation_fn_name}')

    for _ in range(self.num_message_passing_steps):
      net = jraph.GraphNetwork(
          update_edge_fn=_make_mlp(
              self.hidden_dims, dropout=dropout, activation_fn=activation_fn),
          update_node_fn=_make_mlp(
              self.hidden_dims, dropout=dropout, activation_fn=activation_fn),
          update_global_fn=_make_mlp(
              self.hidden_dims, dropout=dropout, activation_fn=activation_fn))
      print(graph.globals.shape)
      graph = net(graph)

    # Map globals to represent the final result
    decoder = jraph.GraphMapFeatures(embed_global_fn=nn.Dense(self.num_outputs))
    graph = decoder(graph)

    return graph.globals