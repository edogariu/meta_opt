from algorithmic_efficiency.workloads.ogbg.ogbg_jax.workload import OgbgWorkload 
from algorithmic_efficiency.workloads.ogbg.input_pipeline import _load_dataset, _get_batch_iterator, get_dataset_iter

from typing import Tuple, Callable, List

import tensorflow as tf; tf.config.experimental.set_visible_devices([], "GPU")

import jraph
import jax
import jax.numpy as jnp

_B = OgbgWorkload()

# silly little iterator wrapper to handle data
class DS:
    def __init__(self, dataset):
        self.ds = dataset

    def __iter__(self):
        return self

    def __next__(self): 
        n = next(self.ds)
        inp = n['inputs']
        inp = jraph.GraphsTuple(
        n_node=inp.n_node[0],
        n_edge=inp.n_edge[0],
        nodes=inp.nodes[0],
        edges=inp.edges[0],
        globals=jnp.zeros((1, _B._num_outputs)),
        senders=inp.senders[0],
        receivers=inp.receivers[0])
        return {'x': inp,
                'y': {'targets': n['targets'][0], 'weights': n['weights'][0]}}

    def as_numpy_iterator(self):
       return iter(self)
    
class EvalDS:
    def __init__(self, cfg, dataset_dir, rng) -> None:
        self.data_rng = rng if rng is not None else jax.random.PRNGKey(0)
        self.dataset_dir = dataset_dir
        self.cfg = cfg
        pass

    def as_numpy_iterator(self):
        batch_size, num_eval_iters = self.cfg['batch_size'], self.cfg['num_eval_iters']
        eval_ds = _load_dataset('validation', False, self.data_rng, self.dataset_dir)
        if num_eval_iters != -1: 
            raise NotImplementedError('cant use less eval iters for GNN')
        eval_ds = eval_ds.shuffle(1024)
        eval_ds = DS(_get_batch_iterator(iter(eval_ds), batch_size))
        return eval_ds


# ------------------------------------------------------------------
# ------------------------- Dataset --------------------------------
# ------------------------------------------------------------------
def load_gnn(cfg, dataset_dir: str = './datasets', rng: int = None) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[int], Callable, Callable]:
    """Load train and test datasets into memory."""
    num_iters, batch_size, full_batch = cfg['num_iters'], cfg['batch_size'], cfg['full_batch']
    data_rng = rng if rng is not None else jax.random.PRNGKey(0)

    # get the example input
    example_input = jraph.GraphsTuple(
        n_node=jnp.asarray([1]),
        n_edge=jnp.asarray([1]),
        nodes=jnp.ones((1, 9)),
        edges=jnp.ones((1, 3)),
        globals=jnp.zeros((1, _B._num_outputs)),
        senders=jnp.asarray([0]),
        receivers=jnp.asarray([0]))

    if full_batch: 
        train_ds = _load_dataset('train', True, data_rng, dataset_dir)
        train_ds = train_ds.shuffle(1024).take(batch_size).cache().repeat(num_iters)#.ignore_errors(log_warning=False)
        train_ds = DS(_get_batch_iterator(iter(train_ds), batch_size))
        eval_ds = None
    else:
        train_ds = _load_dataset('train', True, data_rng, dataset_dir)
        # num_epochs = int(1 + (num_iters * batch_size) / len(train_ds))
        # train_ds = train_ds.repeat(num_epochs).shuffle(1024).take(num_iters * batch_size)
        train_ds = train_ds.shuffle(1024).take(num_iters * batch_size).cache()
        train_ds = DS(_get_batch_iterator(iter(train_ds), batch_size))
        eval_ds = EvalDS(cfg, dataset_dir, data_rng)

        # train_ds = _B._build_input_queue(data_rng, 'train', data_dir=dataset_dir, global_batch_size=batch_size)
        # eval_ds = _B._build_input_queue(data_rng, 'test', data_dir=dataset_dir, global_batch_size=batch_size)
        # train_ds = DS(train_ds)
        # eval_ds = DS(test_ds)

    @jax.jit
    def loss_fn(yhat, y):
        logits = yhat
        labels = y['targets']
        mask = y['weights']

        per_example_losses = _B._binary_cross_entropy_with_mask(
                labels=labels,
                logits=logits,
                mask=mask,
                label_smoothing=0.0)
        if mask is not None:
            n_valid_examples = mask.sum()
        else:
            n_valid_examples = len(per_example_losses)
        summed_loss = per_example_losses.sum()
        return summed_loss / n_valid_examples
    
    @jax.jit
    def accuracy(yhat, y):
        labels, mask = y['targets'], y['weights']
        mask = (yhat != -1)
        preds = labels > 0
        v = (preds == yhat).astype(jnp.float32)
        return jnp.where(mask, v, 0).sum() / mask.sum()
    
    return train_ds, eval_ds, example_input, loss_fn, {'loss': loss_fn, 'acc': accuracy}  # train dataset, test dataset, unbatched input dimensions, loss function, eval metrics
