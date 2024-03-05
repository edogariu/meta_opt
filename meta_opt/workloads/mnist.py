from typing import Tuple, Callable, List
from copy import deepcopy

import tensorflow as tf; tf.config.experimental.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds

import jax.numpy as jnp
import flax.linen as jnn

from .utils import cross_entropy, accuracy

# ------------------------------------------------------------------
# ------------------------- Dataset --------------------------------
# ------------------------------------------------------------------
def load_mnist(cfg, dataset_dir: str = './datasets') -> Tuple[tf.data.Dataset, tf.data.Dataset, List[int], Callable, Callable]:
    """Load MNIST train and test datasets into memory."""
    num_iters, batch_size, num_eval_iters, full_batch = cfg['num_iters'], cfg['batch_size'], cfg['num_eval_iters'], cfg['full_batch']
    
    if full_batch:
        train_ds = tfds.load('mnist', split='train', data_dir=dataset_dir)
        train_ds = train_ds.map(lambda sample: {'x': tf.cast(sample['image'],
                                                            tf.float32) / 255.,
                                            'y': sample['label']}) # normalize train set
        train_ds = train_ds.shuffle(1024).take(batch_size).cache().batch(batch_size).repeat(num_iters).ignore_errors(log_warning=False)
        test_ds = None
    else:
        train_ds = tfds.load('mnist', split='train', data_dir=dataset_dir)
        test_ds = tfds.load('mnist', split='test', data_dir=dataset_dir)
        if num_eval_iters != -1: 
            percent = min(int(100 * num_eval_iters * batch_size / len(test_ds)), 100); del test_ds
            test_ds = tfds.load('mnist', split=f'test[:{percent}%]', data_dir=dataset_dir)
        
        train_ds = train_ds.map(lambda sample: {'x': tf.cast(sample['image'],
                                                            tf.float32) / 255.,
                                            'y': sample['label']}) # normalize train set
        test_ds = test_ds.map(lambda sample: {'x': tf.cast(sample['image'],
                                                            tf.float32) / 255.,
                                            'y': sample['label']}) # normalize test set
        
        num_epochs = int(1 + (num_iters * batch_size) / len(train_ds))
        train_ds = train_ds.repeat(num_epochs).shuffle(1024).batch(batch_size, drop_remainder=True).take(num_iters).prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.shuffle(1024).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, test_ds, jnp.zeros((1, 28, 28, 1)), cross_entropy, {'loss': cross_entropy, 'acc': accuracy}  # train dataset, test dataset, unbatched input dimensions, loss function, eval metrics

# ------------------------------------------------------------------
# ------------------------------ Models ----------------------------
# ------------------------------------------------------------------
class MLP(jnn.Module):
    dims: List[int]   # dims of each layer
    activation: jnn.Module = jnn.activation.relu
    normalization: jnn.Module = None  # normalize before the activation
    drop_last_activation: bool = True
    use_bias: bool = True
        
    def setup(self):  # create the modules to build the network
        self.input_dim = self.dims[0]
        self.output_dim = self.dims[-1]
        
        layers = []
        for i, dim in enumerate(self.dims[1:]):
            layers.append(jnn.Dense(features=dim, use_bias=self.use_bias, name=f'dense {i}'))
            if self.normalization is not None: layers.append(self.normalization())
            layers.append(self.activation)
        if self.drop_last_activation: 
            layers.pop()  # removes activation from final layer
            if self.normalization is not None: layers.pop()  # removes last normalization too
        self.model = jnn.Sequential(layers)
        pass

    def __call__(self, x: jnp.ndarray, train=False):  # forward pass. train is dummy variable
        x = x.reshape(-1, self.dims[0])
        x = self.model(x)
        return x
        
class CNN(jnn.Module):
    channels: List[int]   # dims of each layer
    layer_dims: List[int] = None
    activation: jnn.Module = jnn.activation.relu
    normalization: jnn.Module = None  # normalize before the activation
    drop_last_activation: bool = True
    use_bias: bool = True
    pool_every: int = 1
        
    def setup(self):  # create the modules to build the network
        self.in_channels = self.channels[0]
        self.out_channels = self.channels[-1]
        
        layers = []
        for i, chan in enumerate(self.channels[1:]):
            layers.append(jnn.Conv(features=chan, kernel_size=(3, 3), use_bias=self.use_bias, name=f'conv {i}'))
            if self.normalization is not None: layers.append(self.normalization())
            layers.append(self.activation)
            if (i + 1) % self.pool_every == 0: layers.append(lambda x: jnn.max_pool(x, (2, 2), (2, 2)))
        if self.drop_last_activation: 
            layers.pop()  # removes activation from final layer
            if self.normalization is not None: layers.pop()  # removes last normalization too
        self.model = jnn.Sequential(layers)
        if self.layer_dims is not None: self.body = MLP(self.layer_dims)
        pass

    def __call__(self, x: jnp.ndarray, train=False):  # forward pass. train is dummy variable
        x = self.model(x)
        if self.layer_dims is not None:
            x = x.reshape((x.shape[0], -1))
            print(x.shape)
            x = self.body(x)
        return x
    