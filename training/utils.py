from typing import Tuple, Callable, List

import tensorflow as tf
import tensorflow_datasets as tfds

import jax.numpy as jnp
import flax.linen as jnn
import optax  

# ------------------------------------------------------------------
# ----------------------- Loss Functions ---------------------------
# ------------------------------------------------------------------
def cross_entropy(yhat, y):
    """Assumes `yhat` is logits and `y` is integer labels"""
    return optax.softmax_cross_entropy_with_integer_labels(logits=yhat, labels=y).mean()

def mse(yhat, y):
    return optax.l2_loss(predictions=yhat, targets=y).mean()


# ------------------------------------------------------------------
# ------------------------- Datasets -------------------------------
# ------------------------------------------------------------------
def load_mnist(num_iters: int, batch_size: int) -> Tuple[tf.data.Dataset, tf.data.Dataset, Callable, List[int]]:
    """Load MNIST train and test datasets into memory."""
    train_ds = tfds.load('mnist', split='train')
    test_ds = tfds.load('mnist', split='test')
    
    train_ds = train_ds.map(lambda sample: {'x': tf.cast(sample['image'],
                                                           tf.float32) / 255.,
                                          'y': sample['label']}) # normalize train set
    test_ds = test_ds.map(lambda sample: {'x': tf.cast(sample['image'],
                                                         tf.float32) / 255.,
                                        'y': sample['label']}) # normalize test set
    
    num_epochs = 1 + (num_iters * batch_size) // len(train_ds)
    train_ds = train_ds.repeat(num_epochs).shuffle(1024)
    train_ds = train_ds.batch(batch_size, drop_remainder=True).take(num_iters).prefetch(1)
    test_ds = test_ds.shuffle(1024)
    test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)
    
    return train_ds, test_ds, cross_entropy, [28, 28, 1]  # train dataset, test dataset, loss function, unbatched input dimensions

def load_cifar10(num_iters: int, batch_size: int) -> Tuple[tf.data.Dataset, tf.data.Dataset, Callable, List[int]]:
    """Load MNIST train and test datasets into memory."""
    train_ds = tfds.load('cifar10', split='train')
    test_ds = tfds.load('cifar10', split='test')
    
    train_ds = train_ds.map(lambda sample: {'x': tf.cast(sample['image'],
                                                           tf.float32) / 255.,
                                          'y': sample['label']}) # normalize train set
    test_ds = test_ds.map(lambda sample: {'x': tf.cast(sample['image'],
                                                         tf.float32) / 255.,
                                        'y': sample['label']}) # normalize test set
    
    num_epochs = 1 + (num_iters * batch_size) // len(train_ds)
    train_ds = train_ds.repeat(num_epochs).shuffle(1024)
    train_ds = train_ds.batch(batch_size, drop_remainder=True).take(num_iters).prefetch(1)
    test_ds = test_ds.shuffle(1024)
    test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)
    
    return train_ds, test_ds, cross_entropy, [32, 32, 3]  # train dataset, test dataset, loss function, unbatched input dimensions

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

    def __call__(self, x: jnp.ndarray):  # forward pass
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
        
    def setup(self):  # create the modules to build the network
        self.in_channels = self.channels[0]
        self.out_channels = self.channels[-1]
        
        layers = []
        for i, chan in enumerate(self.channels[1:]):
            layers.append(jnn.Conv(features=chan, kernel_size=(3, 3), use_bias=self.use_bias, name=f'conv {i}'))
            if self.normalization is not None: layers.append(self.normalization())
            layers.append(self.activation)
            layers.append(lambda x: jnn.max_pool(x, (2, 2), (2, 2)))
        if self.drop_last_activation: 
            layers.pop()  # removes activation from final layer
            if self.normalization is not None: layers.pop()  # removes last normalization too
        self.model = jnn.Sequential(layers)
        if self.layer_dims is not None: self.body = MLP(self.layer_dims)
        pass

    def __call__(self, x: jnp.ndarray):  # forward pass
        x = self.model(x)
        if self.layer_dims is not None:
            x = x.reshape((x.shape[0], -1))
            x = self.body(x)
        return x
    