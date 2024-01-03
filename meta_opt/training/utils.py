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

    def __call__(self, x: jnp.ndarray):  # forward pass
        x = self.model(x)
        if self.layer_dims is not None:
            x = x.reshape((x.shape[0], -1))
            print(x.shape)
            x = self.body(x)
        return x
    
class VGG(jnn.Module):
    stages: List[List[int]]   # channels of each stage for all the stages
    layer_dims: List[int] = None  # layer dims for the mlp
    activation: jnn.Module = jnn.activation.relu
    drop_last_activation: bool = True
    use_bias: bool = True
    dropout: float = 0.1
    
    
    @jnn.compact
    def __call__(self, x, train: bool):
        for i, stage in enumerate(self.stages):
            for j, chan in enumerate(stage):
                x = jnn.Conv(features=chan, kernel_size=(3, 3), use_bias=self.use_bias, name=f'conv {i},{j}')(x)
                x = self.activation(x)
            x = jnn.max_pool(x, (2, 2), (2, 2))
            x = jnn.Dropout(self.dropout, deterministic=not train)(x)
        x = x.reshape((x.shape[0], -1))
        for j, l in enumerate(self.layer_dims):
            x = jnn.Dense(features=l, use_bias=self.use_bias)(x)
            if j != len(self.layer_dims) - 1 or not self.drop_last_activation:
                x = self.activation(x)    
                x = jnn.Dropout(self.dropout, deterministic=not train)(x)
        return x
    
4# ----------------------------------------------------------------------------
# TAKEN DIRECTLY FROM https://github.com/fattorib/Flax-ResNets/tree/master
    
import jax
from typing import Any, Callable, Sequence, Optional
from jax import lax, random, numpy as jnp
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
import copy
from functools import partial
import numpy as np

ModuleDef = Any
dtypedef = Any


class ResidualBlock(nn.Module):
    # Define collection of datafields here
    in_channels: int

    # For batchnorm, you can pass it as a ModuleDef
    norm: ModuleDef

    # dtype for fp16/32 training
    dtype: dtypedef = jnp.float32

    # define init for conv layers
    kernel_init: Callable = nn.initializers.kaiming_normal()

    @nn.compact
    def __call__(self, x):
        residual = x

        x = nn.Conv(
            kernel_size=(3, 3),
            strides=1,
            features=self.in_channels,
            padding="SAME",
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
        )(x)
        x = self.norm()(x)
        x = nn.relu(x)
        x = nn.Conv(
            kernel_size=(3, 3),
            strides=1,
            features=self.in_channels,
            padding="SAME",
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
        )(x)
        x = self.norm()(x)

        x = x + residual

        return nn.relu(x)


class DownSampleResidualBlock(nn.Module):
    # Define collection of datafields here
    in_channels: int
    out_channels: int

    # For batchnorm, you can pass it as a ModuleDef
    norm: ModuleDef

    # dtype for fp16/32 training
    dtype: dtypedef = jnp.float32

    # define init for conv layers
    kernel_init: Callable = nn.initializers.kaiming_normal()

    @nn.compact
    def __call__(self, x):
        residual = x

        x = nn.Conv(
            kernel_size=(3, 3),
            strides=1,
            features=self.in_channels,
            padding="SAME",
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
        )(x)
        x = self.norm()(x)
        x = nn.relu(x)
        x = nn.Conv(
            kernel_size=(3, 3),
            strides=(2, 2),
            features=self.out_channels,
            padding=((1, 1), (1, 1)),
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
        )(x)
        x = self.norm()(x)

        x = x + self.pad_identity(residual)

        return nn.relu(x)

    @nn.nowrap
    def pad_identity(self, x):
        # Pad identity connection when downsampling
        return jnp.pad(
            x[:, ::2, ::2, ::],
            ((0, 0), (0, 0), (0, 0), (self.out_channels // 4, self.out_channels // 4)),
            "constant",
        )


class ResNet(nn.Module):
    # Define collection of datafields here
    filter_list: Sequence[int]
    N: int
    num_classes: int

    # dtype for fp16/32 training
    dtype: dtypedef = jnp.float32

    # define init for conv and linear layers
    kernel_init: Callable = nn.initializers.kaiming_normal()

    # For train/test differences, want to pass “mode switches” to __call__
    @nn.compact
    def __call__(self, x, train=True):

        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.1,
            epsilon=1e-5,
            dtype=self.dtype,
        )
        x = nn.Conv(
            kernel_size=(3, 3),
            strides=1,
            features=self.filter_list[0],
            padding="SAME",
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
        )(x)

        x = norm()(x)
        x = nn.relu(x)

        # First stage
        for _ in range(0, self.N - 1):
            x = ResidualBlock(
                in_channels=self.filter_list[0], norm=norm, dtype=self.dtype
            )(x)

        x = DownSampleResidualBlock(
            in_channels=self.filter_list[0],
            out_channels=self.filter_list[1],
            norm=norm,
            dtype=self.dtype,
        )(x)

        # Second stage
        for _ in range(0, self.N - 1):
            x = ResidualBlock(
                in_channels=self.filter_list[1], norm=norm, dtype=self.dtype
            )(x)

        x = DownSampleResidualBlock(
            in_channels=self.filter_list[1],
            out_channels=self.filter_list[2],
            norm=norm,
            dtype=self.dtype,
        )(x)

        # Third stage
        for _ in range(0, self.N):
            x = ResidualBlock(
                in_channels=self.filter_list[2], norm=norm, dtype=self.dtype
            )(x)

        # Global pooling
        x = jnp.mean(x, axis=(1, 2))

        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(
            features=self.num_classes, kernel_init=self.kernel_init, dtype=self.dtype
        )(x)

        return x


def _resnet(layers, N, dtype=jnp.float32, num_classes=10):
    model = ResNet(filter_list=layers, N=N, dtype=dtype, num_classes=num_classes)
    return model


def ResNet20(
    dtype=jnp.float32,
):
    return _resnet(layers=[16, 32, 64], N=3, dtype=dtype, num_classes=10)

# ------------------------------------------------------------------------
