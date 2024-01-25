from typing import Tuple, Callable, List

import tensorflow as tf; tf.config.experimental.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds

import jax.numpy as jnp
import flax.linen as jnn
from .utils import cross_entropy, accuracy

# ------------------------------------------------------------------
# ------------------------- Dataset --------------------------------
# ------------------------------------------------------------------
def load_cifar10(cfg, dataset_dir: str = './datasets') -> Tuple[tf.data.Dataset, tf.data.Dataset, List[int], Callable, Callable]:
    """Load CIFAR-10 train and test datasets into memory."""
    num_iters, batch_size, num_eval_iters = cfg['num_iters'], cfg['batch_size'], cfg['num_eval_iters']
    train_ds = tfds.load('cifar10', split='train', data_dir=dataset_dir)
    test_ds = tfds.load('cifar10', split='test', data_dir=dataset_dir)
    if num_eval_iters != -1: 
        percent = min(int(100 * num_eval_iters * batch_size / len(test_ds)) + 1, 100)
        test_ds = tfds.load('mnist', split=f'test[:{percent}%]', data_dir=dataset_dir)
    
    train_ds = train_ds.map(lambda sample: {'x': tf.cast(sample['image'],
                                                           tf.float32) / 255.,
                                          'y': sample['label']}) # normalize train set
    test_ds = test_ds.map(lambda sample: {'x': tf.cast(sample['image'],
                                                         tf.float32) / 255.,
                                        'y': sample['label']}) # normalize test set
    
    num_epochs = 1 + (num_iters * batch_size) // len(train_ds)
    train_ds = train_ds.repeat(num_epochs).shuffle(1024).batch(batch_size, drop_remainder=True).take(num_iters).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size, drop_remainder=True).shuffle(1024).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, test_ds, jnp.zeros((1, 32, 32, 3)), cross_entropy, {'loss': cross_entropy, 'acc': accuracy}  # train dataset, test dataset, unbatched input dimensions, loss function, eval metrics


# ------------------------------------------------------------------
# ------------------------------ Models ----------------------------
# ------------------------------------------------------------------

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
        x = jnn.Dropout(self.dropout, deterministic=not train)(x)
        for j, l in enumerate(self.layer_dims):
            x = jnn.Dense(features=l, use_bias=self.use_bias)(x)
            if j != len(self.layer_dims) - 1 or not self.drop_last_activation:
                x = self.activation(x)    
                x = jnn.Dropout(self.dropout, deterministic=not train)(x)
        return x
    
def make_vgg16(): return VGG(stages=((64, 64), (128, 128), (256, 256, 256), (512, 512, 512), (512, 512, 512)), layer_dims=[512, 512, 10], drop_last_activation=True, dropout=0.1)  # this is VGG-16

# # ----------------------------------------------------------------------------
# # TAKEN DIRECTLY FROM https://github.com/fattorib/Flax-ResNets/tree/master
    
# import jax
# from typing import Any, Callable, Sequence, Optional
# from jax import lax, random, numpy as jnp
# import flax
# from flax.core import freeze, unfreeze
# from flax import linen as nn
# import copy
# from functools import partial
# import numpy as np

# ModuleDef = Any
# dtypedef = Any


# class ResidualBlock(nn.Module):
#     # Define collection of datafields here
#     in_channels: int

#     # For batchnorm, you can pass it as a ModuleDef
#     norm: ModuleDef

#     # dtype for fp16/32 training
#     dtype: dtypedef = jnp.float32

#     # define init for conv layers
#     kernel_init: Callable = nn.initializers.kaiming_normal()

#     @nn.compact
#     def __call__(self, x):
#         residual = x

#         x = nn.Conv(
#             kernel_size=(3, 3),
#             strides=1,
#             features=self.in_channels,
#             padding="SAME",
#             use_bias=False,
#             kernel_init=self.kernel_init,
#             dtype=self.dtype,
#         )(x)
#         x = self.norm()(x)
#         x = nn.relu(x)
#         x = nn.Conv(
#             kernel_size=(3, 3),
#             strides=1,
#             features=self.in_channels,
#             padding="SAME",
#             use_bias=False,
#             kernel_init=self.kernel_init,
#             dtype=self.dtype,
#         )(x)
#         x = self.norm()(x)

#         x = x + residual

#         return nn.relu(x)


# class DownSampleResidualBlock(nn.Module):
#     # Define collection of datafields here
#     in_channels: int
#     out_channels: int

#     # For batchnorm, you can pass it as a ModuleDef
#     norm: ModuleDef

#     # dtype for fp16/32 training
#     dtype: dtypedef = jnp.float32

#     # define init for conv layers
#     kernel_init: Callable = nn.initializers.kaiming_normal()

#     @nn.compact
#     def __call__(self, x):
#         residual = x

#         x = nn.Conv(
#             kernel_size=(3, 3),
#             strides=1,
#             features=self.in_channels,
#             padding="SAME",
#             use_bias=False,
#             kernel_init=self.kernel_init,
#             dtype=self.dtype,
#         )(x)
#         x = self.norm()(x)
#         x = nn.relu(x)
#         x = nn.Conv(
#             kernel_size=(3, 3),
#             strides=(2, 2),
#             features=self.out_channels,
#             padding=((1, 1), (1, 1)),
#             use_bias=False,
#             kernel_init=self.kernel_init,
#             dtype=self.dtype,
#         )(x)
#         x = self.norm()(x)

#         x = x + self.pad_identity(residual)

#         return nn.relu(x)

#     @nn.nowrap
#     def pad_identity(self, x):
#         # Pad identity connection when downsampling
#         return jnp.pad(
#             x[:, ::2, ::2, ::],
#             ((0, 0), (0, 0), (0, 0), (self.out_channels // 4, self.out_channels // 4)),
#             "constant",
#         )


# class ResNet(nn.Module):
#     # Define collection of datafields here
#     filter_list: Sequence[int]
#     N: int
#     num_classes: int

#     # dtype for fp16/32 training
#     dtype: dtypedef = jnp.float32

#     # define init for conv and linear layers
#     kernel_init: Callable = nn.initializers.kaiming_normal()

#     # For train/test differences, want to pass “mode switches” to __call__
#     @nn.compact
#     def __call__(self, x, train=True):

#         norm = partial(
#             nn.BatchNorm,
#             use_running_average=not train,
#             momentum=0.1,
#             epsilon=1e-5,
#             dtype=self.dtype,
#         )
#         x = nn.Conv(
#             kernel_size=(3, 3),
#             strides=1,
#             features=self.filter_list[0],
#             padding="SAME",
#             use_bias=False,
#             kernel_init=self.kernel_init,
#             dtype=self.dtype,
#         )(x)

#         x = norm()(x)
#         x = nn.relu(x)

#         # First stage
#         for _ in range(0, self.N - 1):
#             x = ResidualBlock(
#                 in_channels=self.filter_list[0], norm=norm, dtype=self.dtype
#             )(x)

#         x = DownSampleResidualBlock(
#             in_channels=self.filter_list[0],
#             out_channels=self.filter_list[1],
#             norm=norm,
#             dtype=self.dtype,
#         )(x)

#         # Second stage
#         for _ in range(0, self.N - 1):
#             x = ResidualBlock(
#                 in_channels=self.filter_list[1], norm=norm, dtype=self.dtype
#             )(x)

#         x = DownSampleResidualBlock(
#             in_channels=self.filter_list[1],
#             out_channels=self.filter_list[2],
#             norm=norm,
#             dtype=self.dtype,
#         )(x)

#         # Third stage
#         for _ in range(0, self.N):
#             x = ResidualBlock(
#                 in_channels=self.filter_list[2], norm=norm, dtype=self.dtype
#             )(x)

#         # Global pooling
#         x = jnp.mean(x, axis=(1, 2))

#         x = x.reshape(x.shape[0], -1)
#         x = nn.Dense(
#             features=self.num_classes, kernel_init=self.kernel_init, dtype=self.dtype
#         )(x)

#         return x


# def _resnet(layers, N, dtype=jnp.float32, num_classes=10):
#     model = ResNet(filter_list=layers, N=N, dtype=dtype, num_classes=num_classes)
#     return model


# def ResNet20(
#     dtype=jnp.float32,
# ):
#     return _resnet(layers=[16, 32, 64], N=3, dtype=dtype, num_classes=10)

# # ------------------------------------------------------------------------
