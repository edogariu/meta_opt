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
    
    num_epochs = int(1 + (num_iters * batch_size) / len(train_ds))
    train_ds = train_ds.repeat(num_epochs).shuffle(1024).batch(batch_size, drop_remainder=True).take(num_iters).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.shuffle(1024).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, test_ds, jnp.zeros((1, 32, 32, 3)), cross_entropy, {'loss': cross_entropy, 'acc': accuracy}  # train dataset, test dataset, unbatched input dimensions, loss function, eval metrics


# ------------------------------------------------------------------
# ------------------------------ Models ----------------------------
# ------------------------------------------------------------------

class ConvBNReLU(jnn.Module):
    num_channels: int
    kernel_size: int
    name: str
    dropout: float = 0.1
    use_bias: bool = True
    
    @jnn.compact
    def __call__(self, x, train: bool):
        x = jnn.Conv(features=self.num_channels, kernel_size=(self.kernel_size, self.kernel_size), use_bias=self.use_bias, name=self.name)(x)
        x = jnn.activation.relu(x)
        x = jnn.BatchNorm(use_running_average=not train)(x)
        if self.dropout > 0: x = jnn.Dropout(self.dropout, deterministic=not train)(x)
        return x



class VGG16(jnn.Module):
    """CIFAR-10 VGG-16 architecture taken from https://github.com/SeHwanJoo/cifar10-vgg16/blob/master/vgg16.py"""
    # i changed the dropouts, they initially were 0.3, 0, 0.4, 0, 0.4, 0.4, 0, 0.4, 0.4, 0, 00.20
    
    @jnn.compact
    def __call__(self, x, train: bool):
        # first stage
        x = ConvBNReLU(num_channels=64, kernel_size=3, name='conv1_1', dropout=0.3)(x, train)
        x = ConvBNReLU(num_channels=64, kernel_size=3, name='conv1_2', dropout=0.)(x, train)
        x = jnn.max_pool(x, (2, 2), (2, 2))
        # second stage
        x = ConvBNReLU(num_channels=128, kernel_size=3, name='conv2_1', dropout=0.4)(x, train)
        x = ConvBNReLU(num_channels=128, kernel_size=3, name='conv2_2', dropout=0.)(x, train)
        x = jnn.max_pool(x, (2, 2), (2, 2))
        # third stage
        x = ConvBNReLU(num_channels=256, kernel_size=3, name='conv3_1', dropout=0.4)(x, train)
        x = ConvBNReLU(num_channels=256, kernel_size=3, name='conv3_2', dropout=0.4)(x, train)
        x = ConvBNReLU(num_channels=256, kernel_size=3, name='conv3_3', dropout=0.)(x, train)
        x = jnn.max_pool(x, (2, 2), (2, 2))
        # fourth stage
        x = ConvBNReLU(num_channels=512, kernel_size=3, name='conv4_1', dropout=0.4)(x, train)
        x = ConvBNReLU(num_channels=512, kernel_size=3, name='conv4_2', dropout=0.4)(x, train)
        x = ConvBNReLU(num_channels=512, kernel_size=3, name='conv4_3', dropout=0.)(x, train)
        x = jnn.max_pool(x, (2, 2), (2, 2))
        # fifth stage
        x = ConvBNReLU(num_channels=512, kernel_size=3, name='conv5_1', dropout=0.4)(x, train)
        x = ConvBNReLU(num_channels=512, kernel_size=3, name='conv5_2', dropout=0.4)(x, train)
        x = ConvBNReLU(num_channels=512, kernel_size=3, name='conv5_3', dropout=0.)(x, train)
        x = jnn.max_pool(x, (2, 2), (2, 2))

        # flatten and produce logits        
        x = x.reshape((x.shape[0], -1))
        x = jnn.Dropout(0.5, deterministic=not train)(x)
        x = jnn.activation.relu(jnn.Dense(features=512)(x))
        x = jnn.BatchNorm(use_running_average=not train)(x)
        x = jnn.Dense(features=10)(x)
        return x

        
# class VGG(jnn.Module):
#     stages: List[List[int]]   # channels of each stage for all the stages
#     layer_dims: List[int] = None  # layer dims for the mlp
#     activation: jnn.Module = jnn.activation.relu
#     drop_last_activation: bool = True
#     use_bias: bool = True
#     dropout: float = 0.1
    
#     @jnn.compact
#     def __call__(self, x, train: bool):
#         for i, stage in enumerate(self.stages):
#             for j, chan in enumerate(stage):
#                 x = jnn.Conv(features=chan, kernel_size=(3, 3), use_bias=self.use_bias, name=f'conv {i},{j}')(x)
#                 x = self.activation(x)
#             x = jnn.max_pool(x, (2, 2), (2, 2))
#             x = jnn.Dropout(self.dropout, deterministic=not train)(x)
#         x = x.reshape((x.shape[0], -1))
#         x = jnn.Dropout(self.dropout, deterministic=not train)(x)
#         for j, l in enumerate(self.layer_dims):
#             x = jnn.Dense(features=l, use_bias=self.use_bias)(x)
#             if j != len(self.layer_dims) - 1 or not self.drop_last_activation:
#                 x = self.activation(x)    
#                 x = jnn.Dropout(self.dropout, deterministic=not train)(x)
#         return x
    
# def make_vgg16(): return VGG(stages=((64, 64), (128, 128), (256, 256, 256), (512, 512, 512), (512, 512, 512)), layer_dims=[512, 512, 10], drop_last_activation=True, dropout=0.1)  # this is VGG-16
