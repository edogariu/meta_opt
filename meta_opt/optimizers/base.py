from typing import Callable, Iterable
import abc
from flax import struct

from torch import optim, Tensor
import optax

"""
Optimizer hyperparameters and meta-information in a *makeable* config format.
Each optimizer config will have a way to make that optimizer for torch or for jax, see
below for details
"""

class OptimizerConfig(abc.ABC):
    optimizer_name: str
    self_tuning: bool
    reset_opt_state: bool

    @abc.abstractmethod
    def make_torch(self) -> Callable[[Iterable[Tensor]], optim.Optimizer]:
        """
        Instantiates this optimizer configuration for use with pytorch. 
        For example, if this were SGD, it would return roughly the same thing as
                `lambda params: torch.optim.SGD(params, lr=self.lr, ...)`
        and could be used afterward in the usual way.
        """

    @abc.abstractmethod
    def make_jax(self) -> optax.GradientTransformationExtraArgs:
        """
        Instantiates this optimizer configuration for use with jax/flax/optax. 
        For example, if this were SGD, it would return roughly the same thing as
                `optax.sgd(learning_rate=self.lr, ...)`
        and could be used afterward in the usual way.
        """

    def make(self, framework: str):
        if framework == 'torch': 
            return self.make_torch()
        elif framework == 'jax':
            return self.make_jax()
        else:
            raise NotImplementedError(framework)
