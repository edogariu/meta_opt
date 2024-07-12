from typing import Union, Callable, Iterable, Optional
from flax import struct

# from torch import optim, Tensor
import optax

from .base import OptimizerConfig

# ==============================================================================
# --------------------------   SGD   -------------------------------------------
# ==============================================================================


@struct.dataclass
class SGDConfig(OptimizerConfig):
    # REQUIRED
    learning_rate: Union[float, Callable[[int], float]]  # learning rate or schedule

    # OPTIONAL
    momentum: Optional[float] = None
    nesterov: bool = False
    weight_decay: Optional[float] = None
    grad_clip: Optional[float] = None

    # METADATA
    optimizer_name: str = 'SGD'
    self_tuning: bool = False
    reset_opt_state: bool = True  # whether to also reset the optimizer state during the episodic resets


#     def make_torch(self) -> Callable[[Iterable[Tensor]], optim.Optimizer]:
#         """
#         Instantiates this optimizer configuration for use with pytorch. 
#         For example, if this were SGD, it would return roughly the same thing as
#                 `lambda params: torch.optim.SGD(params, lr=self.lr, ...)`
#         and could be used afterward in the usual way.
#         """
#         assert self.grad_clip is None, 'havent added gradient clipping to pytorch optimizers yet, my bad'
#         return lambda params: optim.SGD(params,
#                                         lr=self.learning_rate, 
#                                         momentum=self.momentum or 0., 
#                                         weight_decay=self.weight_decay or 0., 
#                                         nesterov=self.nesterov)


    def make_jax(self) -> optax.GradientTransformationExtraArgs:
        """
        Instantiates this optimizer configuration for use with jax/flax/optax. 
        For example, if this were SGD, it would return roughly the same thing as
                `optax.sgd(learning_rate=self.lr, ...)`
        and could be used afterward in the usual way.
        """
        opt = optax.sgd(learning_rate=self.learning_rate, 
                        momentum=self.momentum,
                        nesterov=self.nesterov)
        if self.weight_decay is not None: opt = optax.chain(optax.add_decayed_weights(self.weight_decay), opt)
        if self.grad_clip is not None: opt = optax.chain(opt, optax.clip(self.grad_clip))
        return opt