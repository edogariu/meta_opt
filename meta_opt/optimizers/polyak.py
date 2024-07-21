from typing import Callable, Optional

import jax
import optax
import chex

from .base import OptimizerConfig
from .schedules import ScheduleConfig, ConstantScheduleConfig

# ==============================================================================
# --------------------------   Polyak SGD   ------------------------------------
# ==============================================================================


@OptimizerConfig.register
@chex.dataclass
class PolyakConfig(OptimizerConfig):
    # REQUIRED
    f_min: float

    # OPTIONAL
    max_learning_rate: float = 1.0  # learning rate
    scaling: ScheduleConfig = ConstantScheduleConfig(learning_rate=1.0)
    eps: float = 1e-8
    weight_decay: Optional[float] = None
    grad_clip: Optional[float] = None

    # METADATA
    optimizer_name: str = 'Polyak'
    self_tuning: bool = True
    reset_opt_state: bool = True  # whether to also reset the optimizer state during the episodic resets

    @staticmethod
    def from_dict(d: dict):
        ret = {}
        for k in ['f_min',]:  # required
            ret[k] = d[k]
        for k in ['max_learning_rate', 'scaling', 'eps', 'weight_decay', 'grad_clip']:  # optional
            if k in d: 
                if k == 'scaling':
                    ret[k] = ScheduleConfig.from_dict(d[k])
                else:
                    ret[k] = d[k]
        return PolyakConfig(**ret)
    
    def make_jax(self) -> optax.GradientTransformationExtraArgs:
        """
        Instantiates this optimizer configuration for use with jax/flax/optax. 
        For example, if this were SGD, it would return roughly the same thing as
                `optax.sgd(learning_rate=self.lr, ...)`
        and could be used afterward in the usual way.
        """
        opt = optax.polyak_sgd(max_learning_rate=self.max_learning_rate,
                               f_min=self.f_min,
                               eps=self.eps,
                               scaling=self.scaling.make_jax())
        if self.weight_decay is not None: opt = optax.chain(optax.add_decayed_weights(self.weight_decay), opt)
        if self.grad_clip is not None: opt = optax.chain(optax.clip(self.grad_clip), opt)

        @jax.jit
        def update_fn(grads: chex.ArrayTree, 
                      opt_state: optax.OptState, 
                      params: chex.ArrayTree,
                      cost_fn: Callable[[chex.ArrayTree], float],
                      **extra_args ):
            value = cost_fn(params)
            return opt.update(grads, opt_state, params, value=value)
        
        return optax.GradientTransformationExtraArgs(opt.init, update_fn)
