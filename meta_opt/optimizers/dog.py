from typing import Optional, Callable

import jax
import chex
import optax

from .base import OptimizerConfig
from .schedules import ScheduleConfig, ConstantScheduleConfig

# ==============================================================================
# --------------------------  Distance-over-Weighted-Gradients   ---------------
# ==============================================================================


@OptimizerConfig.register
@chex.dataclass
class DoGConfig(OptimizerConfig):
    # REQUIRED

    # OPTIONAL
    learning_rate_schedule_cfg: ScheduleConfig = ConstantScheduleConfig(learning_rate=1.0) # learning rate or schedule
    reps_rel: float = 1e-6
    init_learning_rate: Optional[float] = None
    eps: float = 1e-8
    weight_decay: Optional[float] = None
    grad_clip: Optional[float] = None

    # METADATA
    optimizer_name: str = 'DoG'
    self_tuning: bool = True
    reset_opt_state: bool = True  # whether to also reset the optimizer state during the episodic resets

    @staticmethod
    def from_dict(d: dict):
        ret = {}
        for k in ['learning_rate_schedule_cfg', 'reps_rel', 'init_learning_rate', 'eps', 'weight_decay', 'grad_clip']:  # optional
            if k in d: 
                if k == 'learning_rate_schedule_cfg':
                    ret[k] = ScheduleConfig.from_dict(d[k])
                else:
                    ret[k] = d[k]
        return DoGConfig(**ret)


    def make_jax(self) -> optax.GradientTransformationExtraArgs:
        """
        Instantiates this optimizer configuration for use with jax/flax/optax. 
        For example, if this were SGD, it would return roughly the same thing as
                `optax.sgd(learning_rate=self.lr, ...)`
        and could be used afterward in the usual way.
        """
        opt = optax.contrib.dog(learning_rate=self.learning_rate_schedule_cfg.make_jax(),
                                reps_rel=self.reps_rel,
                                eps=self.eps,
                                init_learning_rate=self.init_learning_rate,
                                weight_decay=self.weight_decay)
        if self.grad_clip is not None: opt = optax.chain(opt, optax.clip(self.grad_clip))

        @jax.jit
        def update_fn(grads: chex.ArrayTree, 
                      opt_state: optax.OptState, 
                      params: chex.ArrayTree,
                      cost_fn: Callable[[chex.ArrayTree], float],
                      **extra_args ):
            value = cost_fn(params)
            return opt.update(grads, opt_state, params, value=value)
        
        return optax.GradientTransformationExtraArgs(opt.init, update_fn)
