from typing import Optional

import chex
import optax

from .base import OptimizerConfig
from .schedules import ScheduleConfig, ConstantScheduleConfig

# ==============================================================================
# --------------------------  Continuous Coin Betting   ------------------------
# ==============================================================================


@OptimizerConfig.register
@chex.dataclass
class COCOBConfig(OptimizerConfig):
    # REQUIRED

    # OPTIONAL
    learning_rate_schedule_cfg: ScheduleConfig = ConstantScheduleConfig(learning_rate=1.0) # learning rate or schedule
    alpha: float = 100
    eps: float = 1e-8
    weight_decay: Optional[float] = None
    grad_clip: Optional[float] = None

    # METADATA
    optimizer_name: str = 'CocoB'
    self_tuning: bool = True
    reset_opt_state: bool = True  # whether to also reset the optimizer state during the episodic resets

    @staticmethod
    def from_dict(d: dict):
        ret = {}
        for k in ['learning_rate_schedule_cfg', 'alpha', 'eps', 'weight_decay', 'grad_clip']:  # optional
            if k in d: 
                if k == 'learning_rate_schedule_cfg':
                    ret[k] = ScheduleConfig.from_dict(d[k])
                else:
                    ret[k] = d[k]
        return COCOBConfig(**ret)


    def make_jax(self) -> optax.GradientTransformationExtraArgs:
        """
        Instantiates this optimizer configuration for use with jax/flax/optax. 
        For example, if this were SGD, it would return roughly the same thing as
                `optax.sgd(learning_rate=self.lr, ...)`
        and could be used afterward in the usual way.
        """
        weight_decay = 0.0 if self.weight_decay is None else self.weight_decay
        opt = optax.contrib.cocob(learning_rate=self.learning_rate_schedule_cfg.make_jax(),
                                  alpha=self.alpha,
                                  eps=self.eps,
                                  weight_decay=weight_decay)
        if self.grad_clip is not None: opt = optax.chain(opt, optax.clip(self.grad_clip))
        return opt
    