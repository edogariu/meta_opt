from typing import Optional

import chex
import optax

from .base import OptimizerConfig
from .schedules import ScheduleConfig

# ==============================================================================
# --------------------------   SGD   -------------------------------------------
# ==============================================================================


@OptimizerConfig.register
@chex.dataclass
class SGDConfig(OptimizerConfig):
    # REQUIRED
    learning_rate_schedule_cfg: ScheduleConfig  # learning rate or schedule

    # OPTIONAL
    momentum: Optional[float] = None
    nesterov: bool = False
    weight_decay: Optional[float] = None
    grad_clip: Optional[float] = None

    # METADATA
    optimizer_name: str = 'SGD'
    self_tuning: bool = False
    reset_opt_state: bool = True  # whether to also reset the optimizer state during the episodic resets

    @staticmethod
    def fromdict(d: dict):
        ret = {}
        for k in ['learning_rate_schedule_cfg', ]:  # required
            if k == 'learning_rate_schedule_cfg':
                ret[k] = ScheduleConfig.from_dict(d[k])
            else:
                ret[k] = d[k]
        for k in ['momentum', 'nesterov', 'weight_decay', 'grad_clip']:  # optional
            if k in d: ret[k] = d[k]
        return SGDConfig(**ret)


    def make_jax(self) -> optax.GradientTransformationExtraArgs:
        """
        Instantiates this optimizer configuration for use with jax/flax/optax. 
        For example, if this were SGD, it would return roughly the same thing as
                `optax.sgd(learning_rate=self.lr, ...)`
        and could be used afterward in the usual way.
        """
        opt = optax.sgd(learning_rate=self.learning_rate_schedule_cfg.make_jax(), 
                        momentum=self.momentum,
                        nesterov=self.nesterov)
        if self.weight_decay is not None: opt = optax.chain(optax.add_decayed_weights(self.weight_decay), opt)
        if self.grad_clip is not None: opt = optax.chain(opt, optax.clip(self.grad_clip))
        return opt