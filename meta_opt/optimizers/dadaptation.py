from typing import Optional

import jax
import chex
import optax

from .base import OptimizerConfig
from .schedules import ScheduleConfig

# ==============================================================================
# --------------------------   D-Adaptation   ----------------------------------
# ==============================================================================


@OptimizerConfig.register
@chex.dataclass
class DAdaptationConfig(OptimizerConfig):
    # REQUIRED
    learning_rate_schedule_cfg: ScheduleConfig  # learning rate or schedule
    b1: float
    b2: float

    # OPTIONAL
    eps: float = 1e-8
    estim_lr0: float = 1e-6
    weight_decay: Optional[float] = None
    grad_clip: Optional[float] = None

    # METADATA
    optimizer_name: str = 'DAdaptation'
    self_tuning: bool = True
    reset_opt_state: bool = True  # whether to also reset the optimizer state during the episodic resets

    @staticmethod
    def from_dict(d: dict):
        ret = {}
        for k in ['learning_rate_schedule_cfg', 'b1', 'b2']:  # required
            if k == 'learning_rate_schedule_cfg':
                ret[k] = ScheduleConfig.from_dict(d[k])
            else: 
                ret[k] = d[k]
        for k in ['eps', 'estim_lr0', 'weight_decay', 'grad_clip']:  # optional
            if k in d: ret[k] = d[k]
        return DAdaptationConfig(**ret)


    def make_jax(self) -> optax.GradientTransformationExtraArgs:
        """
        Instantiates this optimizer configuration for use with jax/flax/optax. 
        For example, if this were SGD, it would return roughly the same thing as
                `optax.sgd(learning_rate=self.lr, ...)`
        and could be used afterward in the usual way.
        """
        weight_decay = 0.0 if self.weight_decay is None else self.weight_decay
        opt = optax.contrib.dadapt_adamw(learning_rate=self.learning_rate_schedule_cfg.make_jax(),
                                         betas=(self.b1, self.b2),
                                         eps=self.eps,
                                         estim_lr0=self.estim_lr0,
                                         weight_decay=weight_decay)
        if self.grad_clip is not None: opt = optax.chain(opt, optax.clip(self.grad_clip))
        update_fn = jax.jit(lambda grads, opt_state, params, **kwargs: opt.update(grads, opt_state, params))
        return optax.GradientTransformation(opt.init, update_fn)
