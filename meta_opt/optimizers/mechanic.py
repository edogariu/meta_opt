import jax
import chex
import optax

from .base import OptimizerConfig

# ==============================================================================
# --------------------------  Mechanic LR Tuner    ----------------------------
# ==============================================================================


@OptimizerConfig.register
@chex.dataclass
class MechanicConfig(OptimizerConfig):
    # REQUIRED
    base_optimizer_cfg: OptimizerConfig

    # OPTIONAL

    # METADATA
    optimizer_name: str = 'Mechanic'
    self_tuning: bool = True
    reset_opt_state: bool = True  # whether to also reset the optimizer state during the episodic resets

    @staticmethod
    def from_dict(d: dict):
        return MechanicConfig(base_optimizer_cfg=OptimizerConfig.from_dict(d['base_optimizer_cfg']))


    def make_jax(self) -> optax.GradientTransformationExtraArgs:
        """
        Instantiates this optimizer configuration for use with jax/flax/optax. 
        For example, if this were SGD, it would return roughly the same thing as
                `optax.sgd(learning_rate=self.lr, ...)`
        and could be used afterward in the usual way.
        """
        opt = self.base_optimizer_cfg.make_jax()
        opt = optax.contrib.mechanize(opt)
        update_fn = jax.jit(lambda grads, opt_state, params, **kwargs: opt.update(grads, opt_state, params))
        return optax.GradientTransformation(opt.init, update_fn)
    