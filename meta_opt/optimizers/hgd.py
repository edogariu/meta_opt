from typing import Optional

import jax
import jax.flatten_util
import jax.numpy as jnp
import chex
import optax
from optax._src import base

from .base import OptimizerConfig


# ==============================================================================
# --------------------------   Hypergradient Descent   -------------------------
#           - 3 variants: SGD, SGD + momentum, adam. we make two classes for them
# ==============================================================================


@OptimizerConfig.register
@chex.dataclass
class SGDHGDConfig(OptimizerConfig):
    # REQUIRED
    meta_optimizer_cfg: OptimizerConfig
    initial_lr: float

    # OPTIONAL
    momentum: Optional[float] = None
    nesterov: bool = False
    weight_decay: Optional[float] = None
    grad_clip: Optional[float] = None

    # METADATA
    optimizer_name: str = 'SGDHGD'
    self_tuning: bool = True
    reset_opt_state: bool = True  # whether to also reset the optimizer state during the episodic resets

    @staticmethod
    def fromdict(d: dict):
        ret = {}
        for k in ['meta_optimizer_cfg', 'initial_lr']:  # required
            if k == 'meta_optimizer_cfg':
                ret[k] = OptimizerConfig.from_dict(d[k])
            else:
                ret[k] = d[k]
        for k in ['momentum', 'nesterov', 'weight_decay', 'grad_clip']:  # optional
            if k in d: ret[k] = d[k]
        return SGDHGDConfig(**ret)


    def make_jax(self) -> optax.GradientTransformationExtraArgs:
        if self.nesterov: assert self.momentum > 0, 'cannot have nesterov without provided momentum'
        meta_opt = self.meta_optimizer_cfg.make_jax()

        @jax.jit
        def init_fn(params):
            opt_state = {'lr': self.initial_lr,
                         'buffer': jax.tree_util.tree_map(jnp.zeros_like, params) if self.momentum else None,
                         'meta_opt_state': meta_opt.init(jnp.ones((1,))[0] * self.initial_lr), 
                         'prev_grads': jax.tree_util.tree_map(jnp.zeros_like, params)}
            return (opt_state, optax.EmptyState())
        
        @jax.jit
        def update_fn(grads: chex.ArrayTree, 
                    opt_state: chex.ArrayTree, 
                    params: Optional[chex.ArrayTree] = None,
                    **extra_args,
                    ):
            del params
            opt_state, _o = opt_state

            # apply the regular update
            if self.nesterov:  # nesterov momentum
                buffer = jax.tree_util.tree_map(lambda v1, v2: v1 * self.momentum + v2, opt_state['buffer'], grads)
                next_grads = jax.tree_util.tree_map(lambda b, g: g + self.momentum * b, buffer, grads)
                updates = jax.tree_util.tree_map(lambda n: -opt_state['lr'] * n, next_grads)
            elif self.momentum:  # momentum
                buffer = jax.tree_util.tree_map(lambda v1, v2: v1 * self.momentum + v2 * (1 - self.momentum), opt_state['buffer'], grads)
                updates = jax.tree_util.tree_map(lambda b: -opt_state['lr'] * b, buffer)
                next_grads = grads
            else:  # vanilla SGD
                buffer = None
                updates = jax.tree_util.tree_map(lambda g: -opt_state['lr'] * g, grads)
                next_grads = grads

            # apply the meta update
            meta_grad = -sum(jax.tree_leaves(jax.tree_map(lambda v1, v2: jnp.dot(v1.reshape(-1), v2.reshape(-1)), grads, opt_state['prev_grads'])))
            meta_updates, meta_opt_state = meta_opt.update(meta_grad, opt_state['meta_opt_state'], opt_state['lr'])
            lr = optax.apply_updates(opt_state['lr'], meta_updates)

            opt_state = {'lr': lr,
                         'buffer': buffer,
                         'meta_opt_state': meta_opt_state,
                         'prev_grads': next_grads}
            return updates, (opt_state, _o)
        
        opt = base.GradientTransformation(init_fn, update_fn)
        if self.weight_decay is not None: opt = optax.chain(optax.add_decayed_weights(self.weight_decay), opt)
        if self.grad_clip is not None: opt = optax.chain(opt, optax.clip(self.grad_clip))
        return opt
    

@OptimizerConfig.register
@chex.dataclass
class AdamHGDConfig(OptimizerConfig):
    # REQUIRED
    meta_optimizer_cfg: OptimizerConfig
    initial_lr: float
    b1: float
    b2: float

    # OPTIONAL
    eps: float = 1e-8
    weight_decay: Optional[float] = None
    grad_clip: Optional[float] = None

    # METADATA
    optimizer_name: str = 'AdamHGD'
    self_tuning: bool = True
    reset_opt_state: bool = True  # whether to also reset the optimizer state during the episodic resets

    @staticmethod
    def fromdict(d: dict):
        ret = {}
        for k in ['meta_optimizer_cfg', 'initial_lr', 'b1', 'b2']:  # required
            if k == 'meta_optimizer_cfg':
                ret[k] = OptimizerConfig.from_dict(d[k])
            else:
                ret[k] = d[k]
        for k in ['eps', 'weight_decay', 'grad_clip']:  # optional
            if k in d: ret[k] = d[k]
        return SGDHGDConfig(**ret)


    def make_jax(self) -> optax.GradientTransformationExtraArgs:
        meta_opt = self.meta_optimizer_cfg.make_jax()

        @jax.jit
        def init_fn(params):
            opt_state = {'lr': self.initial_lr,
                         't': 1,
                         'buffer1': jax.tree_util.tree_map(jnp.zeros_like, params),
                         'buffer2': jax.tree_util.tree_map(jnp.zeros_like, params),
                         'meta_opt_state': meta_opt.init(jnp.ones((1,))[0] * self.initial_lr), 
                         'prev_grads': jax.tree_util.tree_map(jnp.zeros_like, params)}
            return (opt_state, optax.EmptyState())
        
        @jax.jit
        def update_fn(grads: chex.ArrayTree, 
                    opt_state: chex.ArrayTree, 
                    params: Optional[chex.ArrayTree] = None,
                    **extra_args,
                    ):
            del params
            opt_state, _o = opt_state

            # apply the regular update
            buffer1 = jax.tree_util.tree_map(lambda b, g: self.b1 * b + (1 - self.b1) * g, opt_state['buffer1'], grads)
            buffer2 = jax.tree_util.tree_map(lambda b, g: self.b2 * b + (1 - self.b2) * jnp.square(g), opt_state['buffer2'], grads)
            bias1 = jax.tree_util.tree_map(lambda b: b / (1 - self.b1 ** opt_state['t']), buffer1)
            bias2 = jax.tree_util.tree_map(lambda b: b / (1 - self.b2 ** opt_state['t']), buffer2)
            next_grads = jax.tree_util.tree_map(lambda momentum, velocity:  momentum / (jnp.sqrt(velocity) + self.eps), bias1, bias2)
            updates = jax.tree_util.tree_map(lambda n: -(opt_state['lr'] / jnp.sqrt(opt_state['t'])) * n, next_grads)

            # apply the meta update
            meta_grad = -sum(jax.tree_leaves(jax.tree_map(lambda v1, v2: jnp.dot(v1.reshape(-1), v2.reshape(-1)), grads, opt_state['prev_grads'])))
            meta_updates, meta_opt_state = meta_opt.update(meta_grad, opt_state['meta_opt_state'], opt_state['lr'])
            lr = optax.apply_updates(opt_state['lr'], meta_updates)

            opt_state = {'lr': lr,
                         't': opt_state['t'] + 1,
                         'buffer1': buffer1,
                         'buffer2': buffer2,
                         'meta_opt_state': meta_opt_state,
                         'prev_grads': next_grads}
            return updates, (opt_state, _o)
        
        opt = base.GradientTransformation(init_fn, update_fn)
        if self.weight_decay is not None: opt = optax.chain(optax.add_decayed_weights(self.weight_decay), opt)
        if self.grad_clip is not None: opt = optax.chain(opt, optax.clip(self.grad_clip))
        return opt
