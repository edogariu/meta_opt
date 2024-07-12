from absl import logging
from typing import Tuple, Callable, Iterable, Optional, Dict
import functools

# from torch import optim, Tensor

import jax
import jax.flatten_util
import jax.numpy as jnp
from flax import struct
import optax
from optax._src import base
import chex

from .base import OptimizerConfig
from .sgd import SGDConfig
from .adamw import AdamWConfig
from ..utils import bcolors, get_size, sharding_constraint, get_mesh


@struct.dataclass
class MetaOptConfig(OptimizerConfig):
    # params of the base optimizer
    base_learning_rate: float
    weight_decay: float
    grad_clip: float
    scale_by_adam_betas: Optional[Tuple[float, float]]  # set to `None` to not rescale disturbances with Adam rescaling

    # params of the meta-optimizer
    H: int  # number of past disturbances to use
    HH: int  # rollout length
    m_method: str  # how to compute controls from past disturbances, must be one of ['scalar', 'diagonal', 'full']
    meta_optimizer_cfg: OptimizerConfig  # presumably one of `SGDConfig` or `AdamWConfig`
    fake_the_dynamics: bool  # whether to use the gradient buffer to time-evolve the system rather than taking bona fide train_steps during counterfactual rollout
    freeze_gpc_params: bool  # whether to skip the controller update step. set this to False to learn optimizer, and True to deploy it
    freeze_cost_fn_during_rollouts: bool  # whether to use one fixed batch during counterfactual rollouts
    use_bfloat16: bool

    # METADATA
    optimizer_name: str = 'MetaOpt'
    self_tuning: bool = True
    reset_opt_state: bool = True  # Whether to also reset the optimizer state during the episodic resets. Dont worry, this resets everything except the M parameters (including things like disturbance transformation state, for example)

    @staticmethod
    def fromdict(d: dict):
        ret = {}
        meta_optimizer_name = d['meta_optimizer_cfg']['optimizer_name']
        if meta_optimizer_name == 'SGD':
            ret['meta_optimizer_cfg'] = SGDConfig.fromdict(d['meta_optimizer_cfg'])
        elif meta_optimizer_name == 'AdamW':
            ret['meta_optimizer_cfg'] = AdamWConfig.fromdict(d['meta_optimizer_cfg'])
        else:
            raise ValueError(f'unknown meta optimizer {meta_optimizer_name}')
        ret['meta_optimizer_cfg'] = OptimizerConfig.fromdict(d['meta_optimizer_cfg'])
        
        for k in ['base_learning_rate', 'weight_decay', 'grad_clip', 'scale_by_adam_betas',
                  'H', 'HH', 'm_method', 'use_bfloat16',
                  'fake_the_dynamics', 'freeze_gpc_params', 'freeze_cost_fn_during_rollouts',]:  # required
            ret[k] = d[k]
        for k in []:  # optional
            if k in d: ret[k] = d[k]
        return MetaOptConfig(**ret)


    # def make_torch(self) -> Callable[[Iterable[Tensor]], optim.Optimizer]:
    #     """
    #     Instantiates this optimizer configuration for use with pytorch. 
    #     For example, if this were SGD, it would return roughly the same thing as
    #             `lambda params: torch.optim.SGD(params, lr=self.lr, ...)`
    #     and could be used afterward in the usual way.
    #     """
    #     raise NotImplementedError('havent implemented metaopt in pytorch yet, sorry')
             

    def make_jax(self) -> optax.GradientTransformationExtraArgs:
        """
        Instantiates this optimizer configuration for use with jax/flax/optax. 
        For example, if this were SGD, it would return roughly the same thing as
                `optax.sgd(learning_rate=self.lr, ...)`
        and could be used afterward in the usual way.
        """
        meta_optimizer = self.meta_optimizer_cfg.make_jax()
        opt = make_jax_metaopt(base_lr=self.base_learning_rate, weight_decay=self.weight_decay, grad_clip=self.grad_clip, scale_by_adam_betas=self.scale_by_adam_betas,
                               H=self.H, HH=self.HH, m_method=self.m_method,
                               gpc_tx=meta_optimizer,
                               fake_the_dynamics=self.fake_the_dynamics, freeze_gpc_params=self.freeze_gpc_params, freeze_cost_fn_during_rollouts=self.freeze_cost_fn_during_rollouts,
                               dtype=jax.dtypes.bfloat16 if self.use_bfloat16 else float)
        return opt



# ------------------------------------------------------------------------------------------------
# SOURCE CODE FOR JAX METAOPT IMPLEMENTATION
# ------------------------------------------------------------------------------------------------

@jax.jit
def append(arr, val):
    """
    rightmost recent appending, i.e. arr = (val_{t-h}, ..., val_{t-1}, val_t).
    Unlike `jnp.roll`, using a concatenate doesn't make a temp array of size 2 * arr.shape[0]
    """
    # TODO(dogariu) figure out why this allocs a f32 array even when `arr.dtype = jnp.bfloat16`
    arr = jnp.concatenate((arr[1:], val[None]), axis=0, dtype=arr.dtype)
    return arr


@jax.jit
def compute_gpc_control(gpc_params: chex.Array, 
                        disturbance_history: chex.Array) -> chex.Array:
    """Computes the GPC controls.

    Args:
        gpc_params: array of shape `[H,]`, `[H, n]`, or `[H, n, n]`, depending on m_method
        disturbances: array of shape `[H, n]` containing past H disturbances

    Returns:
        chex.Array: shape `[n,]` array of controls
    """
    _, n = disturbance_history.shape

    # the einsum way
    # TODO(dogariu) figure out why this allocs a f32 array even when `arr.dtype = jnp.bfloat16`
    logging.info('[GPC] computing controls the einsum way')
    EINSUM_STRS = {1: 'h,hn->n', 2: 'hn,hn->n', 3: 'hmn,hn->m'}  # how to compute controls for scalar, diagonal, and full gpc_params, respectively
    einsum_str = EINSUM_STRS[gpc_params.ndim]
    ret = jnp.einsum(einsum_str, gpc_params, disturbance_history, preferred_element_type=disturbance_history.dtype)

    # logging.info('[GPC] computing controls the numpy way')
    # if gpc_params.ndim == 1:
    #     ret = disturbance_history.T @ gpc_params
    # elif gpc_params.ndim == 2:
    #     ret = (gpc_params * disturbance_history).sum(axis=0)
    # elif gpc_params.ndim == 3:
    #     ret = jax.lax.batch_matmul(gpc_params, disturbance_history[..., None])[..., 0].sum(axis=0)
    # else:
    #     raise NotImplementedError(f'gpc_params.ndim={gpc_params.ndim} not in [1, 2, 3]')
    
    assert ret.shape == (n,), (ret.shape, (n,))
    return ret


@functools.partial(jax.jit, static_argnums=(1, 10, 12, 13, 14))
def update_gpc_controller_counterfactual(gpc_params: chex.Array,
                                         gpc_tx: optax.GradientTransformation,   # static
                                         gpc_opt_state: optax.OptState,
                                         disturbance_history: chex.Array,
 
                                         base_lr: float,
                                         weight_decay: float,
                                         initial_params: chex.Array,  # params from HH steps ago
                                         cost_fn_history,  # past HH cost functions, starting at the one that would have been used to evolve `initial_params`
                                         curr_cost_fn,
                                         unflatten_fn,
                                         disturbance_transform: optax.GradientTransformation,   # static
                                         initial_disturbance_transform_state: optax.OptState,

                                         # rest of the static args
                                         H: int,
                                         HH: int,
                                         fake_the_dynamics: bool) -> Tuple[chex.Array, optax.OptState, float, chex.Array]:
    @jax.jit
    def gpc_cost_fn(controller_params: chex.Array):
        params = initial_params
        disturbance_transform_state = initial_disturbance_transform_state
        for h in range(HH):  # do it w a loop cause its not long enough for a scan to be faster/better
            if fake_the_dynamics:
                disturbances = jax.lax.dynamic_index_in_dim(disturbance_history, h + H - 1, keepdims=False)
            else:
                grads = jax.grad(lambda p: cost_fn_history[h](unflatten_fn(p)))(params)
                disturbances, disturbance_transform_state = disturbance_transform.update(grads, disturbance_transform_state, params)
            params = (1 - weight_decay) * params - base_lr * disturbances  # play the base SGD step
            params += compute_gpc_control(controller_params, jax.lax.dynamic_slice_in_dim(disturbance_history, h, H))  # play GPC control

        cost = curr_cost_fn(unflatten_fn(params))
        return cost
    
    gpc_cost, gpc_grads = jax.value_and_grad(gpc_cost_fn)(gpc_params)
    gpc_updates, new_gpc_opt_state = gpc_tx.update(gpc_grads, gpc_opt_state, gpc_params)
    gpc_params = optax.apply_updates(gpc_params, gpc_updates)
    return gpc_params, new_gpc_opt_state, gpc_cost, gpc_grads


class JaxMetaOptState(struct.PyTreeNode):

    # learnable controller parameters and corresponding controller optimizer
    gpc_params: chex.Array = struct.field(pytree_node=True)
    gpc_tx: optax.GradientTransformation = struct.field(pytree_node=False)
    gpc_opt_state: optax.OptState = struct.field(pytree_node=True)

    # dynamic optimizer state
    disturbance_history: chex.Array = struct.field(pytree_node=True)
    param_history: chex.Array = struct.field(pytree_node=True)
    cost_fn_history: Tuple[Callable] = struct.field(pytree_node=True)

    # static optimizer state
    H: int = struct.field(pytree_node=False)  # history of the controller, how many past disturbances to use for control
    HH: int = struct.field(pytree_node=False)  # history of the system, how many hallucination steps to take
    t: int = struct.field(pytree_node=True)  # current step
    num_params: int = struct.field(pytree_node=False)  # number of parameters in the model
    flat_size: int = struct.field(pytree_node=False)  # number of elements in flattened iterates (i.e. num_params, but padded for sharding purposes)
    base_lr: float = struct.field(pytree_node=False)

    # for rescaling the gradients/disturbances
    disturbance_transform: optax.GradientTransformation = struct.field(pytree_node=False)
    disturbance_transform_state: optax.OptState = struct.field(pytree_node=True)

    # statistics to log during training
    recent_gpc_grads: chex.Array = struct.field(pytree_node=True)
    recent_gpc_cost: float = struct.field(pytree_node=True)


    def get_logging_metrics(self) -> Dict[str, float]:
        ret = {}
        Ms = self.gpc_params.reshape(self.H, -1).mean(axis=-1)[::-1]
        grad_Ms = self.recent_gpc_grads.reshape(self.H, -1).mean(axis=-1)[::-1]
        assert Ms.shape == (self.H,), (Ms.shape, self.H)
        assert grad_Ms.shape == (self.H,), (grad_Ms.shape, self.H)
        Ms = Ms.at[0].add(-self.base_lr)  # add the effective learning rate to most recent grad coeff
        ret.update({f'M_{i}': m for i, m in enumerate(Ms.reshape(-1))})
        if self.recent_gpc_cost != float('inf'):
            ret.update({f'grad_M_{i}': grad_m for i, grad_m in enumerate(grad_Ms.reshape(-1))})
            ret['gpc_cost'] = self.recent_gpc_cost.reshape(-1).mean()
        sizes = {
            'disturbance_history_memory': get_size(self.disturbance_history),
            'param_history_memory': get_size(self.param_history),
            'cost_fn_history_memory': get_size(self.cost_fn_history),
            'disturbance_transform_state_memory': get_size(self.disturbance_transform_state),
            'total_metaopt_memory': get_size(self),
        }
        ret.update(sizes)
        return ret


def make_jax_metaopt(
        base_lr: float,
        weight_decay: float,
        grad_clip: Optional[float],
        scale_by_adam_betas: Optional[Tuple[float, float]],

        H: int,
        HH: int,
        m_method: str,
        gpc_tx: optax.GradientTransformation,
        dtype,

        fake_the_dynamics: bool,
        freeze_gpc_params: bool,
        freeze_cost_fn_during_rollouts: bool,

        ) -> optax.GradientTransformationExtraArgs:
    """Returns jax optimizer implementing the meta-opt controller.

    Args:
        base_lr: float
            base learning rate before any GPC controls are played
        weight_decay: float
            weight decay applied to the parameters before any GPC controls are played
        grad_clip: float | None
            clipping applied to gradients before they become "disturbances". if None, no clipping
        scale_by_adam_betas: Tuple[float, float]
            tuple of beta1 and beta2 for optax.adam rescaling for gradients before they become "disturbances"
        H: int
            number of past disturbances to use for control
        HH: int
            rollout length
        m_method: str
            method for computing GPC controls. one of 'scalar', 'diagonal', 'full'
        gpc_tx: optax.GradientTransformation
            optax optimizer for the GPC controller parameters
        fake_the_dynamics: bool
            whether to use cached disturbances or compute fresh gradients during rollouts
        freeze_gpc_params: bool
            whether to freeze the GPC controller parameters (i.e. enable deployment mode)
        freeze_cost_fn_during_rollouts: bool
            whether to use the same cost function (i.e. same batch) during the entire rollout
    """

    @jax.jit
    def init_fn(params):

        if freeze_gpc_params:
            logging.warning(f'{bcolors.WARNING}{bcolors.BOLD}the meta-opt controller is frozen! optimizer behavior wont change over time{bcolors.ENDC}')
        else:
            if fake_the_dynamics: 
                logging.warning(f'{bcolors.WARNING}{bcolors.BOLD}we will be faking the dynamics during rollouts. will be faster, but behavior may be different...{bcolors.ENDC}')

        # also make sure that flat size is a multiple of the number of devices along which to shard opt state
        num_params = sum([p.size for p in jax.tree_util.tree_leaves(params)])
        mesh = get_mesh()
        n_opt_devices = mesh.shape['opt'] if mesh is not None else 1
        if n_opt_devices > 1: logging.info(f'{bcolors.OKCYAN}sharding opt state across {n_opt_devices} devices{bcolors.ENDC}')
        if num_params % n_opt_devices != 0:
            flat_size = (1 + num_params // n_opt_devices) * n_opt_devices
            logging.info(f'{bcolors.OKCYAN}needed to pad iterates from {num_params} to {flat_size} to evenly divide over {n_opt_devices} devices{bcolors.ENDC}')
        else: flat_size = num_params

        # make disturbance transform
        if scale_by_adam_betas is not None:
            b1, b2 = scale_by_adam_betas
            disturbance_transform = optax.scale_by_adam(b1=b1, b2=b2, mu_dtype=dtype)
        else:
            disturbance_transform = optax.identity()
        if grad_clip is not None: disturbance_transform = optax.chain(optax.clip(grad_clip), disturbance_transform)

        # make controller
        if m_method == 'scalar': gpc_params = jnp.zeros((H,), dtype=dtype)
        elif m_method == 'diagonal': gpc_params = jnp.zeros((H, flat_size), dtype=dtype)
        elif m_method == 'full': gpc_params = jnp.zeros((H, flat_size, flat_size), dtype=dtype)
        else: raise NotImplementedError(m_method)
        gpc_opt_state = gpc_tx.init(gpc_params)

        # make the big histories and shard them!
        disturbance_history = sharding_constraint(jnp.zeros((H + HH, flat_size), dtype=dtype), (None, 'opt'))
        param_history = sharding_constraint(jnp.zeros((HH, flat_size), dtype=dtype), (None, 'opt'))

        opt_state = JaxMetaOptState(gpc_params=gpc_params,
                                    gpc_tx=gpc_tx,
                                    gpc_opt_state=gpc_opt_state,
                                    disturbance_history=disturbance_history,
                                    param_history=param_history,
                                    cost_fn_history=(jax.tree_util.Partial(lambda _: 0.),) * HH,
                                    H=H,
                                    HH=HH,
                                    t=0,
                                    base_lr=base_lr,
                                    num_params=num_params,
                                    flat_size=flat_size,
                                    disturbance_transform=disturbance_transform,
                                    disturbance_transform_state=disturbance_transform.init(disturbance_history[0]),
                                    recent_gpc_grads=jnp.zeros_like(gpc_params, dtype=gpc_params.dtype), 
                                    recent_gpc_cost=float('inf'))

        return (opt_state, optax.EmptyState())
    
    @jax.jit
    def update_fn(grads: chex.ArrayTree, 
                  opt_state: JaxMetaOptState, 
                  params: chex.ArrayTree,
                  cost_fn: Callable[[chex.ArrayTree], float],
                  **extra_args,
                  ):
        """Applies a single step of the meta-opt optimizer.

        Args:
            grads (chex.ArrayTree): gradients computed for current step
            opt_state (JaxMetaOptState): current state of the meta-optimizer
            params (chex.ArrayTree): current iterates
            cost_fn (Callable[[chex.ArrayTree], float]): autodifferentiable function sending `params -> cost`, whose gradient evaluated at `params` should be `grads`

        Returns:
            chex.ArrayTree: updates
            JaxMetaOptState: new state of the meta-optimizer
        """
        opt_state, _ = opt_state
        assert params is not None, 'failed to provide parameters to the meta-optimizer'
        assert cost_fn is not None, 'failed to provide cost function to the meta-optimizer'

        # flatten things! also make sure that flat size is a multiple of the number of devices along which to shard opt state
        def flatten(v):
            f, u = jax.flatten_util.ravel_pytree(v)
            assert f.shape == (opt_state.num_params,), (f.shape, (opt_state.num_params,))
            f = jnp.pad(f, (0, opt_state.flat_size - opt_state.num_params))
            assert f.shape == (opt_state.flat_size,), (f.shape, (opt_state.flat_size,))
            f = f.astype(dtype)
            unflatten_fn = jax.tree_util.Partial(lambda w: u(w[:opt_state.num_params]))
            return f, unflatten_fn
        flat_params, unflatten_fn = flatten(params)
        del params
        flat_grads, _ = flatten(grads)
        del grads

        # update GPC controller
        if not freeze_gpc_params:
            # if t >= H + HH, compute update to gpc controller
            gpc_params, gpc_opt_state, gpc_cost, gpc_grads = jax.lax.cond(opt_state.t >= H + HH, 
                                                                          
                                                                          # if true
                                                                          lambda: update_gpc_controller_counterfactual(
                                                                                gpc_params=opt_state.gpc_params, 
                                                                                gpc_tx=opt_state.gpc_tx, 
                                                                                gpc_opt_state=opt_state.gpc_opt_state,
                                                                                disturbance_history=opt_state.disturbance_history, 
                                                                                base_lr=base_lr, 
                                                                                weight_decay=weight_decay,
                                                                                initial_params=opt_state.param_history[0], 
                                                                                cost_fn_history=(cost_fn,) * HH if freeze_cost_fn_during_rollouts else opt_state.cost_fn_history, 
                                                                                curr_cost_fn=cost_fn,
                                                                                unflatten_fn=unflatten_fn,
                                                                                disturbance_transform=opt_state.disturbance_transform, 
                                                                                initial_disturbance_transform_state=opt_state.disturbance_transform_state,
                                                                                H=H, 
                                                                                HH=HH, 
                                                                                fake_the_dynamics=fake_the_dynamics,
                                                                            ), 

                                                                            # if false 
                                                                            lambda: (opt_state.gpc_params, opt_state.gpc_opt_state, opt_state.recent_gpc_cost, opt_state.recent_gpc_grads))

            # append to histories
            param_history = append(opt_state.param_history, flat_params)
            # param_history = sharding_constraint(param_history, (None, 'opt'))
            cost_fn_history = opt_state.cost_fn_history[1:] + (cost_fn,)
            disturbances, disturbance_transform_state = opt_state.disturbance_transform.update(flat_grads, opt_state.disturbance_transform_state, params=flat_params)
            disturbance_history = append(opt_state.disturbance_history, disturbances)
            # disturbance_history = sharding_constraint(disturbance_history, (None, 'opt'))
            opt_state = opt_state.replace(gpc_params=gpc_params, gpc_opt_state=gpc_opt_state, disturbance_history=disturbance_history, disturbance_transform_state=disturbance_transform_state, 
                                        param_history=param_history, cost_fn_history=cost_fn_history, 
                                        recent_gpc_grads=gpc_grads, recent_gpc_cost=gpc_cost, t=opt_state.t+1)
        else:
            disturbances, disturbance_transform_state = opt_state.disturbance_transform.update(flat_grads, opt_state.disturbance_transform_state, params=flat_params)
            disturbance_history = append(opt_state.disturbance_history, disturbances)
            # disturbance_history = sharding_constraint(disturbance_history, (None, 'opt'))
            opt_state = opt_state.replace(disturbance_history=disturbance_history, disturbance_transform_state=disturbance_transform_state, t=opt_state.t+1)
        
        # compute GPC control with updated params
        control = (-weight_decay) * flat_params - base_lr * disturbances  # apply base SGD/adam/whatever update
        control += compute_gpc_control(opt_state.gpc_params, disturbance_history[-H:])  # use past H disturbances, including most recent one
        # logging.info('[DTYPES]', opt_state.disturbance_history.dtype, opt_state.param_history.dtype, control.dtype)
        control = unflatten_fn(control)
        control = sharding_constraint(control, (None,))

        return control, (opt_state, optax.EmptyState())
    
    return base.GradientTransformationExtraArgs(init_fn, update_fn)
