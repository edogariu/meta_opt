from typing import Any, Tuple, Dict
import functools
from absl import logging

import jax
import jax.numpy as jnp
import optax
from flax import struct, core, jax_utils

from algorithmic_efficiency import spec

from configs.optimizers import OptimizerConfig, SGDConfig, AdamWConfig, MetaOptConfig
from meta_opt.nn import TrainState
from meta_opt.jax_stuff.jax_meta_opt import jax_meta_opt, JaxMetaOptState
from meta_opt.jax_stuff.jax_utils import sq_norm_pytree
from meta_opt.utils import bcolors, get_size


# -------------------------------------------------------------------------------------------------
# ------------------------------------------ Optimizers -------------------------------------------
# -------------------------------------------------------------------------------------------------

def shard_or_replicate_opt_state(opt_state: optax.OptState):
    opt_state, _ = opt_state
    if isinstance(opt_state, JaxMetaOptState):
        ret = jax_utils.replicate(opt_state)

        # from jax.experimental import mesh_utils
        # from jax.sharding import PositionalSharding
        # def shard_pytree(p):
        #     num_devices = jax.local_device_count()
        #     def shard_array(v):
        #         mesh = PositionalSharding(mesh_utils.create_device_mesh((num_devices,)))
        #         if v.ndim > 1:
        #             shape = [num_devices,] + [1 for _ in range(v.ndim - 1)]
        #             mesh = mesh.reshape(shape)
        #         return jax.device_put(v, mesh)
        #     return jax.tree_map(shard_array, p)
        
        # disturbance_history = shard_pytree(opt_state.disturbance_history)
        # tstate_history = shard_pytree(opt_state.tstate_history)
        # batch_history = shard_pytree(opt_state.batch_history)
        # disturbance_transform_state = opt_state.disturbance_transform_state
        # cstate = opt_state.cstate
        # # disturbance_transform_state = jax_utils.replicate(opt_state.disturbance_transform_state)
        # # cstate = jax_utils.replicate(opt_state.cstate)
        # ret = opt_state.replace(disturbance_history=disturbance_history, tstate_history=tstate_history, batch_history=batch_history, disturbance_transform_state=disturbance_transform_state, cstate=cstate)
        # ret = jax_utils.replicate(ret)
        
        
        # print('starting da print')
        # jax.tree_util.tree_map(lambda v: print(v.__class__, v.shape, v.sharding), ret)
        # print('ending da print')
    else:
        ret = jax_utils.replicate(opt_state)
    return (ret, optax.EmptyState())


def jax_make_optimizer(workload: spec.Workload, cfg: OptimizerConfig) -> optax.GradientTransformation:
    name = cfg.optimizer_name

    if name == 'SGD' or isinstance(cfg, SGDConfig):
        opt = optax.sgd(learning_rate=cfg.learning_rate, 
                        momentum=cfg.momentum,
                        nesterov=cfg.nesterov)
        if cfg.weight_decay is not None: opt = optax.chain(optax.add_decayed_weights(cfg.weight_decay), opt)
        if cfg.grad_clip is not None: opt = optax.chain(opt, optax.clip(cfg.grad_clip))

    elif name == 'AdamW' or isinstance(cfg, AdamWConfig):
        if cfg.weight_decay is not None:
            opt = optax.adamw(learning_rate=cfg.learning_rate,
                              b1=cfg.b1,
                              b2=cfg.b2,
                              eps=cfg.eps,
                              weight_decay=cfg.weight_decay)
        else:
            opt = optax.adam(learning_rate=cfg.learning_rate,
                             b1=cfg.b1,
                             b2=cfg.b2,
                             eps=cfg.eps)
        if cfg.grad_clip is not None: opt = optax.chain(optax.clip(cfg.grad_clip), opt)

    elif name == 'MetaOpt' or isinstance(cfg, MetaOptConfig):
        meta_optimizer = jax_make_optimizer(workload, cfg.meta_optimizer_cfg)

        if cfg.jax_pmap_in_rollouts:
            logging.warn('We are pmapping in both the outer `train_step` function and the inner counterfactual one!')
            train_step, forward = jax_pmapped_train_step, jax_pmapped_forward
        else:
            train_step, forward = jax_train_step, jax_forward

        if not cfg.fake_the_dynamics:  # use the base optimizer to implement `inital_learning_rate` steps
            base_optimizer_cfg = SGDConfig(learning_rate=cfg.initial_learning_rate, momentum=0, nesterov=False, weight_decay=None, grad_clip=None)
            empty_tstate = jax_create_train_state(jax.random.PRNGKey(0), workload, base_optimizer_cfg, distribute_opt_state=cfg.jax_pmap_in_rollouts)
            opt = jax_meta_opt(workload, cfg, meta_optimizer, empty_tstate, train_step, forward)
        else:  # make `train_step` a noop and make sure that the controller itself implements `inital_learning_rate` steps -- someone's gotta :)
            base_optimizer_cfg = SGDConfig(learning_rate=0, momentum=0, nesterov=False, weight_decay=None, grad_clip=None)
            empty_tstate = jax_create_train_state(jax.random.PRNGKey(0), workload, base_optimizer_cfg, distribute_opt_state=cfg.jax_pmap_in_rollouts)
            opt = jax_meta_opt(workload, cfg, meta_optimizer, empty_tstate, (lambda _1, _2, _tstate, _3: (_tstate, None)), forward)

    else:
        raise NotImplementedError(name)
    
    return opt

# -------------------------------------------------------------------------------------------------
# ------------------------------------------ Train States -----------------------------------------
# -------------------------------------------------------------------------------------------------

class JaxTrainState(struct.PyTreeNode, TrainState):
    params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    model_state: Any = struct.field(pytree_node=True)  # may contain auxiliary info (such as batch stats) for some workloads
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState = struct.field(pytree_node=True)

    def get_algoperf_stuff(self) -> Tuple[spec.OptimizerState, spec.ParameterContainer, spec.ModelAuxiliaryState]:
        """Returns a tuple of the things needed by `algorithmic_efficiency` for checkpoints and such."""
        return (self.opt_state, self.tx.update), self.params, self.model_state

    def reset(self, 
              rng: jax.random.PRNGKey, 
              workload: spec.Workload,
              reset_opt_state: bool) -> TrainState:
        """Resets model parameters, auxiliary state, and potentially the optimizer state."""
        model_init_rng, rng = jax.random.split(rng)
        logging.info(f'{bcolors.OKBLUE}{bcolors.BOLD}Resetting model!{bcolors.ENDC}')
        params, model_state = workload.init_model_fn(model_init_rng)
        if reset_opt_state:
            logging.info(f'{bcolors.OKBLUE}{bcolors.BOLD}Also resetting optimizer state!{bcolors.ENDC}')
            params_zeros_like = jax.tree_map(lambda s: jnp.zeros(s.shape_tuple), workload.param_shapes)
            opt_state = self.tx.init(params_zeros_like)
            opt_state = shard_or_replicate_opt_state(opt_state)
            tstate = self.replace(params=params, model_state=model_state, opt_state=opt_state)
        else:
            tstate = self.replace(params=params, model_state=model_state)
        return tstate
    
    def get_num_params(self) -> int:
        return sum(x.size for x in jax.tree_util.tree_leaves(self.params))
    
    def get_memory_usage(self) -> Dict[str, int]:
        params_memory = get_size(jax_utils.unreplicate(self.params))
        model_state_memory = get_size(jax_utils.unreplicate(self.model_state))
        opt_state_memory = get_size(jax_utils.unreplicate(self.opt_state))
        return {'param_memory': params_memory, 
                'model_state_memory': model_state_memory,
                'opt_state_memory': opt_state_memory}
    
    def get_logging_metrics(self) -> Dict[str, Any]:
        ret = {}
        if isinstance(self.opt_state, tuple):
            for o in self.opt_state:
                if hasattr(o, 'get_logging_metrics'):
                    ret.update(o.get_logging_metrics())
        ret.update(self.get_memory_usage())
        ret['param_sq_norm'] = sq_norm_pytree(self.params)
        return ret


def jax_create_train_state(rng: jax.random.PRNGKey,
                           workload: spec.Workload,
                           optimizer_cfg: OptimizerConfig,
                           distribute_opt_state: bool = True) -> JaxTrainState:
    """Creates a train state from scratch. This should initialize model parameters, auxiliary state, and optimizer state."""
    params, model_state = workload.init_model_fn(rng)
    opt = jax_make_optimizer(workload, optimizer_cfg)
    params_zeros_like = jax.tree_map(lambda s: jnp.zeros(s.shape_tuple), workload.param_shapes)
    opt_state = opt.init(params_zeros_like)
    if distribute_opt_state: opt_state = shard_or_replicate_opt_state(opt_state)
    return JaxTrainState(params=params, model_state=model_state, tx=opt, opt_state=opt_state)


def jax_load_train_state(checkpoint,
                         workload: spec.Workload,
                         optimizer_cfg: OptimizerConfig) -> JaxTrainState:
    """Creates a train state from a checkpoint given by `algorithmic_efficiency`."""
    ((opt_state, _), model_params, model_state, _, _, global_step, _) = checkpoint
    logging.info(f'{bcolors.OKGREEN}{bcolors.BOLD}loading train state from checkpoint at step {global_step}{bcolors.ENDC}')
    # opt_state = shard_or_replicate_opt_state(opt_state)
    opt = jax_make_optimizer(workload, optimizer_cfg)
    return JaxTrainState(params=model_params, model_state=model_state, tx=opt, opt_state=opt_state, t=global_step)


# -------------------------------------------------------------------------------------------------
# ------------------------------------- Train Step Functions --------------------------------------
# -------------------------------------------------------------------------------------------------

def _forward(workload: spec.Workload, params, model_state, batch, rng, forward_pass_mode):
    logits, new_model_state = workload.model_fn(
                params,
                batch,
                model_state,
                forward_pass_mode,
                rng,
                update_batch_norm=True)
    loss_dict = workload.loss_fn(
        label_batch=batch['targets'],
        logits_batch=logits,
        mask_batch=batch.get('weights'),
        label_smoothing=0.0)
    summed_loss = loss_dict['summed']
    n_valid_examples = loss_dict['n_valid_examples']
    return summed_loss / n_valid_examples, (new_model_state,)

@functools.partial(jax.jit, static_argnums=(1,))
def jax_forward(rng: jax.random.PRNGKey, 
                workload: spec.Workload,
                tstate: JaxTrainState,
                batch) -> float:
    loss, _ = _forward(workload, tstate.params, tstate.model_state, batch, rng, spec.ForwardPassMode.EVAL)
    return loss
    
@functools.partial(jax.pmap, axis_name='batch', in_axes=(0, None, 0, 0), static_broadcasted_argnums=(1,))
def _jax_pmapped_forward(rng: jax.random.PRNGKey, 
                        workload: spec.Workload,
                        tstate: JaxTrainState,
                        batch):
    """Takes a single forward pass, returning only the loss."""
    loss = jax_forward(rng, workload, tstate, batch)
    return jax.lax.pmean(loss, axis_name='batch')
jax_pmapped_forward = lambda _r, _w, _t, _b: _jax_pmapped_forward(jax.random.split(_r, jax.local_device_count()), _w, _t, _b)[0]

@functools.partial(jax.jit, static_argnums=(1,))
def jax_train_step(rng: jax.random.PRNGKey, 
                   workload: spec.Workload,
                   tstate: JaxTrainState,
                   batch) -> Tuple[JaxTrainState, float, jax.Array]:
    """Takes a single training step, returning the new `tstate` as well as the loss and the gradients."""
    forward_rng, update_rng = jax.random.split(rng)
    def _loss_fn(params):
        """Loss function used for training."""
        return _forward(workload, params, tstate.model_state, batch, forward_rng, spec.ForwardPassMode.TRAIN)
    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
    (loss, (new_model_state,)), grad = grad_fn(tstate.params)

    updates, new_optimizer_state = tstate.tx.update(grad, tstate.opt_state, params=tstate.params, batch=batch, rng=update_rng, model_state=tstate.model_state)
    updated_params = optax.apply_updates(tstate.params, updates)
    def fix_shapes(_p1, _p2):
        if _p1.shape != _p2.shape: _p2 = _p2.reshape(_p1.shape)
        return _p2
    updated_params = jax.tree_map(fix_shapes, tstate.params, updated_params)
    tstate = tstate.replace(opt_state=new_optimizer_state, params=updated_params, model_state=new_model_state)
    return tstate, {'loss': loss, 'grad_sq_norm': sq_norm_pytree(grad)}


@functools.partial(jax.pmap, axis_name='batch', in_axes=(0, None, 0, 0), out_axes=(0, None), static_broadcasted_argnums=(1,))
def _jax_pmapped_train_step(rng: jax.random.PRNGKey, 
                           workload: spec.Workload,
                           tstate: JaxTrainState,
                           batch) -> Tuple[JaxTrainState, float, jax.Array]:
    """Takes a single training step, returning the new `tstate` as well as the loss and the gradients."""
    forward_rng, update_rng = jax.random.split(rng)
    def _loss_fn(params):
        """Loss function used for training."""
        return _forward(workload, params, tstate.model_state, batch, forward_rng, spec.ForwardPassMode.TRAIN)
    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
    (loss, (new_model_state,)), grad = grad_fn(tstate.params)
    
    # Get correct global mean loss and grad.
    (loss, grad) = jax.lax.pmean((loss, grad), axis_name='batch')

    updates, new_optimizer_state = tstate.tx.update(grad, tstate.opt_state, params=tstate.params, batch=batch, rng=update_rng, model_state=tstate.model_state)
    updated_params = optax.apply_updates(tstate.params, updates)
    tstate = tstate.replace(opt_state=new_optimizer_state, params=updated_params, model_state=new_model_state)
    return tstate, {'loss': loss, 'grad_sq_norm': sq_norm_pytree(grad)}
jax_pmapped_train_step = lambda _r, _w, _t, _b: _jax_pmapped_train_step(jax.random.split(_r, jax.local_device_count()), _w, _t, _b)
