from typing import Any, Tuple, Dict
import functools
from absl import logging

import jax
import jax.numpy as jnp
import optax
from flax import struct, core, jax_utils

from algorithmic_efficiency import spec

from meta_opt.optimizers.base import OptimizerConfig
from meta_opt.utils import bcolors, get_size


# -------------------------------------------------------------------------------------------------
# ------------------------------------------ Optimizers -------------------------------------------
# -------------------------------------------------------------------------------------------------

def shard_or_replicate_opt_state(opt_state: optax.OptState):
    opt_state, _ = opt_state
    ret = jax_utils.replicate(opt_state)
    return (ret, optax.EmptyState())

# -------------------------------------------------------------------------------------------------
# ------------------------------------------ Train States -----------------------------------------
# -------------------------------------------------------------------------------------------------

class JaxTrainState(struct.PyTreeNode):
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
              reset_opt_state: bool):
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
        params_memory = get_size(self.params)
        model_state_memory = get_size(self.model_state)
        opt_state_memory = get_size(self.opt_state)
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
        ret['param_sq_norm'] = sum(jax.tree_util.tree_flatten(jax.tree_map(lambda p: (p * p).sum(), self.params))[0])
        return ret


def jax_create_train_state(rng: jax.random.PRNGKey,
                           workload: spec.Workload,
                           optimizer_cfg: OptimizerConfig,
                           distribute_opt_state: bool = True) -> JaxTrainState:
    """Creates a train state from scratch. This should initialize model parameters, auxiliary state, and optimizer state."""
    params, model_state = workload.init_model_fn(rng)
    opt = optimizer_cfg.make_jax()
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
    opt = optimizer_cfg.make_jax()
    return JaxTrainState(params=model_params, model_state=model_state, tx=opt, opt_state=opt_state, t=global_step)



# -------------------------------------------------------------------------------------------------
# ------------------------------------- Train Step Functions --------------------------------------
# -------------------------------------------------------------------------------------------------


def _forward(params, workload: spec.Workload, model_state, batch, rng, forward_pass_mode):
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

@struct.dataclass
class LossFn(struct.PyTreeNode):
    workload: spec.Workload = struct.field(pytree_node=False)
    rng: jax.Array = struct.field(pytree_node=True)
    model_state: jax.Array = struct.field(pytree_node=True)
    batch: jax.Array = struct.field(pytree_node=True)

    def __call__(self, params):
        return _forward(params, self.workload, self.model_state, self.batch, self.rng, spec.ForwardPassMode.TRAIN)[0]

@functools.partial(jax.pmap, axis_name='batch', in_axes=(0, None, 0, 0), out_axes=(0, None), static_broadcasted_argnums=(1,))
def _jax_pmapped_train_step(rng: jax.random.PRNGKey, 
                           workload: spec.Workload,
                           tstate: JaxTrainState,
                           batch) -> Tuple[JaxTrainState, float, jax.Array]:
    """Takes a single training step, returning the new `tstate` as well as the loss and the gradients."""
    grad_fn = jax.value_and_grad(lambda p: _forward(p, workload, tstate.model_state, batch, rng, spec.ForwardPassMode.TRAIN), has_aux=True)
    (loss, (new_model_state,)), grad = grad_fn(tstate.params)
    
    # Get correct global mean loss and grad.
    (loss, grad) = jax.lax.pmean((loss, grad), axis_name='batch')

    loss_fn = LossFn(workload, rng, tstate.model_state, batch)
    updates, new_optimizer_state = tstate.tx.update(grad, tstate.opt_state, params=tstate.params, loss_fn=loss_fn)
    updated_params = optax.apply_updates(tstate.params, updates)
    tstate = tstate.replace(opt_state=new_optimizer_state, params=updated_params, model_state=new_model_state)
    return tstate, {'loss': loss, 'grad_sq_norm': sum(jax.tree_util.tree_flatten(jax.tree_map(lambda p: (p * p).sum(), grad))[0])}
jax_pmapped_train_step = lambda _r, _w, _t, _b: _jax_pmapped_train_step(jax.random.split(_r, jax.local_device_count()), _w, _t, _b)
