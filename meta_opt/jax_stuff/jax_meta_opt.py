from absl import logging
from typing import Any, Tuple, Callable, Dict
import jax
import jax.numpy as jnp
from flax import struct, core
import optax
from optax._src import base
import chex

from algorithmic_efficiency import spec

from configs.optimizers import MetaOptConfig
from meta_opt.nn import TrainState
from meta_opt.utils import bcolors
from meta_opt.jax_stuff.jax_utils import append, add_pytrees, slice_pytree

# --------------------------------------------------------------------------------------------------------------------
# --------------------   DEFINE THE GPC CONTROLLER TO USE IN META-OPT  -----------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

class JaxMetaOptGPCState(struct.PyTreeNode):
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState = struct.field(pytree_node=True)
    cparams: core.FrozenDict[str, Any] = struct.field(pytree_node=True)  # dict of pytrees of disturbance-feedback control matrices (which themselves may be in pytree form if they multiply a pytree)
    
    H: int = struct.field(pytree_node=False)  # history of the controller, how many past disturbances to use for control
    HH: int = struct.field(pytree_node=False)  # history of the system, how many hallucination steps to take

    @classmethod
    def create(cls, 
               workload: spec.Workload,
               cfg: MetaOptConfig,
               meta_optimizer: optax.GradientTransformation,
               ):
        # make controller
        H, HH, m_method = cfg.H, cfg.HH, cfg.m_method
        params_zeros_like = jax.tree_map(lambda s: jnp.zeros(s.shape_tuple), workload.param_shapes)
        if m_method == 'scalar': M = jnp.zeros((H,))
        elif m_method == 'diagonal': M = jax.tree_map(lambda p: jnp.zeros((H, jnp.prod(jnp.array(p.shape)))), params_zeros_like)
        elif m_method == 'full': M = jax.tree_map(lambda p: jnp.zeros((H, jnp.prod(jnp.array(p.shape)), jnp.prod(jnp.array(p.shape)))), params_zeros_like)
        else: raise NotImplementedError(m_method)
        if cfg.fake_the_dynamics: M = jax.tree_map(lambda p: p.at[-1].set(-cfg.initial_learning_rate), M)
        cparams = {'M': M}

        # make the optimizer for `cparams`
        tx = meta_optimizer
        if cfg.meta_grad_clip is not None: tx = optax.chain(optax.clip(cfg.meta_grad_clip), tx)  # clip meta grads
        opt_state = tx.init(cparams)

        return cls(cparams=cparams,
                   H=H, HH=HH,
                   tx=tx, opt_state=opt_state)


@jax.jit
def _compute_control_scalar(M, disturbances): return jax.tree_map(lambda d: (jax.lax.expand_dims(M, range(1, d.ndim)) * d).sum(axis=0), disturbances)

@jax.jit
def _compute_control_full(M, disturbances): 
    def fn(m, d):
        s = d[0].shape
        d = d.reshape(d.shape[0], -1)
        if m.ndim == 2: return (m * d).sum(axis=0).reshape(s)
        else: return jax.lax.batch_matmul(m, jnp.expand_dims(d, axis=-1)).sum(axis=0).reshape(s)
        
    return jax.tree_map(fn, M, disturbances)

def compute_control(cparams, disturbances):
    M = cparams['M']
    if isinstance(M, jnp.ndarray): return _compute_control_scalar(M, disturbances)
    else: return _compute_control_full(M, disturbances)


# --------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------   Counterfactual Update  -------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

def _compute_loss_counterfactual_scan(cparams, H, HH, initial_tstate, 
                                      disturbances,  # past H + HH disturbances
                                      batches,  # past HH batches, starting at the one that would have been used to evolve `initial_params`
                                      curr_batch,  #  the current one
                                      train_step_fn,
                                      forward_fn,
                                      rng):
    # the scanning way
    def _evolve(carry, batch):
        tstate, h, step_rng = carry
        step_rng, train_step_rng = jax.random.split(step_rng)
        tstate, _ = train_step_fn(train_step_rng, tstate, batch)
        params = add_pytrees(tstate.params, compute_control(cparams, slice_pytree(disturbances, h, H)))
        tstate = tstate.replace(params=params)  # undo the tstep increase gotten from `train_step_fn` so that scan doesnt yell at us
        return (tstate, h + 1, step_rng), None
    (tstate, _, rng), _ = jax.lax.scan(_evolve, (initial_tstate, 0, rng), batches)
    return forward_fn(rng, tstate, curr_batch)


def _compute_loss_counterfactual_loop(cparams, H, HH, initial_tstate, 
                                      disturbances,  # past H + HH disturbances
                                      batches,  # past HH batches, starting at the one that would have been used to evolve `initial_params`
                                      curr_batch,  #  the current one
                                      train_step_fn,
                                      forward_fn,
                                      rng):
    tstate = initial_tstate
    for h in range(HH):
        rng, train_step_rng = jax.random.split(rng)
        batch = {k: v[h] for k, v in batches.items()}
        temp, _ = train_step_fn(train_step_rng, tstate, batch)
        del tstate
        params = add_pytrees(temp.params, compute_control(cparams, slice_pytree(disturbances, h, H)))
        tstate = temp.replace(params=params)
        del params, temp
    return forward_fn(rng, tstate, curr_batch)


@jax.jit
def counterfactual_update(cstate: JaxMetaOptGPCState,
           initial_tstate: TrainState,  # tstate from HH steps ago
           disturbances,  # past H + HH disturbances
           batches,  # past HH batches, starting at the one that would have been used to evolve `initial_params`
           curr_batch,  #  the current one
           train_step_fn,
           forward_fn,
           loss_fn,  # one of the two above
           rng):
    
    grads = jax.grad(loss_fn, (0,))(cstate.cparams, cstate.H, cstate.HH, initial_tstate, disturbances, batches, curr_batch, train_step_fn, forward_fn, rng)    
    updates, new_opt_state = cstate.tx.update(grads[0], cstate.opt_state, cstate.cparams)
    cparams = optax.apply_updates(cstate.cparams, updates)
    cparams = jax.lax.pmean(cparams, axis_name='batch')  # TODO check if we should be averaging cparams
    return cstate.replace(cparams=cparams, opt_state=new_opt_state), grads


# --------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------   OPTAX OPTIMIZER  -------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

class JaxMetaOptState(struct.PyTreeNode):
    cfg: MetaOptConfig = struct.field(pytree_node=False)
    t: int = struct.field(pytree_node=False)

    # parts of optimizer state that we need
    cstate: JaxMetaOptGPCState = struct.field(pytree_node=True)
    disturbance_history: chex.ArrayTree = struct.field(pytree_node=True)
    tstate_history: chex.ArrayTree = struct.field(pytree_node=True)
    batch_history: chex.ArrayTree = struct.field(pytree_node=True)

    # for rescaling the gradients
    disturbance_transform: optax.GradientTransformation = struct.field(pytree_node=False)
    disturbance_transform_state: optax.OptState = struct.field(pytree_node=True)

    # statistics to log during training
    grad_Ms: chex.ArrayTree = struct.field(pytree_node=True)

    def reset(self,
              _: jax.random.PRNGKey,
              workload: spec.Workload,
              reset_cstate: bool) -> optax.OptState:
        
        params_zeros_like = jax.tree_map(lambda s: jnp.zeros(s.shape_tuple), workload.param_shapes)
        H, HH = self.cstate.H, self.cstate.HH
        t = 0
        disturbance_history = jax.tree_map(lambda p: jnp.zeros((H, *p.shape)), params_zeros_like)
        tstate_history = (None,) * (HH + 1)
        batch_history = None
        disturbance_transform_state = self.disturbance_transform.init(params_zeros_like)
        ret = self.replace(t=t, disturbance_history=disturbance_history, tstate_history=tstate_history, batch_history=batch_history, disturbance_transform_state=disturbance_transform_state)
        if reset_cstate: 
            logging.warning(f'{bcolors.WARNING}{bcolors.BOLD}we also reset the meta-cstate! are you sure? this would prevent episodic learning...{bcolors.ENDC}')
            cstate = JaxMetaOptGPCState.create(workload, self.cfg, self.cstate.tx)
            ret = ret.replace(cstate=cstate)
        return ret
    
    def get_logging_metrics(self):
        ret = {}
        num_devices = jax.local_device_count()
        Ms = jax.tree_map(lambda p: p.reshape(num_devices, self.cstate.H, -1).mean(axis=-1).mean(axis=0), self.cstate.cparams['M'])
        if self.grad_Ms is not None:
            grad_Ms = jax.tree_map(lambda p: p.reshape(num_devices, self.cstate.H, -1).mean(axis=-1).mean(axis=0), self.grad_Ms[0]['M'])
        else:
            grad_Ms = jax.tree_map(jnp.zeros_like, Ms)
        if self.cfg.m_method == 'scalar':
            Ms = Ms.reshape(-1)[::-1]  # reverse so first idx is coefficient for most recent grad
            grad_Ms = grad_Ms.reshape(-1)[::-1]  # reverse so first idx is coefficient for most recent grad
        else: 
            Ms = jnp.stack(jax.tree_util.tree_flatten(Ms)[0]).mean(axis=0)[::-1]  # reverse so first idx is coefficient for most recent grad
            grad_Ms = jnp.stack(jax.tree_util.tree_flatten(grad_Ms)[0]).mean(axis=0)[::-1]  # reverse so first idx is coefficient for most recent grad
        assert Ms.shape == (self.cstate.H,), (Ms.shape, self.cstate.H)
        if not self.cfg.fake_the_dynamics: Ms = Ms.at[0].add(-self.cfg.initial_learning_rate)  # add the effective learning rate to most recent grad coeff
        ret.update({f'M_{i}': m for i, m in enumerate(Ms.reshape(-1))})
        ret.update({f'grad_M_{i}': grad_m for i, grad_m in enumerate(grad_Ms.reshape(-1))})
        return ret


def jax_meta_opt(workload: spec.Workload,
                 cfg: MetaOptConfig, 
                 meta_optimizer: optax.GradientTransformation,
                 empty_tstate: TrainState,
                 train_step_fn: Callable[[jax.random.PRNGKey, spec.Workload, TrainState, Any], Tuple[TrainState, float, spec.ParameterContainer]],
                 forward_fn: Callable[[jax.random.PRNGKey, spec.Workload, TrainState, Any], float]):
    
    initial_cstate = JaxMetaOptGPCState.create(workload, cfg, meta_optimizer)
    H, HH = initial_cstate.H, initial_cstate.HH
    if cfg.freeze_meta_params:
        logging.warning(f'{bcolors.WARNING}{bcolors.BOLD}the meta-opt controller is frozen! optimizer behavior wont change over time{bcolors.ENDC}')
    else:
        if cfg.fake_the_dynamics:
            logging.warning(f'{bcolors.WARNING}{bcolors.BOLD}we will be faking the dynamics during rollouts. will be faster, but behavior may be different...{bcolors.ENDC}')
        
    if cfg.scale_by_adam_betas:
        b1, b2 = cfg.scale_by_adam_betas
        disturbance_transform = optax.scale_by_adam(b1=b1, b2=b2)
    else:
        disturbance_transform = optax.identity()
    if cfg.weight_decay is not None: disturbance_transform = optax.chain(optax.add_decayed_weights(cfg.weight_decay), disturbance_transform)
    if cfg.grad_clip is not None: disturbance_transform = optax.chain(optax.clip(cfg.grad_clip), disturbance_transform)
    meta_opt_state = JaxMetaOptState(cfg=cfg, cstate=initial_cstate, t=0, 
                                     disturbance_history=None, tstate_history=None, batch_history=None,
                                     disturbance_transform=disturbance_transform, disturbance_transform_state=None,
                                     grad_Ms=None)

    _train_step_fn = jax.tree_util.Partial(lambda _r, _t, _b: train_step_fn(_r, workload, _t, _b))
    _forward_fn = jax.tree_util.Partial(lambda _r, _t, _b: forward_fn(_r, workload, _t, _b))
    _loss_fn = jax.tree_util.Partial(_compute_loss_counterfactual_scan if cfg.jax_compute_loss_with_scan else _compute_loss_counterfactual_loop)

    @jax.jit
    def init_fn(_):
        return (meta_opt_state.reset(_, workload=workload, reset_cstate=False), optax.EmptyState())
    
    @jax.jit
    def update_fn(grads, opt_state: JaxMetaOptState, params, **extra_args):
        assert params is not None, 'failed to provide parameters to the meta-optimizer'
        assert 'batch' in extra_args, 'failed to provide data batch to the meta-optimizer'
        assert 'model_state' in extra_args, 'failed to provide model state to the meta-optimizer'
        assert 'rng' in extra_args, 'failed to provide rng to the meta-optimizer'

        opt_state, _ = opt_state

        disturbances, disturbance_transform_state = opt_state.disturbance_transform.update(grads, opt_state.disturbance_transform_state, params=params)
        disturbance_history = jax.tree_map(append, opt_state.disturbance_history, disturbances)
        control = compute_control(opt_state.cstate.cparams, slice_pytree(disturbance_history, HH, H))  # use past H disturbances, including most recent one
        
        if not cfg.freeze_meta_params:
            # prologue
            batch, model_state, rng = extra_args['batch'], extra_args['model_state'], extra_args['rng']
            tstate = empty_tstate.replace(params=params, model_state=model_state)
            if opt_state.batch_history is None: batch_history = {k: jnp.stack([v for _ in range(HH)]) for k, v in batch.items()}
            else: batch_history = opt_state.batch_history

            # compute counterfactual update
            if opt_state.t >= H + HH:
                initial_tstate, batches, curr_batch = opt_state.tstate_history[0], batch_history, batch
                if cfg.jax_pmap_in_rollouts:
                    # Because we are using this `update_fn` within a pmapped train step (and we will need to call a pmapped train step again),
                    # we add an extra dimension back to the batches and tstate
                    batches = {k: v[:, None] for k, v in batches.items()}
                    expand_fn = lambda pytree: jax.tree_map(lambda v: v[None], pytree)
                    curr_batch = expand_fn(curr_batch)
                    initial_tstate = initial_tstate.replace(params=expand_fn(initial_tstate.params), model_state=expand_fn(initial_tstate.model_state))
                if cfg.freeze_batch_during_rollouts:  # makes sure rollout uses only 1 batch
                    batches = {k: v.at[:].set(curr_batch[k]) for k, v in batches.items()}
                cstate, grad_Ms = counterfactual_update(opt_state.cstate, initial_tstate, opt_state.disturbance_history, batches, curr_batch, 
                                            _train_step_fn, _forward_fn, _loss_fn, rng)
            else: cstate, grad_Ms = opt_state.cstate, opt_state.grad_Ms

            # epilogue
            # tstate = tstate.replace(params=add_pytrees(tstate.params, control))
            tstate_history = opt_state.tstate_history[1:] + (tstate,)
            for k in batch_history.keys(): 
                batch_history[k] = append(batch_history[k], batch[k]) 

            opt_state = opt_state.replace(cstate=cstate, disturbance_history=disturbance_history, disturbance_transform_state=disturbance_transform_state, 
                                        tstate_history=tstate_history, batch_history=batch_history, t=opt_state.t+1, grad_Ms=grad_Ms)
        else:
            opt_state = opt_state.replace(disturbance_history=disturbance_history, disturbance_transform_state=disturbance_transform_state, t=opt_state.t+1)
        
        # if we dont fake the dynamics, the gradient step with `initial_learning_rate` never actually materializes since it is not part of the control
        # we add it back in here
        if not cfg.fake_the_dynamics:  
            control = add_pytrees(control, jax.tree_map(lambda v: -cfg.initial_learning_rate * v, disturbances))  # if `scale_by_adam`, this will apply on top of that

        return control, (opt_state, optax.EmptyState())
    
    return base.GradientTransformation(init_fn, update_fn)
