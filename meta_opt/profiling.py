# assert 'batch' in MESH.axis_names

from meta_opt.optimizers.base import OptimizerConfig
from meta_opt.optimizers.sgd import SGDConfig
from meta_opt.optimizers.adamw import AdamWConfig
from meta_opt.optimizers.metaopt import MetaOptConfig

import tqdm
import functools
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from flax import jax_utils, struct
import optax

# Create a Sharding object to distribute a value across devices:
NUM_DEVICES = jax.local_device_count()
BATCH_NUM_DEVICES = 8
OPT_STATE_NUM_DEVICES = 1
assert BATCH_NUM_DEVICES * OPT_STATE_NUM_DEVICES == NUM_DEVICES

devices = mesh_utils.create_device_mesh((BATCH_NUM_DEVICES, OPT_STATE_NUM_DEVICES))
MESH = Mesh(devices, axis_names=('batch', 'opt'))
print(MESH)

# hyperparams
NUM_PARAMS = 1117392  # cifar has 11173960, wmt has 133521408
NUM_DATA = 200
BATCH_SIZE = 32

# -------------------------------------------------------------------------
# jax training utilities
# -------------------------------------------------------------------------
@jax.jit
def make_params(seed: int):
    return jax.random.normal(jax.random.PRNGKey(seed), (NUM_PARAMS,))

@jax.jit
def make_batch(seed: int):
    return jax.random.normal(jax.random.PRNGKey(seed), (BATCH_SIZE, NUM_DATA))

@jax.jit
def make_opt_state(params, opt):
    return opt[0](params)

@jax.jit
def loss_fn(params, batch):
    return abs((batch * params.reshape(1, -1)[:, :NUM_DATA]).mean()) / (1e-6 + jnp.linalg.norm(params))

# so that we can pass the loss function to metaopt's update fn
@struct.dataclass
class LossFn(struct.PyTreeNode):
    batch: jax.Array = struct.field(pytree_node=True)

    def __call__(self, params):
        return jnp.mean(loss_fn(params, self.batch))

@jax.jit
def train_step(batch, params, opt_state, opt):
    loss_fn = LossFn(batch)
    grads = jax.grad(loss_fn)(params)
    updates, opt_state = opt[1](grads, opt_state, params, cost_fn=loss_fn)
    params = optax.apply_updates(params, updates)
    return params, opt_state

from optax._src.transform import ScaleByAdamState
from meta_opt.optimizers.metaopt import JaxMetaOptState
from dataclasses import asdict

def _shard_adam(opt_state):
    shardings = {
        # to replicate
        'count': NamedSharding(MESH, P()),

        # to shard
        'mu': NamedSharding(MESH, P('opt')),
        'nu': NamedSharding(MESH, P('opt')),
    }
    
    shardings = ScaleByAdamState(**shardings)
    sharded_opt_state = jax.tree_map(jax.device_put, opt_state, shardings)
    return sharded_opt_state, optax.EmptyState()

def _shard_metaopt(opt_state):
    # replicated = opt_state.disturbance_history.ndim == 3
    # print(replicated)
    # shardings = {
    #     # # to shard along H axis
    #     # 'disturbance_history': NamedSharding(MESH, P('opt', None) if not replicated else P(None, 'opt', None)),
    #     # 'param_history': NamedSharding(MESH, P(None, None) if not replicated else P(None, None, None)),

    #     # to shard along num_params axis
    #     'disturbance_history': NamedSharding(MESH, P(None, 'opt') if not replicated else P('batch', None, 'opt')),
    #     'param_history': NamedSharding(MESH, P(None, 'opt') if not replicated else P('batch', None, 'opt')),
    # }
    
    shardings = {
        # to shard along num_params axis
        'disturbance_history': NamedSharding(MESH, P(None, 'opt')),
        'param_history': NamedSharding(MESH, P(None, 'opt')),
    }

    ad = asdict(opt_state)
    ad = {k: ad[k] for k in shardings.keys()}
    sharded_opt_state = opt_state.replace(**jax.device_put(ad, shardings))
    
    return sharded_opt_state, optax.EmptyState()

def shard_opt_state(opt_state):
    opt_state, _ = opt_state
    if isinstance(opt_state, ScaleByAdamState):
        return _shard_adam(opt_state)
    elif isinstance(opt_state, JaxMetaOptState):
        return _shard_metaopt(opt_state)
    else:
        raise NotImplementedError(opt_state.__class__)