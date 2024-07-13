# How to run in `init2winit` codebase 
Here is a list of the changes that must be made in Google's internal `//third_party/py/init2winit` codebase so that our stuff runs.

### Link imports/builds/things
Our config file will need to be able to import `meta_opt.optimizers.*.py` (which in turn needs `meta_opt.utils.py`) as well as `meta_opt.experiment.py` and `meta_opt.init2winit.config_utils.py`. The easiest way to do this is simply to clone this repo into `init2winit/experiments/` next to `base_config.py` and paste this addition to the `init2winit/experiments/BUILD` file.
```python
load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")
py_library(
    name = "config_utils",
    srcs = ["meta_opt/meta_opt/init2winit/config_utils.py",],
    srcs_version = "PY3",
    visibility = ["//third_party/py/init2winit:__subpackages__"],
    deps = [
        "//third_party/py/ml_collections/config_dict",  # buildcleaner: keep
    ],
)

py_library(
    name = "experiment",
    srcs = ["meta_opt/meta_opt/experiment.py",],
    srcs_version = "PY3",
    visibility = ["//third_party/py/init2winit:__subpackages__"],
    deps = [
        "//third_party/py/flax",  # buildcleaner: keep
    ],
)

py_library(
    name = "utils",
    srcs = ["meta_opt/meta_opt/utils.py",],
    srcs_version = "PY3",
    visibility = ["//third_party/py/init2winit:__subpackages__"],
    deps = [
        "//third_party/py/jax",  # buildcleaner: keep
    ],
)

py_library(
    name = "base",
    srcs = ["meta_opt/meta_opt/optimizers/base.py",],
    srcs_version = "PY3",
    visibility = ["//third_party/py/init2winit:__subpackages__"],
    deps = [
        "//third_party/py/flax",  # buildcleaner: keep
        "//third_party/py/optax",  # buildcleaner: keep
    ],
)

py_library(
    name = "sgd",
    srcs = ["meta_opt/meta_opt/optimizers/sgd.py",],
    srcs_version = "PY3",
    visibility = ["//third_party/py/init2winit:__subpackages__"],
    deps = [
        "//third_party/py/flax",  # buildcleaner: keep
        "//third_party/py/optax",  # buildcleaner: keep
        "//third_party/py/init2winit/experiments:base",  # buildcleaner: keep
    ],
)

py_library(
    name = "adamw",
    srcs = ["meta_opt/meta_opt/optimizers/adamw.py",],
    srcs_version = "PY3",
    visibility = ["//third_party/py/init2winit:__subpackages__"],
    deps = [
        "//third_party/py/flax",  # buildcleaner: keep
        "//third_party/py/optax",  # buildcleaner: keep
        "//third_party/py/init2winit/experiments:base",  # buildcleaner: keep
    ],
)

py_library(
    name = "metaopt",
    srcs = ["meta_opt/meta_opt/optimizers/metaopt.py",],
    srcs_version = "PY3",
    visibility = ["//third_party/py/init2winit:__subpackages__"],
    deps = [
        "//third_party/py/chex",  # buildcleaner: keep
        "//third_party/py/flax",  # buildcleaner: keep
        "//third_party/py/jax",  # buildcleaner: keep
        "//third_party/py/optax",  # buildcleaner: keep
        "//third_party/py/init2winit/experiments:base",  # buildcleaner: keep
        "//third_party/py/init2winit/experiments:sgd",  # buildcleaner: keep
        "//third_party/py/init2winit/experiments:adamw",  # buildcleaner: keep
        "//third_party/py/init2winit/experiments:utils",  # buildcleaner: keep
    ],
)
```

### Add metaopt to `optimizer_lib`
Add to `init2winit/optimizer_lib/optimizers.py::get_optimizer(...)` the lines 
```python
elif hps.optimizer == 'metaopt':
    from init2winit.experiments import metaopt
    metaopt_cfg = metaopt.MetaOptConfig.fromdict(hps.opt_hparams['optimizer_cfg'])
    def metaopt_fn(learning_rate: float): return metaopt_cfg.replace(base_learning_rate=learning_rate).make_jax()
    opt_init, opt_update = utils.static_inject_hyperparams(metaopt_fn)(
        learning_rate=0.0,  # Manually injected on each train step
    )
    opt_update = lambda grads, opt_state, cost_fn_params_tuple: _opt_update(grads, opt_state,
                        params=cost_fn_params_tuple[1], cost_fn=cost_fn_params_tuple[0])
    optimizer_requires_cost_fn = True
```
so that, as long as a `MetaOptConfig` is placed in `hps['opt_hparams']['optimizer_cfg']`, we can proceed. We also have to add 
`"//learning/deepmind/python/adhoc_import:binary_import"` to the `BUILD` file list of deps for the `"optimizers"` target.

### Add passing the loss function to the optimizer to `trainer.Trainer`
 Add the following code to right before the optimizer update fn call of `init2winit/trainer_lib/trainer.py::update(...)`
```python
from flax import struct
@struct.dataclass
class LossFn(struct.PyTreeNode):
    rng: jax.Array = struct.field(pytree_node=True)
    batch_stats: jax.Array = struct.field(pytree_node=True)
    batch: jax.Array = struct.field(pytree_node=True)

    def __call__(self, params):
        return training_cost(
            params,
            batch=self.batch,
            batch_stats=self.batch_stats,
            dropout_rng=self.rng,
        )[0]
opt_cost = LossFn(rng, batch_stats, batch)

model_updates, new_optimizer_state = optimizer_update_fn(
    grad,
    optimizer_state,
    params=params,
    batch=batch,
    batch_stats=new_batch_stats,
    cost_fn=opt_cost,
    grad_fn=grad_fn,
    value=cost_value)
new_params = optax.apply_updates(params, model_updates)

new_metrics_state = None
if metrics_state is not None:
    new_metrics_state = metrics_update_fn(metrics_state, step, cost_value, grad,
                                          params, new_params, optimizer_state,
                                          new_batch_stats)
    try:
        curr_stats = new_optimizer_state[0].get_logging_metrics()
        for k, v in curr_stats.items():
            new_metrics_state[k] = metrics_state[k].at[step].set(v)
    except Exception as e:
        pass

return (new_optimizer_state, new_params, new_batch_stats,
      running_train_cost + cost_value, new_metrics_state, grad_norm)
```
and add `"//third_party/py/flax"` to the `trainer` target of `init2winit/trainer_lib/BUILD`.

### Add episodic and fullbatch training to `trainer.Trainer`
Add the following code to right before the train loop of `init2winit/trainer_lib/trainer.py::Trainer.train(...)`
```python
from flax import jax_utils
experiment_cfg, optimizer_cfg = self._hps.opt_hparams['experiment_cfg'], self.opt_hparams['optimizer_cfg']

# add the fullbatch part
if experiment_cfg.full_batch:
    frozen_batch = next(train_iter)
    def _same_batch_generator():
        while True: yield frozen_batch
    train_iter = _same_batch_generator()

# add the episodic part
num_episodes = experiment_cfg.num_episodes
for episode_i in range(1, num_episodes + 1):
    # reset for the episode
    rng, init_rng = jax.random.split(rng)
    logging.info('Resetting model!')

    unreplicated_params, unreplicated_batch_stats = self._model.initialize(
    self._initializer,
    self._hps,
    init_rng,
    self._init_logger,)
    self._params, self._batch_stats = jax_utils.replicate(unreplicated_params), jax_utils.replicate(unreplicated_batch_stats)

    if optimizer_cfg.reset_opt_state:
        logging.info('Also resetting optimizer state!')
        if optimizer_cfg.optimizer_name == 'MetaOpt':
            unreplicated_opt_state = jax_utils.unreplicate(self._optimizer_state)
            gpc_params, gpc_opt_state = unreplicated_opt_state[0].gpc_params, unreplicated_opt_state[0].gpc_opt_state
            unreplicated_optimizer_state = self._optimizer_init_fn(unreplicated_params)
            unreplicated_optimizer_state = (unreplicated_optimizer_state[0].replace(gpc_params=gpc_params, gpc_opt_state=gpc_opt_state),
                                            unreplicated_optimizer_state[1])
            logging.info('Resetting metaopt, so I am putting back the gpc params')
        else:
            unreplicated_optimizer_state = self._optimizer_init_fn(unreplicated_params)          
        self._optimizer_state = jax_utils.replicate(unreplicated_optimizer_state)
        logging.warn('@EVAN DONT FORGET: handle replication of opt state and also DONT RESET THE Ms for metaopt')

    if num_episodes > 1: logging.info(f'Starting training episode {episode_i}.')

    for _ in range(start_step, self._num_train_steps // num_episodes):
        ...
```
so that, as long as an `ExperimentConfig` is placed in `hps['experiment_cfg']` and an `OptimizerConfig` is placed in `hps['optimizer_cfg']`, we can proceed. We also need to add 
```python
optimizer_init_fn, optimizer_update_fn = optimizers.get_optimizer(
    self._hps, self._model, batch_axis_name='batch')
unreplicated_optimizer_state = optimizer_init_fn(unreplicated_params)
self._optimizer_init_fn = optimizer_init_fn
```
to `init2winit/trainer_lib/base_trainer.py::setup_and_maybe_restore(...)` to expose `self._optimizer_init_fn` for episodic resets.

### TODO: HANDLING SHARDING
we gotta set up opt_state sharding for i2w like we did for algoperf...

### Putting out fires
On line 499 in `init2winit/xmanager/launch_utils_v2.py`, there is a note for (znado,gdahl) to convert it to `config.to_json()`. Do this.
```python
if isinstance(config, config_dict.ConfigDict):
    config_json = config.to_json()
else:
    config_json = json.dumps(config_copy)
```
To ensure that we are correctly logging the optimizer state's training metrics, add
```python
opt_cfg = hps['opt_hparams']['optimizer_cfg']
if opt_cfg['optimizer_name'] == 'MetaOpt':
  history_len = opt_cfg['H']
  for h in range(history_len):
    metrics_state[f'M_{h}'] = jnp.zeros(num_train_steps)
    metrics_state[f'grad_M_{h}'] = jnp.zeros(num_train_steps)
  for k in ['gpc_cost', 
            'disturbance_history_memory',
            'param_history_memory',
            'cost_fn_history_memory',
            'disturbance_transform_state_memory',
            'total_metaopt_memory']:
    metrics_state[k] = jnp.zeros(num_train_steps)

return metrics_state
```
to the `init_fn(...)` definition of `init2winit/training_metrics_grabber.py::make_training_metrics(...)`.


### Running it
Now that we have set all this up, we can run a config simply by doing an `hgd` so that we are at `google3` and then executing
```bash
/google/bin/releases/xmanager/cli/xmanager.par --xm_deployment_env=alphabet launch third_party/py/init2winit/xmanager/launch_train_xm_v2.py -- --xm_resource_pool= --xm_resource_alloc= --undefok=xm_gxm_origin --xm_gxm_origin --xm_skip_launch_confirmation -- --xm_resource_pool=gdm --xm_skip_launch_confirmation --xm_resource_alloc=group:gdm/brain-pton --noxm_monitor_on_launch --xm_skip_launch_confirmation --config=third_party/py/init2winit/experiments/meta_opt/configs/test.py --use_fragmented_python --append_timestamp --skip_mitto --cns_group=dogariu
```
where the `--config=third_party/py/init2winit/experiments/meta_opt/configs/test.py` arg is filled in with the location of the config/sweep you wanna run.
