# Things to change in `init2winit` codebase 
Here is a list of the changes that must be made in Google's internal `init2winit` codebase so that our stuff runs.

### Add metaopt to `optimizer_lib`
Add to `init2winit/optimizer_lib/optimizers.py::get_optimizer(...)` the lines 
```
elif hps.optimizer == 'metaopt':
    metaopt_cfg = hps['optimizer_cfg']
    def metaopt_fn(learning_rate: float): return metaopt_cfg.replace(base_learning_rate=learning_rate).make_jax()
    opt_init, opt_update = utils.static_inject_hyperparams(metaopt_fn)(
        learning_rate=0.0,  # Manually injected on each train step
    )
    optimizer_requires_cost_fn = True
```
so that, as long as a `MetaOptConfig` is placed in `hps['optimizer_cfg']`, we can proceed.

### Add episodic and fullbatch training to `trainer_lib`
Import `custom_trainer.CustomTrainer` and add to `init2winit/trainer_lib/trainers.py::_ALL_TRAINERS` the lines 
```
_ALL_TRAINERS = {
    'standard': trainer.Trainer,
    'custom': CustomTrainer,
}
```
so that, as long as an `ExperimentConfig` is placed in `hps['experiment_cfg']` and an `OptimizerConfig` is placed in `hps['optimizer_cfg']`, we can proceed. We also need to add 
```
optimizer_init_fn, optimizer_update_fn = optimizers.get_optimizer(
    self._hps, self._model, batch_axis_name='batch')
unreplicated_optimizer_state = optimizer_init_fn(unreplicated_params)
self._optimizer_init_fn = optimizer_init_fn
```
to `init2winit/trainer_lib/base_trainer.py::setup_and_maybe_restore()` to expose `self._optimizer_init_fn` for episodic resets.
