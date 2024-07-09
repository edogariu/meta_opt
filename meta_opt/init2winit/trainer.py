# coding=utf-8
# Copyright 2024 The init2winit Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Episodic trainer for the init2winit project. Forked from standard trainer."""
import itertools
import time

from absl import logging

from init2winit import utils
from init2winit.trainer_lib import base_trainer
from init2winit.trainer_lib import trainer_utils
from init2winit.trainer_lib.trainer import update
import jax

# ------------------------------------------------------------------------------------
# dogariu: I TOUCHED THIS CODE
from flax import jax_utils
from meta_opt.experiment import ExperimentConfig
from meta_opt.optimizers.base import OptimizerConfig
from meta_opt.utils import bcolors
# ------------------------------------------------------------------------------------


class EpisodicTrainer(base_trainer.BaseTrainer):
  """Episodic trainer."""

  def train(self, 
            experiment_cfg: ExperimentConfig, 
            optimizer_cfg: OptimizerConfig):
    """All training logic.

    The only side-effects are:
      - Initiailizing self._time_at_prev_eval_end to the current time
      - Initiailizing self._prev_eval_step to the current step

    Yields:
      metrics: A dictionary of all eval metrics from the given epoch.
    """
    # NOTE: the initialization RNG should *not* be per-host, as this will create
    # different sets of weights per host. However, all other RNGs should be
    # per-host.
    # TODO(znado,gilmer,gdahl): implement replicating the same initialization
    # across hosts.
    rng, init_rng = jax.random.split(self._rng)
    rng = jax.random.fold_in(rng, jax.process_index())
    rng, data_rng = jax.random.split(rng)
    rng, callback_rng = jax.random.split(rng)

    if jax.process_index() == 0:
      logging.info('Let the training begin!')
      logging.info('Dataset input shape: %r', self._hps.input_shape)
      logging.info('Hyperparameters: %s', self._hps)

    self._setup_and_maybe_restore(init_rng, data_rng, callback_rng, update)

    # ----------------------------------------------------------------------------
    # dogariu: I TOUCHED THIS CODE
    if experiment_cfg.full_batch:
      self._dataset.train_iterator_fn = self._dataset.train_iterator_fn
    # -----------------------------------------------------------------------------

    if jax.process_index() == 0:
      trainer_utils.log_message(
        'Starting training!', self._logging_pool, self._xm_work_unit)

    train_iter = itertools.islice(
      self._dataset.train_iterator_fn(),
      self._num_train_steps,
    )

    train_iter = trainer_utils.prefetch_input_pipeline(
      train_iter, self._hps.num_device_prefetches)

    if self._data_selector:
      train_iter = self._data_selector(
        train_iter,
        optimizer_state=self._optimizer_state,
        params=self._params,
        batch_stats=self._batch_stats,
        hps=self._hps,
        global_step=self._global_step,
        constant_base_rng=rng)
      
    # ----------------------------------------------------------------------------
    # dogariu: I TOUCHED THIS CODE
    if experiment_cfg.full_batch:
      frozen_batch = next(train_iter)
      def _same_batch_generator():
          while True: yield frozen_batch
      train_iter = _same_batch_generator()
    # ----------------------------------------------------------------------------

    start_time = time.time()
    start_step = self._global_step

    # NOTE(dsuo): record timestamps for run_time since we don't have a duration
    # that we can increment as in the case of train_time.
    self._time_at_prev_eval_end = start_time
    self._prev_eval_step = self._global_step

    if self._global_step in self._checkpoint_steps:
      self._save(self._checkpoint_dir, max_to_keep=None)

# ------------------------------------------------------------------------
# dogariu: I TOUCHED THIS CODE
    num_episodes = experiment_cfg.num_episodes
    for episode_i in range(1, num_episodes + 1):
      # reset for the episode
      rng, init_rng = jax.random.split(rng)
      logging.info(f'{bcolors.OKBLUE}{bcolors.BOLD}Resetting model!{bcolors.ENDC}')

      unreplicated_params, unreplicated_batch_stats = self._model.initialize(
        self._initializer,
        self._hps,
        init_rng,
        self._init_logger,)
      self._params, self._batch_stats = jax_utils.replicate(unreplicated_params), jax_utils.replicate(unreplicated_batch_stats)

      if optimizer_cfg.reset_opt_state:
        logging.info(f'{bcolors.OKBLUE}{bcolors.BOLD}Also resetting optimizer state!{bcolors.ENDC}')
        optimizer_init_fn, self._optimizer_update_fn = optimizer_cfg.make_jax()
        unreplicated_optimizer_state = optimizer_init_fn(unreplicated_params)
        self._optimizer_state = jax_utils.replicate(unreplicated_optimizer_state)
      logging.warn('@EVAN DONT FORGET: handle replication of opt state and also DONT RESET THE Ms for metaopt')
        # raise RuntimeError('handle replication of opt state and also DONT RESET THE Ms for metaopt')

      if num_episodes > 1: logging.info(f'{bcolors.FAIL}{bcolors.BOLD}Starting training episode {episode_i}.{bcolors.ENDC}')
# ------------------------------------------------------------------------
      for _ in range(start_step, self._num_train_steps):
        with jax.profiler.StepTraceAnnotation('train', step_num=self._global_step):
          # NOTE(dsuo): to properly profile each step, we must include batch
          # creation in the StepTraceContext (as opposed to putting `train_iter`
          # directly in the top-level for loop).
          batch = next(train_iter)

          lr = self._lr_fn(self._global_step)
          # It looks like we are reusing an rng key, but we aren't.
          # TODO(gdahl): Make it more obvious that passing rng is safe.
          # TODO(gdahl,gilmer,znado): investigate possibly merging the member
          # variable inputs/outputs of this function into a named tuple.
          (self._optimizer_state, self._params, self._batch_stats,
          self._sum_train_cost,
          self._metrics_state, self._grad_norm) = self._update_pmapped(
            self._optimizer_state, self._params, self._batch_stats,
            self._metrics_state, batch, self._global_step, lr, rng,
            self._local_device_indices, self._sum_train_cost)
          self._global_step += 1
          if self._global_step in self._checkpoint_steps:
            self._save(self._checkpoint_dir, max_to_keep=None)

          lr = trainer_utils.fetch_learning_rate(self._optimizer_state)
          # TODO(gdahl, gilmer): consider moving this test up.
          # NB: Since this test is after we increment self._global_step, having 0
          # in eval_steps does nothing.
          if trainer_utils.should_eval(self._global_step, self._eval_frequency, self._eval_steps):
            try:
              report = self._eval(lr, start_step, start_time)
            except utils.TrainingDivergedError as e:
              self.wait_until_orbax_checkpointer_finished()
              raise utils.TrainingDivergedError(
                f'divergence at step {self._global_step}'
              ) from e
            yield report
            if self._check_early_stopping(report):
              return

    # Always log and checkpoint on host 0 at the end of training.
    # If we moved where in the loop body evals happen then we would not need
    # this test.
    if self._prev_eval_step != self._num_train_steps:
      report = self._eval(lr, start_step, start_time)
      yield report
    # To make sure the last checkpoint was correctly saved.
    self.wait_until_orbax_checkpointer_finished()
