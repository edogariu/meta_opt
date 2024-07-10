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

r"""Main file for the init2winit project.

"""

import functools
import json
import os
import struct
import sys
import time

from absl import app
from absl import flags
from absl import logging

from flax import jax_utils
from init2winit import hyperparameters
from init2winit import utils
from init2winit.dataset_lib import datasets
from init2winit.init_lib import initializers
from init2winit.model_lib import models
from init2winit.trainer_lib import trainers
import jax
from jax import lax
from ml_collections.config_dict import config_dict
import numpy as np
import tensorflow as tf

# --------------------------------------------------------------------------------
# dogariu: I TOUCHED THIS CODE
import importlib
from dataclasses import asdict
from algorithmic_efficiency.workloads import workloads
from algorithmic_efficiency import logger_utils
from meta_opt.experiment import ExperimentConfig
from meta_opt.optimizers.base import OptimizerConfig
from meta_opt.init2winit.logger import CustomMetricLogger
from meta_opt.init2winit.workload_defaults import MODELS, LOSSES, METRICS
from meta_opt.utils import bcolors, pretty_dict
# --------------------------------------------------------------------------------

gfile = tf.io.gfile

# For internal compatibility reasons, we need to pull this function out.
makedirs = tf.io.gfile.makedirs

# Setting jax default prng implementation to protect against jax defaults
# change.
jax.config.update('jax_default_prng_impl', 'threefry2x32')
jax.config.update('jax_threefry_partitionable', True)

# Enable flax xprof trace labelling.
os.environ['FLAX_PROFILE'] = 'true'

flags.DEFINE_string('trainer', 'standard', 'Name of the trainer to use.')
flags.DEFINE_string('model', 'fully_connected', 'Name of the model to train.')
flags.DEFINE_string('loss', 'cross_entropy', 'Loss function.')
flags.DEFINE_string('metrics', 'classification_metrics',
                    'Metrics to be used for evaluation.')
flags.DEFINE_string('initializer', 'noop', 'Must be in [noop, meta_init].')
flags.DEFINE_string('experiment_dir', None,
                    'Path to save weights and other results. Each trial '
                    'directory will have path experiment_dir/worker_id/.')
flags.DEFINE_string('dataset', 'mnist', 'Which dataset to train on.')
flags.DEFINE_string('data_selector', 'noop', 'Which data selector to use.')
flags.DEFINE_integer('num_train_steps', None, 'The number of steps to train.')
flags.DEFINE_integer(
    'num_tf_data_prefetches', -1, 'The number of batches to to prefetch from '
    'network to host at each step. Set to -1 for tf.data.AUTOTUNE.')
flags.DEFINE_integer(
    'num_device_prefetches', 0, 'The number of batches to to prefetch from '
    'host to device at each step.')
flags.DEFINE_integer(
    'num_tf_data_map_parallel_calls', -1, 'The number of parallel calls to '
    'make from tf.data.map. Set to -1 for tf.data.AUTOTUNE.'
)
flags.DEFINE_integer('eval_batch_size', None, 'Batch size for evaluation.')
flags.DEFINE_bool('eval_use_ema', None, 'If True evals will use ema of params.')
flags.DEFINE_integer(
    'eval_num_batches', None,
    'Number of batches for evaluation. Leave None to evaluate '
    'on the entire validation and test set.')
flags.DEFINE_integer(
    'test_num_batches', None,
    'Number of batches for eval on test set. Leave None to evaluate '
    'on the entire test set.')
flags.DEFINE_integer('eval_train_num_batches', None,
                     'Number of batches when evaluating on the training set.')
flags.DEFINE_integer('eval_frequency', 1000, 'Evaluate every k steps.')
flags.DEFINE_string(
    'hparam_overrides', '', 'JSON representation of a flattened dict of hparam '
    'overrides. For nested dictionaries, the override key '
    'should be specified as lr_hparams.base_lr.')
flags.DEFINE_string(
    'callback_configs', '', 'JSON representation of a list of dictionaries '
    'which specify general callbacks to be run during eval of training.')
flags.DEFINE_list(
    'checkpoint_steps', [], 'List of steps to checkpoint the'
    ' model. The checkpoints will be saved in a separate'
    'directory train_dir/checkpoints. Note these checkpoints'
    'will be in addition to the normal checkpointing that'
    'occurs during training for preemption purposes.')
flags.DEFINE_string('external_checkpoint_path', None,
                    'If this argument is set, the trainer will initialize'
                    'the parameters, batch stats, optimizer state, and training'
                    'metrics by loading them from the checkpoint at this path.')

flags.DEFINE_string(
    'early_stopping_target_name',
    None,
    'A string naming the metric to use to perform early stopping. If this '
    'metric reaches the value `early_stopping_target_value`, training will '
    'stop. Must include the dataset split (ex: validation/error_rate).')
flags.DEFINE_float(
    'early_stopping_target_value',
    None,
    'A float indicating the value at which to stop training.')
flags.DEFINE_enum(
    'early_stopping_mode',
    None,
    enum_values=['above', 'below'],
    help=(
        'One of "above" or "below", indicates if we should stop when the '
        'metric is above or below the threshold value. Example: if "above", '
        'then training will stop when '
        '`report[early_stopping_target_name] >= early_stopping_target_value`.'))
flags.DEFINE_integer(
    'early_stopping_min_steps',
    0,
    help='Only allows early stopping after at least this many steps.',
)
flags.DEFINE_list(
    'eval_steps', [],
    'List of steps to evaluate the model. Evaluating implies saving a '
    'checkpoint for preemption recovery.')
flags.DEFINE_string(
    'hparam_file', None, 'Optional path to hparam json file for overriding '
    'hyperparameters. Hyperparameters are loaded before '
    'applying --hparam_overrides.')
flags.DEFINE_list(
    'allowed_unrecognized_hparams', [],
    'Downgrades unrecognized hparam override keys from an error to a warning '
    'for the supplied list of keys.')
flags.DEFINE_string(
    'training_metrics_config', '',
    'JSON representation of the training metrics config.')

flags.DEFINE_integer('worker_id', 1,
                     'Client id for hparam sweeps and tuning studies.')

# --------------------------------------------------------------------------
# dogariu: I TOUCHED THIS CODE
flags.DEFINE_string('config_path', None, required=True,
    help='The relative path of the Python file containing the experiment and optimizer configs.'
         'NOTE: the config dir must have an __init__.py file!')
# --------------------------------------------------------------------------

FLAGS = flags.FLAGS


def _write_trial_meta_data(meta_data_path, meta_data):
  d = meta_data.copy()
  d['timestamp'] = time.time()
  with gfile.GFile(meta_data_path, 'w') as f:
    f.write(json.dumps(d, indent=2))


@functools.partial(jax.pmap, axis_name='hosts')
def _sum_seeds_pmapped(seed):
  return lax.psum(seed, 'hosts')


def _create_synchronized_rng_seed():
  rng_seed = np.int64(struct.unpack('q', os.urandom(8))[0])
  rng_seed = _sum_seeds_pmapped(jax_utils.replicate(rng_seed))
  rng_seed = np.sum(rng_seed)
  return rng_seed


def _run(
    # ------------------------------------------------------------------------
    # dogariu: I TOUCHED THIS CODE
    experiment_cfg: ExperimentConfig,
    optimizer_cfg: OptimizerConfig,
    # ------------------------------------------------------------------------
    *,
    trainer_cls,
    dataset_name,
    data_selector_name,
    eval_batch_size,
    eval_use_ema,
    eval_num_batches,
    test_num_batches,
    eval_train_num_batches,
    eval_frequency,
    checkpoint_steps,
    num_tf_data_prefetches,
    num_device_prefetches,
    num_tf_data_map_parallel_calls,
    early_stopping_target_name,
    early_stopping_target_value,
    early_stopping_mode,
    early_stopping_min_steps,
    eval_steps,
    hparam_file,
    allowed_unrecognized_hparams,
    hparam_overrides,
    initializer_name,
    model_name,
    loss_name,
    metrics_name,
    num_train_steps,
    experiment_dir,
    worker_id,
    training_metrics_config,
    callback_configs,
    external_checkpoint_path,
):
  """Function that runs a Jax experiment. See flag definitions for args."""
  model_cls = models.get_model(model_name)
  initializer = initializers.get_initializer(initializer_name)
  dataset_builder = datasets.get_dataset(dataset_name)
  data_selector = datasets.get_data_selector(data_selector_name)
  dataset_meta_data = datasets.get_dataset_meta_data(dataset_name)
  input_pipeline_hps = config_dict.ConfigDict(dict(
      num_tf_data_prefetches=num_tf_data_prefetches,
      num_device_prefetches=num_device_prefetches,
      num_tf_data_map_parallel_calls=num_tf_data_map_parallel_calls,
  ))

  merged_hps = hyperparameters.build_hparams(
      model_name=model_name,
      initializer_name=initializer_name,
      dataset_name=dataset_name,
      hparam_file=hparam_file,
      hparam_overrides=hparam_overrides,
      input_pipeline_hps=input_pipeline_hps,
      allowed_unrecognized_hparams=allowed_unrecognized_hparams)
  
# ------------------------------------------------------------------------
# dogariu: I TOUCHED THIS CODE

#   # Note that one should never tune an RNG seed!!! The seed is only included in
#   # the hparams for convenience of running hparam trials with multiple seeds per
#   # point.
#   rng_seed = merged_hps.rng_seed
#   if merged_hps.rng_seed < 0:
#     rng_seed = _create_synchronized_rng_seed()

  rng_seed = experiment_cfg.seed
  logging.info(f'{bcolors.OKGREEN}{bcolors.BOLD}seed={rng_seed}{bcolors.ENDC}')
  # count GPUs and set up framework-specific things
  logging.info(f'Using {bcolors.WARNING}{bcolors.BOLD}{jax.lib.xla_bridge.get_backend().platform}{bcolors.ENDC} for jax')
  n_gpus = jax.local_device_count()
  logging.info(f' {bcolors.WARNING}{bcolors.BOLD}{n_gpus} devices{bcolors.ENDC}')
# ------------------------------------------------------------------------

  xm_experiment = None
  xm_work_unit = None
  if jax.process_index() == 0:
    logging.info('Running with seed %d', rng_seed)
  rng = jax.random.PRNGKey(rng_seed)

  # Build the loss_fn, metrics_bundle, and flax_module.
  model = model_cls(merged_hps, dataset_meta_data, loss_name, metrics_name)
  trial_dir = os.path.join(experiment_dir, str(worker_id))
  meta_data_path = os.path.join(trial_dir, 'meta_data.json')
  meta_data = {'worker_id': worker_id, 'status': 'incomplete'}
  if jax.process_index() == 0:
    logging.info('rng: %s', rng)
    makedirs(trial_dir)
    # Set up the metric loggers for host 0.
    metrics_logger, init_logger = utils.set_up_loggers(trial_dir, xm_work_unit)

    # --------------------------------------------------------------------------
    # dogariu: I TOUCHED THIS CODE
    csv_path = os.path.join(trial_dir, 'measurements.csv')
    pytree_path = os.path.join(trial_dir, 'training_metrics')
    metrics_logger = CustomMetricLogger(csv_path=csv_path,
        pytree_path=pytree_path,
        xm_work_unit=xm_work_unit,
        events_dir=trial_dir, experiment_cfg=experiment_cfg)
    with open(f"{experiment_dir}/logfile.INFO", "r") as f: logs_so_far = f.read()
    lines = '=' * 74
    logging.info(f'{bcolors.BOLD}{lines}{bcolors.ENDC}')
    logging.info(f'{bcolors.BOLD}EXPERIMENT CONFIG{bcolors.ENDC}')
    logging.info(f'{bcolors.BOLD}{lines}{bcolors.ENDC}')
    logging.info(pretty_dict(asdict(experiment_cfg)))
    logging.info('\n')
    logging.info(f'{bcolors.BOLD}{lines}{bcolors.ENDC}')
    logging.info(f'{bcolors.BOLD}OPTIMIZER CONFIG{bcolors.ENDC}')
    logging.info(f'{bcolors.BOLD}{lines}{bcolors.ENDC}')
    logging.info(pretty_dict(asdict(optimizer_cfg)))
    logging.info('\n\n')
    logging.info(logs_so_far)
    logging.info(f'{bcolors.OKGREEN}{bcolors.BOLD}experiment_dir={experiment_dir}{bcolors.ENDC}')
    # --------------------------------------------------------------------------

    hparams_fname = os.path.join(trial_dir, 'hparams.json')
    logging.info('saving hparams to %s', hparams_fname)
    with gfile.GFile(hparams_fname, 'w') as f:
      f.write(merged_hps.to_json())
    _write_trial_meta_data(meta_data_path, meta_data)
  else:
    metrics_logger = None
    init_logger = None
  try:
    epoch_reports = list(
        trainer_cls(
            trial_dir,
            model,
            dataset_builder,
            initializer,
            num_train_steps,
            merged_hps,
            rng,
            eval_batch_size,
            eval_use_ema,
            eval_num_batches,
            test_num_batches,
            eval_train_num_batches,
            eval_frequency,
            checkpoint_steps,
            early_stopping_target_name,
            early_stopping_target_value,
            early_stopping_mode,
            early_stopping_min_steps,
            eval_steps,
            metrics_logger,
            init_logger,
            training_metrics_config=training_metrics_config,
            callback_configs=callback_configs,
            external_checkpoint_path=external_checkpoint_path,
            dataset_meta_data=dataset_meta_data,
            loss_name=loss_name,
            metrics_name=metrics_name,
            data_selector=data_selector,
    # ------------------------------------------------------------------------
    # dogariu: I TOUCHED THIS CODE
        ).train(experiment_cfg, optimizer_cfg)
    # ------------------------------------------------------------------------
    )
    logging.info(epoch_reports)
    meta_data['status'] = 'done'
  except utils.TrainingDivergedError as err:
    meta_data['status'] = 'diverged'
    raise err
  finally:
    if jax.process_index() == 0:
      _write_trial_meta_data(meta_data_path, meta_data)

# ------------------------------------------------------------------------
# dogariu: I TOUCHED THIS CODE
def run(experiment_cfg: ExperimentConfig, 
        optimizer_cfg: OptimizerConfig):
    assert experiment_cfg.framework == 'jax', f'framework must be jax for init2winit experiments and not whatever {experiment_cfg.framework}'
# ------------------------------------------------------------------------

    # Don't let TF see the GPU, because all we use it for is tf.data loading.
    tf.config.experimental.set_visible_devices([], 'GPU')

    # TODO(gdahl) Figure out a better way to handle passing more complicated
    # flags to the binary.
    training_metrics_config = None
    if FLAGS.training_metrics_config:
        training_metrics_config = json.loads(FLAGS.training_metrics_config)
    if FLAGS.callback_configs:
        callback_configs = json.loads(FLAGS.callback_configs)
    else:
        callback_configs = []

    # ------------------------------------------------------------------------
    # dogariu: I TOUCHED THIS CODE
    experiment_dir = logger_utils.get_log_dir('./experiments',
                                              experiment_cfg.workload_name,
                                              experiment_cfg.framework,
                                              experiment_cfg.experiment_name,
                                              False,
                                              True)
    flags.FLAGS.alsologtostderr = True  # so that we see what gets logged in the terminal
    logging.get_absl_handler().use_absl_log_file('logfile', experiment_dir) 
    logging.info(f'Creating directory at {experiment_dir} for experiments to be saved to.')
    logger_utils.makedir(experiment_dir)

    checkpoint_steps = (list(range(0, experiment_cfg.num_iters * experiment_cfg.num_episodes, (experiment_cfg.num_iters * experiment_cfg.num_episodes) // experiment_cfg.checkpoint_every)) + [experiment_cfg.num_iters * experiment_cfg.num_episodes-1,]) if experiment_cfg.checkpoint_every > 0 else []
    eval_steps = (list(range(0, experiment_cfg.num_iters * experiment_cfg.num_episodes, (experiment_cfg.num_iters * experiment_cfg.num_episodes) // experiment_cfg.eval_every)) + [experiment_cfg.num_iters * experiment_cfg.num_episodes-1,]) if experiment_cfg.eval_every > 0 else []
    # ------------------------------------------------------------------------
    
    if jax.process_index() == 0:
        makedirs(experiment_dir)
    log_dir = os.path.join(experiment_dir, 'r=3/')
    makedirs(log_dir)
    log_path = os.path.join(
        log_dir, 'worker{}_{}.log'.format(FLAGS.worker_id, jax.process_index()))
    with gfile.GFile(log_path, 'a') as logfile:
        utils.add_log_file(logfile)
    if jax.process_index() == 0:
        logging.info('argv:\n%s', ' '.join(sys.argv))
        logging.info('device_count: %d', jax.device_count())
        logging.info('num_hosts : %d', jax.process_count())
        logging.info('host_id : %d', jax.process_index())
        logging.info('checkpoint_steps: %r', checkpoint_steps)
        logging.info('eval_steps: %r', eval_steps)

    trainer_cls = trainers.get_trainer_cls(FLAGS.trainer)

    # ------------------------------------------------------------------------
    # dogariu: I TOUCHED THIS CODE
    from meta_opt.init2winit.trainer import EpisodicTrainer
    trainer_cls = EpisodicTrainer
    assert experiment_cfg.workload_name in MODELS
    model_name = MODELS[experiment_cfg.workload_name]
    loss_name = LOSSES[experiment_cfg.workload_name]
    metrics_name = METRICS[experiment_cfg.workload_name]
    logging.info(f'{bcolors.OKGREEN}{bcolors.BOLD}model={model_name}, loss={loss_name}, metrics={metrics_name}!{bcolors.ENDC}')

    _run(
        experiment_cfg=experiment_cfg,
        optimizer_cfg=optimizer_cfg,

        trainer_cls=trainer_cls,
        dataset_name=experiment_cfg.workload_name,
        data_selector_name='noop',
        eval_batch_size=None,
        eval_use_ema=None,
        eval_num_batches=None,
        test_num_batches=None,
        eval_train_num_batches=None,
        eval_frequency=experiment_cfg.eval_every if experiment_cfg.eval_every > 0 else int(1e9),
        checkpoint_steps=checkpoint_steps,
        num_tf_data_prefetches=-1,
        num_device_prefetches=0,
        num_tf_data_map_parallel_calls=-1,
        early_stopping_target_name=None,
        early_stopping_target_value=None,
        early_stopping_mode=None,
        early_stopping_min_steps=0,
        eval_steps=eval_steps,
        hparam_file=None,
        allowed_unrecognized_hparams=[],
        hparam_overrides='',
        initializer_name='noop',
        model_name=MODELS[experiment_cfg.workload_name],
        loss_name=LOSSES[experiment_cfg.workload_name],
        metrics_name=METRICS[experiment_cfg.workload_name],
        experiment_dir=experiment_dir,
        worker_id=1,
        training_metrics_config=training_metrics_config,
        callback_configs=callback_configs,
        external_checkpoint_path=None,
        num_train_steps=experiment_cfg.num_iters,
    )
    # ------------------------------------------------------------------------


# ------------------------------------------------------------------------
# dogariu: I TOUCHED THIS CODE
def main(_):
    logging.set_verbosity(logging.INFO)
    config_path = flags.FLAGS.config_path

    # Remove the trailing '.py' and convert the filepath to a Python module.
    config_module_path = workloads.convert_filepath_to_module(config_path)
    config_module = importlib.import_module(config_module_path, package='.')

    experiment_cfg, optimizer_cfg = config_module.get_configs()
    run(experiment_cfg, optimizer_cfg)
    return

if __name__ == '__main__':
    app.run(main)
# ------------------------------------------------------------------------
