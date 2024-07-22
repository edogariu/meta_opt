import os
from copy import deepcopy
from absl import logging, flags, app
import importlib
import time
import traceback
from collections import namedtuple
from dataclasses import asdict
from typing import Tuple, Iterable

import numpy as np
import jax
from flax import jax_utils

from algorithmic_efficiency import checkpoint_utils, logger_utils, random_utils
from algorithmic_efficiency.profiler import Profiler
from algorithmic_efficiency.workloads import workloads

import meta_opt.algoperf.jax_nn as jax_nn
from meta_opt.experiment import ExperimentConfig
from meta_opt.optimizers import OptimizerConfig
from meta_opt.utils import bcolors, pretty_dict, shard, make_mesh


DEFAULT_BATCH_SIZES = {
    'mnist': 512,
    'cifar': 128,
    'ogbg': 128,
    'wmt': 16,
}

DEFAULT_NUM_ITERS = {
    'mnist': 20000,
    'cifar': 100000,
    'ogbg': 20000,
    'wmt': 200000,
}

# required flags to run
flags.DEFINE_string('config_path', None, required=True,
    help='The relative path of the Python file containing the experiment and optimizer configs.'
         'NOTE: the config dir must have an __init__.py file!')
flags.DEFINE_string('experiment_dir', os.path.abspath('./experiments'),
                    help='The root directory to store all experiments. '
                    'It is required and the directory should have '
                    'an absolute path rather than a relative path.')
flags.DEFINE_string('data_dir', os.path.abspath('./datasets'),
                    help='The root directory to store all datasets. '
                    'It is required and the directory should have '
                    'an absolute path rather than a relative path.')

# the below flags are redundant since the info is provided in the configs; they are just for algoperf compatability
flags.DEFINE_string('experiment_name', None, help='Name of the experiment.'); flags.FLAGS.experiment_name = ''
flags.DEFINE_string('workload', None, help=f'The name of the workload to run.'); flags.FLAGS.workload = ''
flags.DEFINE_string('framework', None, help='Whether to use Jax or Pytorch.'); flags.FLAGS.framework = ''
flags.DEFINE_string('submission_path', None, help='The relative path of the Python file containing the experiment and optimizer configs. NOTE: the config dir must have an __init__.py file!'); flags.FLAGS.submission_path = ''


def handle_defaults(experiment_cfg: ExperimentConfig):
    name = experiment_cfg.workload_name

    if experiment_cfg.batch_size is None: 
        batch_size = DEFAULT_BATCH_SIZES[name]
        logging.info(f'{bcolors.OKCYAN}{bcolors.BOLD}no `batch_size` provided. using default of {batch_size} for the workload {name}!{bcolors.ENDC}')
        experiment_cfg = experiment_cfg.replace(batch_size=batch_size)
    if experiment_cfg.num_iters is None: 
        num_iters = DEFAULT_NUM_ITERS[name]
        logging.info(f'{bcolors.OKCYAN}{bcolors.BOLD}no `num_iters` provided. using default of {num_iters} for the workload {name}!{bcolors.ENDC}')
        experiment_cfg = experiment_cfg.replace(num_iters=num_iters)

    num_iters = experiment_cfg.num_iters
    if experiment_cfg.eval_every is None: 
        logging.info(f'{bcolors.OKCYAN}{bcolors.BOLD}{bcolors.ENDC}')
    
    return experiment_cfg

def make_workload_and_dataset(rng: jax.random.PRNGKey, 
                              workload_name: str, 
                              batch_size: int, 
                              full_batch: bool, 
                              data_dir: str = './datasets') -> Tuple[workloads.spec.Workload, Iterable]:
    workload_metadata = deepcopy(workloads.WORKLOADS[workload_name])
    workload_metadata['workload_path'] = os.path.join(workloads.BASE_WORKLOADS_DIR,
                                                      workload_metadata['workload_path'] + '_jax',
                                                      'workload.py')
    workload = workloads.import_workload(
        workload_path=workload_metadata['workload_path'],
        workload_class_name=workload_metadata['workload_class_name'],
        workload_init_kwargs={})
    inq = workload._build_input_queue(rng, 'train', data_dir=data_dir, global_batch_size=batch_size)

    @jax.jit
    def _reshape_fn(v):
        d, n, *shape = v.shape
        return v.reshape(d * n, *shape)
    def _reshape_and_shard_generator():
        while True: 
            expanded_batch = next(inq)
            batch = jax.tree_map(_reshape_fn, expanded_batch)
            batch = shard(batch, ('batch',))  # shard along the 'batch' axis of the mesh
            yield batch
    input_queue = _reshape_and_shard_generator()

    
    if full_batch:
        batch = next(input_queue)
        def _same_batch_generator():
            while True: yield batch
        input_queue = _same_batch_generator()
    return workload, input_queue

def evaluate(rng: jax.random.PRNGKey,
             workload: workloads.spec.Workload,
             tstate: jax_nn.JaxTrainState,
             data_dir: str,
             step: int = 0):
    _, model_params, model_state = tstate.get_algoperf_stuff()
    return workload.eval_model(workload.eval_batch_size,
                                jax_utils.replicate(model_params),
                                model_state,
                                rng,
                                data_dir,
                                None,  # hopefully we dont need imagenetv2 dataset dir...
                                step)

def run(experiment_cfg: ExperimentConfig, 
        optimizer_cfg: OptimizerConfig):
    
    assert experiment_cfg.experimental_setup == 'algoperf'
    
    if experiment_cfg.print_with_colors: bcolors.enable()
    else: bcolors.disable()

    # set up profiler
    profiler = Profiler()

    # set up (framework-specific) workload meta-information
    workload_name = experiment_cfg.workload_name.lower()
    
    # for algoperf compatability
    flags.FLAGS.workload = workload_name
    flags.FLAGS.framework = experiment_cfg.framework
    flags.FLAGS.submission_path = None

    # make experiment directory
    experiment_dir = logger_utils.get_log_dir(flags.FLAGS.experiment_dir,
                                              workload_name,
                                              experiment_cfg.framework,
                                              experiment_cfg.experiment_name,
                                              False,
                                              True)
    flags.FLAGS.alsologtostderr = True  # so that we see what gets logged in the terminal
    logging.get_absl_handler().use_absl_log_file('logfile', experiment_dir) 
    logging.info(f'Creating directory at {experiment_dir} for experiments to be saved to.')
    logger_utils.makedir(experiment_dir)

    # set default params if unprovided
    experiment_cfg = handle_defaults(experiment_cfg)
    mesh = make_mesh(experiment_cfg.num_batch_devices, experiment_cfg.num_opt_devices)

    # set up rng's
    seed = experiment_cfg.seed
    logging.info(f'{bcolors.OKGREEN}{bcolors.BOLD}seed={seed}{bcolors.ENDC}')
    rng = random_utils.PRNGKey(seed)
    data_rng, tstate_rng, rng = random_utils.split(rng, 3)  
    data_dir = os.path.abspath(experiment_cfg.data_dir)
    workload, input_queue = make_workload_and_dataset(
        rng=data_rng, workload_name=workload_name, batch_size=experiment_cfg.batch_size,
        full_batch=experiment_cfg.full_batch, data_dir=data_dir
    )
    
    # count GPUs and set up framework-specific things
    if experiment_cfg.framework == 'jax':
        logging.info(f'Using {bcolors.WARNING}{bcolors.BOLD}{jax.lib.xla_bridge.get_backend().platform}{bcolors.ENDC} for jax')
    if experiment_cfg.framework == 'pytorch':
        raise NotImplementedError('will do a pytorch release of this code soon!')
    n_batch_devices = mesh.shape['batch']
    if experiment_cfg.batch_size % n_batch_devices != 0:
        raise ValueError(
            f'The global batch size ({experiment_cfg.batch_size}) has to be divisible by '
            f'the number of batch devices ({n_batch_devices}).')
    if workload.eval_batch_size % n_batch_devices != 0:
        raise ValueError(
            f'The global eval batch size ({workload.eval_batch_size}) has to be '
            f'divisible by the number of batch devices ({n_batch_devices}).')

    # train
    with profiler.profile('Train'):
        if experiment_cfg.framework == 'jax':
            create_train_state = jax_nn.jax_create_train_state
            load_train_state = jax_nn.jax_load_train_state
            train_step = jax_nn.jax_train_step
        elif experiment_cfg.framework == 'pytorch':
            raise NotImplementedError('will do a pytorch release of this code soon!')
        else: 
            raise ValueError(experiment_cfg.framework)

        # log all the configs and everything. this also sets up wandb if we are doing that
        metrics_logger = logger_utils.set_up_loggers(experiment_dir, experiment_cfg, namedtuple('hi', '')())
        workload.attach_metrics_logger(metrics_logger)
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

        # fix a bug in algoperf cifar
        if experiment_cfg.workload_name == 'cifar':
            logging.warning(f'{bcolors.BOLD}{bcolors.WARNING}proactively fixing a bug in the python code for the jax CifarWorkload in algoperf{bcolors.ENDC}')
            from algorithmic_efficiency.workloads.cifar.cifar_jax.workload import CifarWorkload, spec, Optional, param_utils, jax_utils, models, jnp
            def init_model_fn(
                workload,
                rng: spec.RandomState,
                dropout_rate: Optional[float] = None,
                aux_dropout_rate: Optional[float] = None) -> spec.ModelInitState:
                """Dropout is unused."""
                del dropout_rate
                del aux_dropout_rate
                model_cls = getattr(models, 'ResNet18')
                model = model_cls(num_classes=workload._num_classes, dtype=jnp.float32)
                workload._model = model
                input_shape = (1, 32, 32, 3)
                variables = jax.jit(model.init)({'params': rng},
                                                jnp.ones(input_shape, model.dtype))
                model_state, params = variables.pop('batch_stats'), variables.pop('params')  # THIS IS THE ONLY LINE I CHANGED
                workload._param_shapes = param_utils.jax_param_shapes(params)
                workload._param_types = param_utils.jax_param_types(workload._param_shapes)
                model_state = jax_utils.replicate(model_state)
                params = jax_utils.replicate(params)
                return params, model_state
            CifarWorkload.init_model_fn = init_model_fn

        # TrainState (model & optimizer) setup
        s = f'Initializing the model (for {bcolors.OKCYAN}{bcolors.BOLD}{experiment_cfg.workload_name}{bcolors.ENDC}) ' \
            + f'and also optimizer: {bcolors.OKGREEN}{bcolors.BOLD}{optimizer_cfg.optimizer_name}{bcolors.ENDC}'
        logging.info(s)
        with profiler.profile(s):
            tstate = create_train_state(tstate_rng, workload, optimizer_cfg)
        
        model_params = tstate.get_num_params()
        size_of_opt_state_mb = tstate.get_memory_usage()['opt_state_memory'] / (1024 ** 2)
        logging.info(f'Model has {bcolors.BOLD}{model_params}{bcolors.ENDC} parameters and ' + \
                    f'the optimizer state takes {bcolors.BOLD}{size_of_opt_state_mb:.2f}MB{bcolors.ENDC}')
        
        # Metrics and checkpoint setup, along with some bookkeeping.
        global_step = 1
        loss_results = []
        eval_results = []
        preemption_count = 0
        (optimizer_state, model_params, model_state) = tstate.get_algoperf_stuff()
        # If the checkpoint exists, load from the checkpoint.
        logging.info('Initializing checkpoint and metrics.')
        try:
            checkpoint = checkpoint_utils.maybe_restore_checkpoint(
                experiment_cfg.framework,
                optimizer_state,
                model_params,
                model_state,
                {},
                eval_results,
                global_step,
                preemption_count,
                checkpoint_dir=experiment_dir)
            (optimizer_state, model_params, model_state, _, eval_results, global_step, preemption_count) = checkpoint
        except Exception as e:
            logging.error(f'{bcolors.FAIL}{bcolors.BOLD}unable to load checkpoint, see error below:{bcolors.ENDC}\n\t', e)
            pass
        meta_file_name = os.path.join(experiment_dir, f'meta_data_{preemption_count}.json')
        logging.info(f'Saving meta data to {meta_file_name}.')
        meta_data = logger_utils.get_meta_data(workload)
        logger_utils.write_json(meta_file_name, meta_data)
        config_file_name = os.path.join(experiment_dir, f'config_{preemption_count}.json')
        logging.info(f'Saving config to {config_file_name}.')
        e_d = asdict(experiment_cfg)
        o_d = asdict(optimizer_cfg)
        logger_utils.write_json(config_file_name, {'experiment_cfg': e_d, 'optimizer_cfg': o_d})
        if experiment_cfg.num_episodes > 1:
            logging.info(f'{bcolors.WARNING}{bcolors.BOLD}wont load checkpoints for episodic learning, its weird{bcolors.ENDC}')
        else:
            if global_step == 1:
                logging.info(f'{bcolors.WARNING}{bcolors.BOLD}I found no checkpoint, so we proceed without loading :){bcolors.ENDC}')
            else:
                tstate = load_train_state(checkpoint, workload, optimizer_cfg)

        # the actual train loop
        logging.info(f'{bcolors.FAIL}{bcolors.BOLD}Starting training loop.{bcolors.ENDC}')
        num_iters, num_episodes, eval_every, checkpoint_every, log_every = experiment_cfg.num_iters, experiment_cfg.num_episodes, experiment_cfg.eval_every, experiment_cfg.checkpoint_every, experiment_cfg.log_every
        global_start_time = time.time()

        for episode_i in range(1, num_episodes + 1):
            # reset for the episode
            rng, reset_rng = random_utils.split(rng)
            tstate = tstate.reset(reset_rng, workload, optimizer_cfg.reset_opt_state)

            if num_episodes > 1: logging.info(f'{bcolors.FAIL}{bcolors.BOLD}Starting training episode {episode_i}.{bcolors.ENDC}')
            for local_step in range(1, num_iters + 1):
                step_start_time = time.time()

                # cycle rngs
                step_rng = random_utils.fold_in(rng, global_step)
                rng, update_rng, eval_rng = random_utils.split(step_rng, 3)
                boundary_step = (local_step == 1 or local_step == num_iters - 1)

                # perform parameter update
                with profiler.profile('Update parameters'):
                    batch = next(input_queue)
                    tstate, latest_train_result = train_step(update_rng, workload, tstate, batch)
                    loss_results.append((global_step, latest_train_result))
                    # print(latest_train_result['loss'], tstate.opt_state[0].recent_gpc_cost, tstate.opt_state[0].recent_gpc_grads.sum(), tstate.opt_state[0].disturbance_history.mean(axis=1))

                # eval if we want (including on the first and last steps)
                if (eval_every > 0) and (local_step % eval_every == 0 or boundary_step):
                    if experiment_cfg.full_batch:
                        logging.warning(f'{bcolors.WARNING}{bcolors.BOLD}skipped evaluation at step {local_step} because `full_batch=True`{bcolors.ENDC}')
                    else:
                        try:
                            with profiler.profile('Evaluating'):
                                latest_eval_result = evaluate(eval_rng, workload, tstate, data_dir, local_step)

                            time_since_start = time.time() - global_start_time
                            logging.info(f'Time: {time_since_start:.2f}s, '
                                        f'\tStep: {local_step},\tEpisode:{episode_i},\t{bcolors.FAIL}{bcolors.BOLD}{latest_eval_result}{bcolors.ENDC}')
                            eval_results.append((global_step, latest_eval_result))
                            metrics_logger.append_scalar_metrics(latest_eval_result, global_step=global_step, preemption_count=preemption_count, is_eval=True)
                        except:
                            logging.error(f'{bcolors.FAIL}{bcolors.BOLD}failed on eval at step {local_step} of episode {episode_i} with error {traceback.format_exc()}{bcolors.ENDC}')
                            pass

                # checkpoint if we want (including on the last step)
                if (checkpoint_every > 0) and (local_step % checkpoint_every == 0 or (boundary_step and local_step > 1)):
                    optimizer_state, model_params, model_state = tstate.get_algoperf_stuff()
                    checkpoint_utils.save_checkpoint(
                        framework=experiment_cfg.framework,
                        optimizer_state=optimizer_state,
                        model_params=model_params,
                        model_state=model_state,
                        train_state={},
                        eval_results=eval_results,
                        global_step=global_step,
                        preemption_count=preemption_count,
                        checkpoint_dir=experiment_dir,
                        save_intermediate_checkpoints=True)
                    logging.info(f'{bcolors.OKGREEN}{bcolors.BOLD}saved checkpoint at step {local_step} of episode {episode_i} to directory {experiment_dir} under name {experiment_cfg.experiment_name}{bcolors.ENDC}')
                
                # collect metrics from this step and write to logs
                if (log_every > 0) and experiment_dir and (local_step % log_every == 0 or boundary_step):
                    with profiler.profile('Collecting step metrics'):
                        step_stats = {k: v.item() if hasattr(v, 'item') else v for k, v in latest_train_result.items()}
                        step_stats['step_time'] = time.time() - step_start_time
                        step_stats.update(logger_utils._get_utilization())
                        step_stats.update(tstate.get_logging_metrics())
                        metrics_logger.append_scalar_metrics(step_stats, global_step=global_step, preemption_count=None, is_eval=False)
                
                local_step += 1
                global_step += 1

        metrics_logger.finish()

    # epilogue
    logging.info(profiler.summary())
    try:
        loss_results = np.array([(l[0], l[1]['loss']) for l in loss_results])
        if len(eval_results) > 0:
            eval_results = np.array([(e[0], e[1]['validation/accuracy']) for e in eval_results])
    except:
        pass
    return loss_results, eval_results, tstate

# def run_fullbatch(optimizer_cfg: OptimizerConfig,
#                   rng: jax.random.PRNGKey,
#                   num_iters: int,
#                   num_episodes: int,

#                   workload, input_queue):
#     # make rng, data, and model
#     init_rng, rng = jax.random.split(rng)
#     tstate = jax_nn.jax_create_train_state(init_rng, workload, optimizer_cfg)
#     batch = next(input_queue)
#     return jax_nn._run_fullbatch(tstate, rng, num_episodes, num_iters, workload, batch, optimizer_cfg.reset_opt_state)


def main(_):
    logging.set_verbosity(logging.INFO)
    config_path = flags.FLAGS.config_path

    # Remove the trailing '.py' and convert the filepath to a Python module.
    config_module_path = workloads.convert_filepath_to_module(config_path)
    config_module = importlib.import_module(config_module_path, package='.')

    # run the configs
    cfgs = config_module.get_config()
    for i, (experiment_cfg, optimizer_cfg) in enumerate(cfgs):
        e = experiment_cfg.replace(experiment_name=f'{experiment_cfg.experiment_name}_{i}')
        run(e, optimizer_cfg)
    return

if __name__ == '__main__':
    app.run(main)
