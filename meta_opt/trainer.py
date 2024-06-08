from absl import logging, flags
import time
import os
from dataclasses import asdict
import jax

from algorithmic_efficiency.spec import Workload
from algorithmic_efficiency.profiler import Profiler
from algorithmic_efficiency import checkpoint_utils, logger_utils, random_utils

from meta_opt.experiment import ExperimentConfig
from meta_opt.optimizers import OptimizerConfig
import meta_opt.jax_stuff.jax_nn as jax_nn
from meta_opt.utils import bcolors

def train(seed: int,
          workload: Workload,
          profiler: Profiler,
          experiment_cfg: ExperimentConfig, 
          optimizer_cfg: OptimizerConfig,
          experiment_dir: str
          ):
    
    rng = random_utils.PRNGKey(seed)
    data_rng, tstate_rng, rng = random_utils.split(rng, 3)

    # -------------------------------------------------------------------------------------------------------------------------------
    # these should be the !only lines! that differentiate behavior based on the framework.
    # Elsewhere, the deep learning code runs for all frameworks using the API given in `nn.py`.
    if experiment_cfg.framework == 'jax':
        create_train_state = jax_nn.jax_create_train_state
        load_train_state = jax_nn.jax_load_train_state
        train_step = jax_nn.jax_pmapped_train_step
    elif experiment_cfg.framework == 'pytorch':
        raise NotImplementedError('pytorch')
    else: 
        raise ValueError(experiment_cfg.framework)
    # -------------------------------------------------------------------------------------------------------------------------------

    # Workload setup.
    data_dir = os.path.expanduser(flags.FLAGS.data_dir)
    s = f'Initializing dataset for workload: {bcolors.OKCYAN}{bcolors.BOLD}{experiment_cfg.workload_name}{bcolors.ENDC}.'
    logging.info(s)
    with profiler.profile(s):
        input_queue = workload._build_input_queue(data_rng, 'train', data_dir=data_dir, global_batch_size=experiment_cfg.batch_size)
        if experiment_cfg.full_batch:
            batch = next(input_queue)
            def _same_batch_generator():
                while True: yield batch
            input_queue = _same_batch_generator()

    # TrainState (model & optimizer) setup
    s = f'Initializing the model (for {bcolors.OKCYAN}{bcolors.BOLD}{experiment_cfg.workload_name}{bcolors.ENDC}) ' \
        + f'and also optimizer: {bcolors.OKGREEN}{bcolors.BOLD}{optimizer_cfg.optimizer_name}{bcolors.ENDC}'
    logging.info(s)
    with profiler.profile(s):
        tstate = create_train_state(tstate_rng, workload, optimizer_cfg)
    model_params = tstate.get_num_params()
    size_of_opt_state_mb = tstate.get_opt_state_memory() / (1024 ** 2)
    logging.info(f'Model has {bcolors.BOLD}{model_params}{bcolors.ENDC} parameters and ' + \
                 f'the optimizer state takes {bcolors.BOLD}{size_of_opt_state_mb:.2f}MB{bcolors.ENDC}')
    
    # Loggers and checkpoint setup, along with some bookkeeping.
    global_step = tstate.t + 1
    eval_results = []
    preemption_count = 0
    (optimizer_state, model_params, model_state) = tstate.get_algoperf_stuff()
    # If the checkpoint exists, load from the checkpoint.
    logging.info('Initializing checkpoint and logger.')
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
    metrics_logger = logger_utils.set_up_loggers(experiment_dir, experiment_cfg, None)
    workload.attach_metrics_logger(metrics_logger)
    if global_step == 1:
        logging.warning(f'{bcolors.WARNING}{bcolors.BOLD}I found no checkpoint, so we proceed without loading :){bcolors.ENDC}')
    else:
        tstate = load_train_state(checkpoint, workload, optimizer_cfg)

    # the actual train loop
    logging.info(f'{bcolors.FAIL}{bcolors.BOLD}Starting training loop.{bcolors.ENDC}')
    num_iters, eval_every, reset_every, checkpoint_every, print_every, log_every = experiment_cfg.num_iters, experiment_cfg.eval_every, experiment_cfg.reset_every, experiment_cfg.checkpoint_every, experiment_cfg.print_every, experiment_cfg.log_every
    global_start_time = time.time()

    while global_step < num_iters:
        step_start_time = time.time()

        # cycle rngs
        step_rng = random_utils.fold_in(rng, global_step)
        rng, update_rng, eval_rng = random_utils.split(step_rng, 3)
        boundary_step = (global_step == 1 or global_step == num_iters - 1)

        # perform parameter update
        with profiler.profile('Update parameters'):
            batch = next(input_queue)
            tstate, latest_train_result = train_step(update_rng, workload, tstate, batch)

        # print if we want (including on the first and last steps)
        if print_every and (global_step % print_every == 0 or boundary_step):
            time_since_start = time.time() - global_start_time
            logging.info(f'Time: {time_since_start:.2f}s, '
                         f'\tStep: {global_step}, \ttrain_metrics={bcolors.BOLD}{latest_train_result}{bcolors.ENDC}')

        # eval if we want (including on the first and last steps)
        if eval_every and (global_step % eval_every == 0 or boundary_step):
            if experiment_cfg.full_batch:
                logging.warning(f'{bcolors.WARNING}{bcolors.BOLD}skipped evaluation at step {global_step} because `full_batch=True`{bcolors.ENDC}')
            else:
                with profiler.profile('Evaluating'):
                    _, model_params, model_state = tstate.get_algoperf_stuff()
                    latest_eval_result = workload.eval_model(workload.eval_batch_size,
                                                             model_params,
                                                             model_state,
                                                             eval_rng,
                                                             data_dir,
                                                             None,  # hopefully we dont need imagenetv2 dataset dir...
                                                             global_step)

                time_since_start = time.time() - global_start_time
                logging.info(f'Time: {time_since_start:.2f}s, '
                            f'\tStep: {global_step}, \t{bcolors.FAIL}{bcolors.BOLD}{latest_eval_result}{bcolors.ENDC}')
                eval_results.append((global_step, latest_eval_result))
                metrics_logger.append_scalar_metrics(latest_eval_result, global_step=global_step, preemption_count=preemption_count, is_eval=True)

        # checkpoint if we want (including on the last step)
        if checkpoint_every and (global_step % checkpoint_every == 0 or (boundary_step and global_step > 1)):
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
            logging.info(f'{bcolors.OKGREEN}{bcolors.BOLD}saved checkpoint at step {global_step} to directory {experiment_dir} under name {experiment_cfg.experiment_name}{bcolors.ENDC}')
                
        # reset the episode if we want
        if reset_every and (global_step % reset_every) == 0:
            rng, reset_rng = random_utils.split(rng)
            tstate = tstate.reset(reset_rng, workload, experiment_cfg.reset_opt_state)
        
        # collect metrics from this step and write to logs
        if log_every and experiment_dir and (global_step % log_every == 0 or boundary_step):
            with profiler.profile('Collecting step metrics'):
                step_stats = {k: v.item() if hasattr(v, 'item') else v for k, v in latest_train_result.items()}
                step_stats['step_time'] = time.time() - step_start_time
                step_stats.update(logger_utils._get_utilization())
                step_stats.update(tstate.get_logging_metrics())
                metrics_logger.append_scalar_metrics(step_stats, global_step=global_step, preemption_count=preemption_count, is_eval=False)

        global_step += 1
    metrics_logger.finish()
    return
