import os
from copy import deepcopy
from absl import logging, flags, app
import importlib
import jax

from algorithmic_efficiency import logger_utils
from algorithmic_efficiency.profiler import Profiler, PassThroughProfiler
from algorithmic_efficiency.workloads import workloads

from configs.workload_defaults import handle_defaults
from configs.experiment import ExperimentConfig
from configs.optimizers import OptimizerConfig
from meta_opt.trainer import train
from meta_opt.utils import bcolors

# required flags to run
flags.DEFINE_integer('seed', None, required=True, help='Random seed for experiments.')
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
flags.FLAGS.alsologtostderr = True  # so that we see what gets logged in the terminal

# the below flags are redundant since the info is provided in the configs; they are just for algoperf compatability
flags.DEFINE_string('experiment_name', None, help='Name of the experiment.'); flags.FLAGS.experiment_name = ''
flags.DEFINE_string('workload', None, help=f'The name of the workload to run.'); flags.FLAGS.workload = ''
flags.DEFINE_string('framework', None, help='Whether to use Jax or Pytorch.'); flags.FLAGS.framework = ''
flags.DEFINE_string('submission_path', None, help='The relative path of the Python file containing the experiment and optimizer configs. NOTE: the config dir must have an __init__.py file!'); flags.FLAGS.submission_path = ''


def run(seed: int,
        experiment_cfg: ExperimentConfig, 
        optimizer_cfg: OptimizerConfig):
        
    # set up profiler
    profiler = Profiler() if experiment_cfg.profile else PassThroughProfiler()

    # set up (framework-specific) workload meta-information
    workload_name = experiment_cfg.workload_name.lower()
    assert workload_name in workloads.WORKLOADS, f'workload {workload_name} simply doesn\'t exist :)'
    if workloads.get_base_workload_name(workload_name) in ['librispeech_conformer', 'librispeech_deepspeech']: 
        raise NotImplementedError('need to write some special code for this!')
    workload_metadata = deepcopy(workloads.WORKLOADS[workload_name])
    workload_metadata['workload_path'] = os.path.join(workloads.BASE_WORKLOADS_DIR,
                                                      workload_metadata['workload_path'] + f'_{experiment_cfg.framework}',
                                                      'workload.py')
    
    # for algoperf compatability
    flags.FLAGS.workload = workload_name
    flags.FLAGS.framework = experiment_cfg.framework
    flags.FLAGS.submission_path = None

    # make experiment directory
    experiment_dir = logger_utils.get_log_dir(flags.FLAGS.experiment_dir,
                                              workload_name,
                                              experiment_cfg.framework,
                                              experiment_cfg.experiment_name,
                                              experiment_cfg.resume_last_run,
                                              experiment_cfg.overwrite)
    logging.get_absl_handler().use_absl_log_file('logfile', experiment_dir) 
    logging.info(f'Creating directory at {experiment_dir} for experiments to be saved to.')
    logger_utils.makedir(experiment_dir)

    # set default params if unprovided
    experiment_cfg = handle_defaults(experiment_cfg)

    # import the workload
    workload = workloads.import_workload(
        workload_path=workload_metadata['workload_path'],
        workload_class_name=workload_metadata['workload_class_name'],
        workload_init_kwargs={})
    
    # count GPUs and set up framework-specific things
    if experiment_cfg.framework == 'jax':
        logging.info(f'Using {bcolors.WARNING}{bcolors.BOLD}{jax.lib.xla_bridge.get_backend().platform}{bcolors.ENDC} for jax')
    if experiment_cfg.framework == 'pytorch':
        raise NotImplementedError('will do a pytorch release of this code soon!')
    n_gpus = jax.local_device_count()
    logging.info(f' {bcolors.WARNING}{bcolors.BOLD}{n_gpus} devices{bcolors.ENDC}')
    if experiment_cfg.batch_size % n_gpus != 0:
        raise ValueError(
            f'The global batch size ({experiment_cfg.batch_size}) has to be divisible by '
            f'the number of GPUs ({n_gpus}).')
    if workload.eval_batch_size % n_gpus != 0:
        raise ValueError(
            f'The global eval batch size ({workload.eval_batch_size}) has to be '
            f'divisible by the number of GPUs ({n_gpus}).')

    # train
    with profiler.profile('Train'):
        train(seed, workload, profiler, experiment_cfg, optimizer_cfg, experiment_dir)

    # epilogue
    if experiment_cfg.profile:
        logging.info(profiler.summary())
    return


def main(_):
    logging.set_verbosity(logging.INFO)
    seed = flags.FLAGS.seed
    config_path = flags.FLAGS.config_path

    # Remove the trailing '.py' and convert the filepath to a Python module.
    config_module_path = workloads.convert_filepath_to_module(config_path)
    config_module = importlib.import_module(config_module_path, package='.')

    experiment_cfg, optimizer_cfg = config_module.get_configs()
    run(seed, experiment_cfg, optimizer_cfg)
    return


if __name__ == '__main__':
    app.run(main)
