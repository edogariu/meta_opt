from absl import logging

from meta_opt.training.experiment import ExperimentConfig
from meta_opt.utils import bcolors

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
    'wmt': 100000,
}


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

