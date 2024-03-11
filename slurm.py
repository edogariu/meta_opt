"""
FOR SLURM SCRIPTS
"""
# handle the system stuff, colab stuff, etc
import os
DIR = os.path.abspath(".")

# make sure we have the necessary folders
for subdir in ['data', 'figs', 'datasets']: 
    temp = os.path.join(DIR, subdir)
    if not os.path.isdir(temp): os.mkdir(temp)

from meta_opt.train_loops import train_standard_opt, train_hgd, train_meta_opt
from meta_opt.utils.experiment_utils import make, save_checkpoint, process_results, bcolors, plot, get_final_cparams
import meta_opt.configs as configs

import re
import matplotlib.pyplot as plt
import numpy as np
import dill as pkl
import optax


# ==================================================
from scripts.main_cifar_pretrained import CFG, run
# ==================================================


SEEDS = range(500)
if __name__ == '__main__':
    idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    CFG['experiment_name'] = f'{NAME}_{idx}'
    s = SEEDS[idx]
    
    run([s,], CFG)    
