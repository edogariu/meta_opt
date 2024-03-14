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


# ==================================================
from scripts.main_wmt_baselines import CFG, run
# ==================================================


SEEDS = range(500)
if __name__ == '__main__':
    idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    name = CFG['experiment_name']
    CFG['experiment_name'] = f'{name}_{idx}'
    s = SEEDS[idx]
    
    run([s,], CFG)    
