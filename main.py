
# ==================================================
from scripts.main_wmt_cf import CFG, run
# ==================================================


SEEDS = [0,]
if __name__ == '__main__':
    try: 
        idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
        name = CFG['experiment_name']
        CFG['experiment_name'] = f'{name}_{idx}'
        SEEDS = [idx,]  # set seed to the index
    except:
        pass    
    run(SEEDS, CFG)    
