#!/bin/bash
#SBATCH --job-name=cifar_pretrained         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00          # total run time limit (HH:MM:SS)
#SBATCH --array=0-5              # job array with index values 0, 1, 2
#SBATCH --mail-type=all
#SBATCH --mail-user=edogariu@princeton.edu

module purge
module load anaconda3/2023.9
conda activate meta-opt

python scripts/slurm.py