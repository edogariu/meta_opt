import os
try:
    from google import colab  # for use in google colab!!
    DIR = os.path.abspath("./drive/My Drive/meta-opt")
except: 
    DIR = os.path.abspath(".")
assert os.path.isdir(DIR)

# make sure we have the necessary folders
for subdir in ['data', 'figs', 'datasets']: 
    temp = os.path.join(DIR, subdir)
    if not os.path.isdir(temp): os.mkdir(temp)

MNIST_FULLBATCH = {
    # training options
    'workload': 'MNIST',
    'num_iters': 12000,
    'eval_every': -1,
    'num_eval_iters': -1,
    'batch_size': 512,
    'full_batch': True,
    'reset_every': 400,

    # experiment options
    'experiment_name': 'mnist_fullbatch',
    'load_checkpoint': True,
    'overwrite': False,  # whether to allow us to overwrite existing checkpoints or throw errors
    'directory': DIR,
}