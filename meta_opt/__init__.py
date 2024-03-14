# handle the system stuff, colab stuff, etc
import os
try:
    from google import colab  # for use in google colab!!
    os.system('git clone https://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@github.com/edogariu/meta-opt')
    os.system('pip install -q ./meta-opt')
    os.system('pip install -q dill')
    # !pip install -q jax[cuda12_pip]==0.4.20 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html  # for disabling prealloc, see https://github.com/google/jax/discussions/19014
    os.system('pip install -q tensorflow-text ml_collections clu sentencepiece')  # for WMT
    from google.colab import drive
    drive.mount('/content/drive')
    DIR = os.path.abspath("./drive/My Drive/meta-opt")
except: 
    DIR = os.path.abspath("../")
assert os.path.isdir(DIR)

# make sure we have the necessary folders
for subdir in ['data', 'figs', 'datasets']: 
    temp = os.path.join(DIR, subdir)
    if not os.path.isdir(temp): os.mkdir(temp)
    