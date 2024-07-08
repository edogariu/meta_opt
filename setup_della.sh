#!/bin/bash

gpu=$(lspci | grep -i '.* vga .* nvidia .*')
shopt -s nocasematch

# ensure all the needed folders exist
echo "making directories..."
mkdir ./datasets/
mkdir ./experiments/

# load venv
echo "loading venv..."
module load anaconda3/2024.2
conda activate meta-opt

# install required things
echo "installing dependencies..."

# install workload deps first because otherwise it is impossible sometimes
pip3 install tensorflow==2.15.*
pip3 install sentencepiece==0.1.99 sacrebleu==1.3.1 pydub==0.25.1
pip3 install tensorflow-text==2.12.1  # high probability of failure :)

# install algorithmic_efficiency (ie algoperf, see https://arxiv.org/abs/2306.07179)
git clone https://github.com/mlcommons/algorithmic-efficiency/
cd algorithmic-efficiency
sed -i -e 's/tensorflow==2.12.0/tensorflow/g' ./setup.cfg  # because tensorflow 2.12.0 doesnt exist lol
if [[ $gpu == *' nvidia '* ]]; then
    printf 'Nvidia GPU is present:  %s\n' "$gpu"
    pip3 install  .[jax_gpu,pytorch_gpu,ogbg,criteo1tb,fastmri,wandb] -f https://storage.googleapis.com/jax-releases/jax_releases.html  # get gpu setup
else
    pip3 install  .[jax_cpu,pytorch_cpu,ogbg,criteo1tb,fastmri,wandb]  # get cpu setup
fi
cd ..
rm -rf ./algorithmic-efficiency

# fix a psutil bug (that only shows up on arm macs I think, but might as well fix it)
pip3 uninstall -y psutil
pip3 install --no-binary :all: psutil

# install the right jax/flax/optax with cuda
pip3 install optax==0.1.5 flax==0.6.10
pip3 install ml_dtypes==0.2.0
pip3 install "jax[cuda12_pip]"==0.4.10 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# install the meta-opt package
pip3 install -r requirements.txt
echo "installing meta-opt..."
pip3 install -e .
