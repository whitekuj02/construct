#!/bin/bash

export PATH=~/anaconda3/bin:~/anaconda3/condabin:$PATH

. ~/anaconda3/etc/profile.d/conda.sh

conda env create --file environment.yaml

conda activate construct

# 2. PyTorch 및 필수 라이브러리 설치 (CUDA 11.8 기준)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
