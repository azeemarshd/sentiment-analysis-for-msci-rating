#!/bin/sh

#SBATCH --job-name torchmodel1
#SBATCH --partition=shared-gpu
#SBATCH --time=0-08:00:00
#SBATCH --mem=0
#SBATCH --gres=gpu:1
#SBATCH --output=../Results/torch_model1.log

. ~/thesis/baobab_env/bin/activate

# Set TOKENIZERS_PARALLELISM
export TOKENIZERS_PARALLELISM=true

python ~/thesis/src/torch_model.py
