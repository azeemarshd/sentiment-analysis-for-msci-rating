#!/bin/sh

#SBATCH --job-name minibatch_model
#SBATCH --partition=shared-gpu
#SBATCH --time=0-10:00:00
#SBATCH --mem=0
#SBATCH --gres=gpu:3
#SBATCH --output=../Results/pl_minibatch_model.log

. ~/thesis/baobab_env/bin/activate

# Set TOKENIZERS_PARALLELISM
export TOKENIZERS_PARALLELISM=true

python ~/thesis/src/pl_model_minibatch.py 
