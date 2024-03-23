#!/bin/sh

#SBATCH --job-name cb_plModel
#SBATCH --partition=shared-gpu
#SBATCH --time=0-00:30:00
#SBATCH --mem=0
#SBATCH --gres=gpu:2
#SBATCH --output=../Results/cb_plModel_0401.log

. ~/thesis/baobab_env/bin/activate

# Set TOKENIZERS_PARALLELISM
export TOKENIZERS_PARALLELISM=true

python ~/thesis/src/pl_model.py 
