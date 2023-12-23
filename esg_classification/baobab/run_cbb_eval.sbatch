#!/bin/sh

#SBATCH --job-name cb_eval
#SBATCH --partition=shared-gpu
#SBATCH --time=10:00:00
#SBATCH --mem=0
#SBATCH --gres=gpu:1
#SBATCH --output=./Results/logs_CBeval.log

. ~/thesis/baobab_env/bin/activate
python ~/thesis/src/camembert_base_eval.py
