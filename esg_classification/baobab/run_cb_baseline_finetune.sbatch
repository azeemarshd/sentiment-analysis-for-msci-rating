#!/bin/sh

#SBATCH --job-name cbb_1gpu
#SBATCH --partition=shared-gpu
#SBATCH --time=4:00:00
#SBATCH --mem=0
#SBATCH --gres=gpu:1
#SBATCH --output=./Results/logs_cbb_finetuned.log

. ~/thesis/baobab_env/bin/activate
python ~/thesis/src/cbb_baseline_finetune.py 
