#!/bin/sh

#SBATCH --job-name cb-long
#SBATCH --partition=shared-gpu
#SBATCH --time=0-12:00:00
#SBATCH --mem=0
#SBATCH --gres=gpu:1,VramPerGpu:30G
#SBATCH --output=model-cb-long/run1/result_logs.log

. ~/thesis/baobab_env/bin/activate

# Set TOKENIZERS_PARALLELISM
export TOKENIZERS_PARALLELISM=true

python ~/thesis/src/train_cb_model_long.py \
    --model_checkpoint camembert-base \
    --output_dir ./model-cb-long/run1 \
    --batch_size 64 \
    --dataset_frac 1 \
    --num_epochs 35 \
    --rounds 5 \
    --grad_steps 8
    
