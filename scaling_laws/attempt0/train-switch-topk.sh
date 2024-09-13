#!/bin/bash
#SBATCH --job-name=dtk0
#SBATCH --partition=tegmark
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2-00:00:00
#SBATCH --output=/om2/user/ericjm/switch_sae/scaling_laws/attempt0/slurm-%A_%a.out
#SBATCH --array=8,64

python /om2/user/ericjm/switch_sae/scaling_laws/attempt0/train-switch-topk.py --num_experts $SLURM_ARRAY_TASK_ID
