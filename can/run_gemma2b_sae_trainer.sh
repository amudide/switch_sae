#!/bin/bash

python3 gemma2b_sae_trainer.py \
    --save_dir /share/u/can/shift_eval/train_saes/trained_saes/gemma2b_sweep0710 \
    --layer 12 \
    # --no_wandb_logging \
    # --dry_run \