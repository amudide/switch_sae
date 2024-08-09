python train-switch-1on.py --gpu 4 --ks 8 --num_experts 32 --heavisides True
python train-switch-1on.py --gpu 5 --ks 16 --num_experts 32 --heavisides True
python train-switch-1on.py --gpu 6 --ks 32 --num_experts 32 --heavisides True
python train-switch-1on.py --gpu 7 --ks 48 --num_experts 32 --heavisides True
python train-switch-1on.py --gpu 4 --ks 64 --num_experts 32 --heavisides True
python train-switch-1on.py --gpu 5 --ks 96 --num_experts 32 --heavisides True
python train-switch-1on.py --gpu 6 --ks 128 --num_experts 32 --heavisides True
python train-switch-1on.py --gpu 7 --ks 192 --num_experts 32 --heavisides True

python train-switch-1on.py --gpu 4 --ks 8 16 --num_experts 32
python train-switch-1on.py --gpu 5 --ks 32 48 --num_experts 32
python train-switch-1on.py --gpu 6 --ks 64 96 --num_experts 32
python train-switch-1on.py --gpu 7 --ks 128 192 --num_experts 32

python train-switch-1on.py --gpu 5 --ks 64 --num_experts 32 --lb_alphas 0.001 0.003
python train-switch-1on.py --gpu 6 --ks 64 --num_experts 32 --lb_alphas 0.01 0.03
python train-switch-1on.py --gpu 7 --ks 64 --num_experts 32 --lb_alphas 0.1 0.3
python train-switch-1on.py --gpu 5 --ks 64 --num_experts 32 --lb_alphas 1 3
python train-switch-1on.py --gpu 6 --ks 64 --num_experts 32 --lb_alphas 10 30
python train-switch-1on.py --gpu 7 --ks 64 --num_experts 32 --lb_alphas 100 300

## Pre-blog post

python train-switch.py --gpu 0 --ks 8 16 --num_experts 16
python train-switch.py --gpu 1 --ks 32 48 --num_experts 16
python train-switch.py --gpu 2 --ks 64 96 --num_experts 16
python train-switch.py --gpu 3 --ks 128 192 --num_experts 16

python train-switch.py --gpu 3 --ks 8 16 --num_experts 32
python train-switch.py --gpu 4 --ks 32 48 --num_experts 32
python train-switch.py --gpu 5 --ks 64 96 --num_experts 32
python train-switch.py --gpu 6 --ks 128 192 --num_experts 32

python train-switch.py --gpu 4 --ks 8 16 --num_experts 64
python train-switch.py --gpu 5 --ks 32 48 --num_experts 64
python train-switch.py --gpu 6 --ks 64 96 --num_experts 64
python train-switch.py --gpu 7 --ks 128 192 --num_experts 64

python train-switch.py --gpu 6 --ks 8 16 32 48 --num_experts 128
python train-switch.py --gpu 7 --ks 64 96 128 192 --num_experts 128

python train-switch-flop.py --gpu 3 --ks 8 16 32 --num_experts 4
python train-switch-flop.py --gpu 4 --ks 64 128 192 --num_experts 4

python train-switch-flop.py --gpu 6 --ks 8 16 32 --num_experts 8
python train-switch-flop.py --gpu 7 --ks 64 128 192 --num_experts 8

## Gated 1e-3

python train-gated.py --gpu 4 --l1_penalties 1 1.46
python train-gated.py --gpu 5 --l1_penalties 2.13 3.11
python train-gated.py --gpu 6 --l1_penalties 4.53 6.62
python train-gated.py --gpu 6 --l1_penalties 9.65 14.09
python train-gated.py --gpu 7 --l1_penalties 20.56 30

## FLOP-matched
# (2 experts can be run on A6000)
python train-switch-flop.py --gpu 5 --ks 8 16 --num_experts 2
python train-switch-flop.py --gpu 6 --ks 32 64 --num_experts 2
python train-switch-flop.py --gpu 7 --ks 128 192 --num_experts 2

## Sample Efficiency

python train-switch-flop.py --gpu 3 --ks 64 --num_experts 16
python train-switch-flop.py --gpu 4 --ks 64 --num_experts 32

#### TODO

## FLOP-matched

python train-switch-flop.py --gpu 6 --ks 8 16 32 --num_experts 16
python train-switch-flop.py --gpu 7 --ks 64 128 192 --num_experts 16





### old train.sh files

## ADD WANDB NAME TO ARGPARSE FOR SH FILES!

python train-switch.py --gpu 5 --ks 8 --num_experts 16 --heavisides f t
python train-switch.py --gpu 6 --ks 32 --num_experts 16 --heavisides f t
python train-switch.py --gpu 7 --ks 64 --num_experts 16 --heavisides f t
python train-switch.py --gpu 4 --ks 128 --num_experts 16 --heavisides f t

python train-switch.py --gpu 5 --ks 8 --num_experts 32 --heavisides f t
python train-switch.py --gpu 5 --ks 32 --num_experts 32 --heavisides f t
python train-switch.py --gpu 6 --ks 64 --num_experts 32 --heavisides f t
python train-switch.py --gpu 7 --ks 128 --num_experts 32 --heavisides f t

python train-switch-flop.py --gpu 2 --ks 8 32 --num_experts 8 --heavisides f
python train-switch-flop.py --gpu 3 --ks 64 128 --num_experts 8 --heavisides f

## Test LB

python train-switch.py --gpu 5 --ks 64 --num_experts 16 --lb_alphas 0 0.001
python train-switch.py --gpu 6 --ks 64 --num_experts 16 --lb_alphas 0.003 0.01
python train-switch.py --gpu 7 --ks 64 --num_experts 16 --lb_alphas 0.03 0.1

python train-switch.py --gpu 5 --ks 64 --num_experts 16 --lb_alphas 0.3 1
python train-switch.py --gpu 6 --ks 64 --num_experts 16 --lb_alphas 3 10
python train-switch.py --gpu 7 --ks 64 --num_experts 16 --lb_alphas 30 100



python train-switch.py --gpu 4 --ks 64 --num_experts 32 --lb_alphas 0 0.001
python train-switch.py --gpu 5 --ks 64 --num_experts 32 --lb_alphas 0.003 0.01
python train-switch.py --gpu 6 --ks 64 --num_experts 32 --lb_alphas 0.03 0.1

python train-switch.py --gpu 5 --ks 64 --num_experts 32 --lb_alphas 0.3 1
python train-switch.py --gpu 6 --ks 64 --num_experts 32 --lb_alphas 3 10
python train-switch.py --gpu 7 --ks 64 --num_experts 32 --lb_alphas 30 100



cd /data/cb/scratch/amudide/switch_sae
conda activate sae

## even older file


## ADD WANDB NAME TO ARGPARSE FOR SH FILES!

python train-topk.py --gpu 3 --ks 8 16 32 48 --dict_ratio 64
python train-topk.py --gpu 4 --ks 64 96 128 192 --dict_ratio 64

python train-topk.py --gpu 4 --ks 8 16
python train-topk.py --gpu 5 --ks 32 48
python train-topk.py --gpu 6 --ks 64 96
python train-topk.py --gpu 7 --ks 128 192

python train-relu.py --gpu 4 --l1_penalties 5 6.46
python train-relu.py --gpu 5 --l1_penalties 8.34 10.78
python train-relu.py --gpu 6 --l1_penalties 13.92 17.98
python train-relu.py --gpu 7 --l1_penalties 23.23 30
python train-relu.py --gpu 5 --l1_penalties 38.71 50
python train-relu.py --gpu 0 --l1_penalties 56 64
python train-relu.py --gpu 1 --l1_penalties 75 95
python train-relu.py --gpu 2 --l1_penalties 1 2
python train-relu.py --gpu 3 --l1_penalties 3 4

python train-gated.py --gpu 3 --l1_penalties 1 1.46
python train-gated.py --gpu 4 --l1_penalties 2.13 3.11
python train-gated.py --gpu 5 --l1_penalties 4.53 6.62
python train-gated.py --gpu 6 --l1_penalties 9.65 14.09
python train-gated.py --gpu 7 --l1_penalties 20.56 30
python train-gated.py --gpu 4 --l1_penalties 5 6


cd /data/cb/scratch/amudide/switch_sae
conda activate sae