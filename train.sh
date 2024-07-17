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
